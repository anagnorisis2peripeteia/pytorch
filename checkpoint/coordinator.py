"""Coordinator server for torch-process orchestration.

Clients register by integer rank. Exactly one rank holds the baton at a
time. Data exchange goes through :func:`prepare` (see ``protocol.py``).
Wire framing and op contracts are defined there.

Run as::

    python coordinator.py                          # UDS at default path
    python coordinator.py --socket /tmp/foo.sock   # explicit UDS path
    python coordinator.py --tcp-port 0             # TCP; 0 picks a free port

Add ``--initial-rank N`` to pre-seed the baton so the client registering
as rank ``N`` is guaranteed to hold first regardless of connection order.

On startup, prints the bound address on ONE line of stdout so launchers
can capture it::

    ADDR uds:/tmp/foo.sock
    ADDR tcp:127.0.0.1:54321
"""

import argparse
import asyncio
import collections
import os
import signal
import sys

from protocol import (
    ERR_MISMATCH,
    ERR_NO_PEERS,
    ERR_PEER_GONE,
    OP_DONE,
    OP_PREPARE,
    OP_REGISTER,
    OP_RELEASE_GPU,
    OP_WAIT_FOR_TURN,
    read_message,
    write_message,
)


# ---- Internal signals ----


class _NoPeers(Exception):
    """Internal signal: no peers remain to hand the baton to."""


class _PeerGone(Exception):
    """Internal signal: a peer we were waiting on disappeared."""


# ---- Pending state ----


class _Prepare:
    """A rank's outstanding prepare() call. Holds the parsed send/recv specs
    plus the futures the coordinator has to complete when scheduling fires."""

    def __init__(self, send: list, recv: list[int]):
        # send: list of {"dsts": [int,...], "tensor": TensorHeader|None, "payload": bytes}
        self.send = send
        self.recv: list[int] = list(recv)
        # Populated as recv sources deposit:
        # dict[src_rank] -> (tensor_header|None, payload_bytes)
        self.received: dict[int, tuple[dict | None, bytes]] = {}
        # Set when the client calls release_gpu (which must follow a block=true prepare).
        # Resolved by the coordinator once (a) all recvs satisfied AND
        # (b) this rank is at the head of the eligible queue AND
        # (c) no one else holds the baton.
        self.release_future: asyncio.Future | None = None


# ---- Coordinator ----


class Coordinator:
    def __init__(self, initial_rank: int | None = None):
        # Registered clients: rank -> writer.
        self.clients: dict[int, asyncio.StreamWriter] = {}
        self.writer_to_rank: dict[asyncio.StreamWriter, int] = {}

        # Baton pre-seed: see protocol docs.
        self.current_holder: int | None = initial_rank

        # Bootstrap waiters (OP_WAIT_FOR_TURN): deque[(rank, future)].
        self.waiters: collections.deque[tuple[int, asyncio.Future]] = (
            collections.deque()
        )

        # Active prepare per rank (one at a time).
        self.pending: dict[int, _Prepare] = {}

        # Mailboxes: (src, dst) -> FIFO of deposits made by src for dst.
        # Each deposit is (tensor_header_or_None, payload_bytes).
        self.mailboxes: dict[
            tuple[int, int], collections.deque[tuple[dict | None, bytes]]
        ] = {}

        # Eligible queue: ranks whose pending prepare is satisfied and who
        # are waiting for the GPU. FIFO.
        self.eligible: collections.deque[int] = collections.deque()

    def log(self, *args):
        print("[coord]", *args, file=sys.stderr, flush=True)

    # ---- Baton helpers ----

    async def _grant_baton(self, rank: int) -> None:
        assert self.current_holder is None
        self.current_holder = rank
        self.log(f"granted baton -> {rank}")

    async def _try_schedule_next(self) -> None:
        """Called when the baton becomes free. Prefers eligible (data-ready)
        clients over bootstrap waiters."""
        if self.current_holder is not None:
            return
        # Eligible first.
        while self.eligible:
            rank = self.eligible.popleft()
            prep = self.pending.get(rank)
            if prep is None or prep.release_future is None:
                continue  # stale
            await self._grant_baton(rank)
            # Deliver recv data via the release_gpu future.
            if not prep.release_future.done():
                prep.release_future.set_result(prep.received)
            # The pending prepare is now consumed; remove it.
            self.pending.pop(rank, None)
            return
        # Fall back to bootstrap waiters.
        while self.waiters:
            rank, fut = self.waiters.popleft()
            if rank not in self.clients:
                continue
            await self._grant_baton(rank)
            if not fut.done():
                fut.set_result(None)
            return

    # ---- Op: register ----

    async def handle_register(self, hdr, writer) -> None:
        rank = hdr["rank"]
        if not isinstance(rank, int):
            await write_message(writer, {"ok": False, "error": "rank must be int"})
            return
        if rank in self.clients:
            await write_message(
                writer, {"ok": False, "error": f"rank {rank} already registered"}
            )
            return
        self.clients[rank] = writer
        self.writer_to_rank[writer] = rank
        self.log(f"registered rank {rank}")
        await write_message(writer, {"ok": True, "error": None})

    # ---- Op: wait_for_turn (bootstrap) ----

    async def handle_wait_for_turn(self, hdr, writer) -> None:
        rank = self.writer_to_rank.get(writer)
        if rank is None:
            await write_message(writer, {"ok": False, "error": "not registered"})
            return
        if self.current_holder is None:
            await self._grant_baton(rank)
            await write_message(writer, {"ok": True, "error": None})
            return
        if self.current_holder == rank:
            # Pre-seeded initial-holder claim.
            await write_message(writer, {"ok": True, "error": None})
            return
        fut = asyncio.get_running_loop().create_future()
        self.waiters.append((rank, fut))
        try:
            await fut
        except _NoPeers:
            await write_message(writer, {"ok": False, "error": ERR_NO_PEERS})
            return
        except asyncio.CancelledError:
            self.waiters = collections.deque(
                (r, f) for r, f in self.waiters if f is not fut
            )
            raise
        await write_message(writer, {"ok": True, "error": None})

    # ---- Op: prepare ----

    async def handle_prepare(self, hdr, payload: bytes, writer) -> None:
        rank = self.writer_to_rank.get(writer)
        if rank is None:
            await write_message(writer, {"ok": False, "error": "not registered"})
            return
        if rank in self.pending:
            await write_message(
                writer, {"ok": False, "error": "already has pending prepare"}
            )
            return

        send_spec = hdr.get("send") or []
        recv_spec = hdr.get("recv") or []

        # Split the concatenated payload into per-send-entry slices.
        # Null-tensor send entries contribute no bytes.
        offset = 0
        send_entries = []
        for entry in send_spec:
            tensor_hdr = entry.get("tensor")
            dsts = entry["dsts"]
            if tensor_hdr is None:
                send_entries.append({"dsts": dsts, "tensor": None, "payload": b""})
            else:
                nbytes = tensor_hdr["nbytes"]
                body = payload[offset : offset + nbytes]
                offset += nbytes
                send_entries.append(
                    {"dsts": dsts, "tensor": tensor_hdr, "payload": body}
                )

        # Mismatch detection — check against peers' pending prepares.
        mismatch = self._check_mismatch(rank, send_entries, recv_spec)
        if mismatch is not None:
            await write_message(
                writer, {"ok": False, "error": ERR_MISMATCH, "detail": mismatch}
            )
            return

        # Deposit sends.
        for entry in send_entries:
            for dst in entry["dsts"]:
                self.mailboxes.setdefault((rank, dst), collections.deque()).append(
                    (entry["tensor"], entry["payload"])
                )

        # Try to satisfy this rank's recv immediately.
        received: dict[int, tuple[dict | None, bytes]] = {}
        for src in recv_spec:
            mbox = self.mailboxes.get((src, rank))
            if mbox:
                received[src] = mbox.popleft()
                if not mbox:
                    self.mailboxes.pop((src, rank), None)
            else:
                received.clear()
                break

        if len(received) == len(recv_spec):
            # Fast path. Respond with data; rank keeps baton.
            await write_message(
                writer,
                *self._build_prepare_fast_response(recv_spec, received),
            )
        else:
            # Slow path. Record pending prepare; client must call release_gpu.
            prep = _Prepare(send_entries, recv_spec)
            # Seed partial receptions if any.
            for src in recv_spec:
                mbox = self.mailboxes.get((src, rank))
                if mbox:
                    prep.received[src] = mbox.popleft()
                    if not mbox:
                        self.mailboxes.pop((src, rank), None)
            self.pending[rank] = prep
            await write_message(writer, {"ok": True, "block": True, "error": None})

        # This rank's sends may have satisfied OTHER pending prepares. Check
        # and enqueue them as eligible.
        self._sweep_eligible_after_send(rank)

    def _check_mismatch(
        self, rank: int, send_entries: list, recv_spec: list[int]
    ) -> str | None:
        # For each dst in send_entries: if dst has a pending prepare whose
        # recv doesn't include rank, mismatch.
        for entry in send_entries:
            for dst in entry["dsts"]:
                pd = self.pending.get(dst)
                if pd is not None and rank not in pd.recv:
                    return f"rank {rank} sent to {dst}; {dst} does not recv from {rank}"
        # For each src in recv_spec: if src has a pending prepare whose
        # send doesn't target rank AND mailbox (src, rank) is empty, mismatch.
        for src in recv_spec:
            ps = self.pending.get(src)
            if ps is None:
                continue
            targets = set()
            for entry in ps.send:
                targets.update(entry["dsts"])
            if rank in targets:
                continue
            if self.mailboxes.get((src, rank)):
                continue
            return f"rank {rank} recv from {src}; {src} does not send to {rank}"
        return None

    def _build_prepare_fast_response(
        self,
        recv_spec: list[int],
        received: dict[int, tuple[dict | None, bytes]],
    ) -> tuple[dict, bytes]:
        entries = []
        bodies: list[bytes] = []
        for src in recv_spec:
            hdr, body = received[src]
            entries.append({"src": src, "tensor": hdr})
            if body:
                bodies.append(body)
        return (
            {"ok": True, "block": False, "error": None, "recv": entries},
            b"".join(bodies),
        )

    def _sweep_eligible_after_send(self, just_sent_rank: int) -> None:
        """After a deposit, some pending prepares may now be fully
        satisfied. Enqueue them as eligible (preserving order of discovery)."""
        for rank, prep in list(self.pending.items()):
            if prep.release_future is None:
                continue  # client hasn't released yet; not eligible to run
            # Pull any newly-available deposits into prep.received.
            for src in prep.recv:
                if src in prep.received:
                    continue
                mbox = self.mailboxes.get((src, rank))
                if mbox:
                    prep.received[src] = mbox.popleft()
                    if not mbox:
                        self.mailboxes.pop((src, rank), None)
            if len(prep.received) == len(prep.recv) and rank not in self.eligible:
                self.eligible.append(rank)
                self.log(f"rank {rank} eligible")

    # ---- Op: release_gpu ----

    async def handle_release_gpu(self, hdr, writer) -> None:
        rank = self.writer_to_rank.get(writer)
        if rank is None:
            await write_message(writer, {"ok": False, "error": "not registered"})
            return
        prep = self.pending.get(rank)
        if prep is None:
            await write_message(
                writer, {"ok": False, "error": "no pending prepare to release"}
            )
            return
        # Mark this rank as released (no longer holds the baton).
        if self.current_holder == rank:
            self.current_holder = None
        # Attach future for scheduling callback.
        fut = asyncio.get_running_loop().create_future()
        prep.release_future = fut
        # If this rank's prepare is already fully satisfied (deposits arrived
        # between prepare and release_gpu), enqueue eligible.
        for src in prep.recv:
            if src in prep.received:
                continue
            mbox = self.mailboxes.get((src, rank))
            if mbox:
                prep.received[src] = mbox.popleft()
                if not mbox:
                    self.mailboxes.pop((src, rank), None)
        if len(prep.received) == len(prep.recv) and rank not in self.eligible:
            self.eligible.append(rank)

        # Try to schedule someone (possibly this rank if eligible).
        await self._try_schedule_next()

        try:
            received = await fut
        except _PeerGone:
            await write_message(writer, {"ok": False, "error": ERR_PEER_GONE})
            return
        except _NoPeers:
            await write_message(writer, {"ok": False, "error": ERR_NO_PEERS})
            return
        except asyncio.CancelledError:
            raise

        await write_message(
            writer, *self._build_prepare_fast_response(prep.recv, received)
        )

    # ---- Op: done ----

    async def handle_done(self, hdr, writer) -> None:
        await self._cleanup_client(writer)
        await write_message(writer, {"ok": True, "error": None})

    # ---- Connection cleanup ----

    async def _cleanup_client(self, writer: asyncio.StreamWriter) -> None:
        rank = self.writer_to_rank.pop(writer, None)
        if rank is None:
            return
        self.clients.pop(rank, None)

        # Drop from waiters / eligible / pending.
        self.waiters = collections.deque((r, f) for r, f in self.waiters if r != rank)
        self.eligible = collections.deque(r for r in self.eligible if r != rank)

        prep = self.pending.pop(rank, None)
        if prep is not None and prep.release_future is not None:
            if not prep.release_future.done():
                prep.release_future.cancel()

        # Drop mailboxes whose dst is this rank (nobody will consume).
        for k in [k for k in self.mailboxes if k[1] == rank]:
            self.mailboxes.pop(k, None)

        # Fail other ranks' pending prepares that expected from this rank
        # with PeerGone.
        for other_rank, other_prep in list(self.pending.items()):
            if rank in other_prep.recv and rank not in other_prep.received:
                if (
                    other_prep.release_future is not None
                    and not other_prep.release_future.done()
                ):
                    other_prep.release_future.set_exception(_PeerGone())
                self.pending.pop(other_rank, None)
                # Drop from eligible if enqueued by mistake.
                self.eligible = collections.deque(
                    r for r in self.eligible if r != other_rank
                )

        # If holder vanished, free the baton and try to schedule.
        if self.current_holder == rank:
            self.log(f"holder {rank} vanished; releasing baton")
            self.current_holder = None
            await self._try_schedule_next()

        # If no peers remain and there are waiters, fail them with no_peers.
        if len(self.clients) == 0:
            for _r, fut in list(self.waiters):
                if not fut.done():
                    fut.set_exception(_NoPeers())
            self.waiters.clear()

    async def handle_connection(
        self, reader: asyncio.StreamReader, writer: asyncio.StreamWriter
    ) -> None:
        peer = writer.get_extra_info("peername") or writer.get_extra_info("sockname")
        self.log(f"connection from {peer}")
        try:
            while True:
                try:
                    header, payload = await read_message(reader)
                except asyncio.IncompleteReadError:
                    break
                op = header.get("op")
                try:
                    if op == OP_REGISTER:
                        await self.handle_register(header, writer)
                    elif op == OP_WAIT_FOR_TURN:
                        await self.handle_wait_for_turn(header, writer)
                    elif op == OP_PREPARE:
                        await self.handle_prepare(header, payload, writer)
                    elif op == OP_RELEASE_GPU:
                        await self.handle_release_gpu(header, writer)
                    elif op == OP_DONE:
                        await self.handle_done(header, writer)
                        break
                    else:
                        await write_message(
                            writer, {"ok": False, "error": f"unknown op {op!r}"}
                        )
                except Exception as e:
                    self.log(f"op {op!r} raised: {e!r}")
                    try:
                        await write_message(writer, {"ok": False, "error": repr(e)})
                    except Exception:
                        break
        finally:
            await self._cleanup_client(writer)
            try:
                writer.close()
                await writer.wait_closed()
            except Exception:
                pass


# ---- Entry point ----


async def serve(coord: Coordinator, socket_path: str | None, tcp_port: int | None):
    if tcp_port is not None:
        server = await asyncio.start_server(
            coord.handle_connection, host="127.0.0.1", port=tcp_port
        )
        bound = server.sockets[0].getsockname()
        print(f"ADDR tcp:{bound[0]}:{bound[1]}", flush=True)
    else:
        assert socket_path
        if os.path.exists(socket_path):
            os.unlink(socket_path)
        server = await asyncio.start_unix_server(
            coord.handle_connection, path=socket_path
        )
        print(f"ADDR uds:{socket_path}", flush=True)

    loop = asyncio.get_running_loop()
    stop_event = asyncio.Event()
    for sig in (signal.SIGINT, signal.SIGTERM):
        loop.add_signal_handler(sig, stop_event.set)

    async with server:
        stop_task = asyncio.create_task(stop_event.wait())
        serve_task = asyncio.create_task(server.serve_forever())
        _done, pending = await asyncio.wait(
            {stop_task, serve_task}, return_when=asyncio.FIRST_COMPLETED
        )
        for t in pending:
            t.cancel()
        server.close()
        await server.wait_closed()


def main():
    ap = argparse.ArgumentParser()
    g = ap.add_mutually_exclusive_group()
    g.add_argument(
        "--socket",
        help="UDS path (default: $XDG_RUNTIME_DIR/coord.sock or /tmp/coord.sock)",
    )
    g.add_argument(
        "--tcp-port", type=int, help="Bind TCP on 127.0.0.1; 0 picks a free port"
    )
    ap.add_argument(
        "--initial-rank",
        type=int,
        help="Pre-seed this rank as the initial baton holder.",
    )
    args = ap.parse_args()

    coord = Coordinator(initial_rank=args.initial_rank)
    if args.tcp_port is None:
        socket_path = args.socket or os.path.join(
            os.environ.get("XDG_RUNTIME_DIR") or "/tmp", "coord.sock"
        )
        asyncio.run(serve(coord, socket_path=socket_path, tcp_port=None))
    else:
        asyncio.run(serve(coord, socket_path=None, tcp_port=args.tcp_port))


if __name__ == "__main__":
    main()
