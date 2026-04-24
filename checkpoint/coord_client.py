"""Synchronous client for the coordinator server.

Exposes:

* :class:`CoordClient` — primary API. ``register(rank)``,
  ``wait_for_turn()`` (bootstrap), ``prepare(send, recv)``,
  ``release_gpu()``, ``done()``.

Addressing: pass ``addr`` as ``"uds:/path/to/sock"`` or ``"tcp:host:port"``.
Also reads the ``COORD_ADDR`` env var as the default.

Threading model: the client runs an asyncio event loop in a daemon thread.
Foreground (torch) code calls synchronous methods which dispatch to the
loop via ``asyncio.run_coroutine_threadsafe`` and block on ``.result()``.
No ``asyncio.Queue`` across threads.

Lifecycle invariant: between ``checkpoint_self()`` and ``restore_self()``
the client has no live CUDA. ``prepare()`` and ``release_gpu()`` call those
primitives on the caller's behalf; do not call CUDA-touching ops outside
that window.
"""

import asyncio
import os
import threading
from typing import Any

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


class CoordClientError(RuntimeError):
    pass


class CheckpointedStateError(CoordClientError):
    """Raised if a CUDA-touching op is attempted while checkpointed."""


class PeerGone(CoordClientError):
    """Raised if a peer we were waiting on disappears."""


class NoPeers(CoordClientError):
    """Raised by wait_for_turn when no other client remains to hand over."""


class CollectiveMismatch(CoordClientError):
    """Raised when this rank's prepare is inconsistent with a peer's."""


# ---- Tensor serialization (raw-bytes fast path) ----


def _serialize_tensor(tensor) -> tuple[dict, bytes]:
    """Encode a torch.Tensor as (header, raw_bytes). Caller must have live
    CUDA; tensor is materialized to CPU contiguous. Uses untyped storage
    bytes so bfloat16 and torch-only dtypes roundtrip cleanly."""
    import torch

    if not isinstance(tensor, torch.Tensor):
        raise TypeError(f"expected torch.Tensor, got {type(tensor).__name__}")
    cpu_contig = tensor.detach().contiguous().cpu()
    nbytes = cpu_contig.numel() * cpu_contig.element_size()
    storage = cpu_contig.untyped_storage()
    payload = bytes(storage[:nbytes])
    header = {
        "shape": list(cpu_contig.shape),
        "dtype": str(cpu_contig.dtype).removeprefix("torch."),
        "device": str(tensor.device),
        "nbytes": nbytes,
    }
    return header, payload


def _deserialize_tensor(header: dict, payload: bytes):
    import torch

    dtype = getattr(torch, header["dtype"])
    shape = tuple(header["shape"])
    buf = bytearray(payload)
    numel = 1
    for d in shape:
        numel *= d
    flat = torch.frombuffer(buf, dtype=dtype, count=numel).clone()
    t = flat.reshape(shape)
    target = header.get("device", "cpu")
    if target != "cpu":
        t = t.to(target)
    return t


# ---- Async engine (runs in background thread) ----


class _Engine:
    """Owns the event loop and the server connection. Request-response is
    strictly sequential (one in flight at a time)."""

    def __init__(self, addr: str):
        self.addr = addr
        self.reader: asyncio.StreamReader | None = None
        self.writer: asyncio.StreamWriter | None = None
        self._lock = asyncio.Lock()

    async def connect(self) -> None:
        if self.addr.startswith("uds:"):
            path = self.addr[4:]
            self.reader, self.writer = await asyncio.open_unix_connection(path)
        elif self.addr.startswith("tcp:"):
            host, _, port = self.addr[4:].rpartition(":")
            self.reader, self.writer = await asyncio.open_connection(host, int(port))
        else:
            raise ValueError(f"unrecognized addr {self.addr!r}")

    async def close(self) -> None:
        if self.writer is not None:
            self.writer.close()
            try:
                await self.writer.wait_closed()
            except Exception:
                pass
            self.writer = None
            self.reader = None

    async def rpc(self, header: dict, payload: bytes = b"") -> tuple[dict, bytes]:
        async with self._lock:
            if self.writer is None:
                raise CoordClientError("not connected")
            await write_message(self.writer, header, payload)
            try:
                resp_hdr, resp_payload = await read_message(self.reader)
            except asyncio.IncompleteReadError as e:
                raise PeerGone("coordinator closed the connection") from e
            if not resp_hdr.get("ok"):
                err = resp_hdr.get("error") or "request failed"
                detail = resp_hdr.get("detail")
                if err == ERR_NO_PEERS:
                    raise NoPeers(err)
                if err == ERR_PEER_GONE:
                    raise PeerGone(err)
                if err == ERR_MISMATCH:
                    raise CollectiveMismatch(detail or err)
                raise CoordClientError(err)
            return resp_hdr, resp_payload


# ---- CoordClient ----


class CoordClient:
    def __init__(self, addr: str | None = None):
        self.addr = addr or os.environ.get("COORD_ADDR")
        if not self.addr:
            raise CoordClientError("addr not provided and COORD_ADDR env not set")
        self._engine = _Engine(self.addr)
        self._loop: asyncio.AbstractEventLoop | None = None
        self._thread: threading.Thread | None = None
        self._ready = threading.Event()
        self._rank: int | None = None
        self._checkpointed = False
        self._start_loop()
        self._run_coro(self._engine.connect())

    # ---- Loop plumbing ----

    def _start_loop(self) -> None:
        def run_loop():
            self._loop = asyncio.new_event_loop()
            asyncio.set_event_loop(self._loop)
            self._ready.set()
            try:
                self._loop.run_forever()
            finally:
                self._loop.close()

        self._thread = threading.Thread(
            target=run_loop, name="CoordClient", daemon=True
        )
        self._thread.start()
        self._ready.wait()

    def _run_coro(self, coro, *, timeout: float | None = None):
        if self._loop is None:
            raise CoordClientError("loop not running")
        fut = asyncio.run_coroutine_threadsafe(coro, self._loop)
        try:
            return fut.result(timeout=timeout)
        except KeyboardInterrupt:
            self._loop.call_soon_threadsafe(fut.cancel)
            raise

    def _guard_cuda(self) -> None:
        if self._checkpointed:
            raise CheckpointedStateError("CUDA-touching op called while checkpointed")

    # ---- Public API ----

    def register(self, rank: int) -> None:
        if not isinstance(rank, int):
            raise TypeError(f"rank must be int, got {type(rank).__name__}")
        self._run_coro(self._engine.rpc({"op": OP_REGISTER, "rank": rank}))
        self._rank = rank

    def wait_for_turn(self) -> None:
        """Bootstrap: block until this rank holds the baton. Call exactly
        once after register() for ranks that aren't the initial holder.
        """
        self._run_coro(self._engine.rpc({"op": OP_WAIT_FOR_TURN}))

    def prepare(
        self,
        send: dict[tuple[int, ...], Any],
        recv: tuple[int, ...] | list[int],
    ) -> dict[int, Any] | None:
        """Declare this rank's collective step.

        Args:
            send: mapping from tuple-of-destination-ranks to tensor OR None.
                  A ``None`` value is a notification-only deposit.
            recv: iterable of source ranks this rank expects to receive from.

        Returns:
            ``dict[src_rank -> tensor | None]`` if the fast path hit (all
            recv sources have already deposited); caller keeps the baton.
            ``None`` if the caller must follow up with :meth:`release_gpu`.

        Raises:
            :class:`CollectiveMismatch` if any peer's pending prepare is
            inconsistent with this call's send/recv spec.

        Collective mapping
        ------------------
        Every PyTorch collective maps to a single ``prepare`` call. Below,
        ``self`` is this rank's id, ``others`` is the tuple of peer ranks
        participating in the group. ``None`` as a ``send`` value is a
        notification-only deposit — the matching ``recv`` returns ``None``.

        =============================  =================================================  =================================  =============================
        Collective                     ``send``                                           ``recv``                           post-local work
        =============================  =================================================  =================================  =============================
        ``send(dst, t)``               ``{(dst,): t}``                                    ``()``                             —
        ``recv(src)``                  ``{}``                                             ``(src,)``                         —
        ``broadcast(t, src)`` at src   ``{tuple(others): t}``                             ``()``                             —
        ``broadcast`` at non-src       ``{}``                                             ``(src,)``                         —
        ``reduce(t, dst)`` non-dst     ``{(dst,): t}``                                    ``()``                             —
        ``reduce`` at dst              ``{}``                                             ``tuple(others)``                  local reduce
        ``scatter(list, src)`` src     ``{(i,): list[i] for i != self}``                  ``()``                             use ``list[self]``
        ``scatter`` non-src            ``{}``                                             ``(src,)``                         —
        ``gather(t, dst)`` non-dst     ``{(dst,): t}``                                    ``()``                             —
        ``gather`` at dst              ``{}``                                             ``tuple(others)``                  assemble list
        ``all_gather(t)``              ``{tuple(others): t}``                             ``tuple(others)``                  concat
        ``all_reduce(t)``              ``{tuple(others): t}``                             ``tuple(others)``                  local reduce
        ``reduce_scatter(chunks)``     ``{(i,): chunks[i] for i != self}``                ``tuple(others)``                  reduce with ``chunks[self]``
        ``all_to_all(chunks)``         ``{(j,): chunks[j] for j != self}``                ``tuple(others)``                  —
        ``barrier()``                  ``{tuple(others): None}``                          ``tuple(others)``                  —
        =============================  =================================================  =================================  =============================
        """
        self._guard_cuda()
        import torch  # ensure torch is importable for serialization

        # Build the wire-form send list; gather payloads.
        send_entries = []
        payloads: list[bytes] = []
        for dsts, tensor in send.items():
            if not isinstance(dsts, tuple):
                raise TypeError("send keys must be tuples of destination ranks")
            if tensor is None:
                send_entries.append({"dsts": list(dsts), "tensor": None})
            elif isinstance(tensor, torch.Tensor):
                hdr, body = _serialize_tensor(tensor)
                send_entries.append({"dsts": list(dsts), "tensor": hdr})
                payloads.append(body)
            else:
                raise TypeError(
                    f"send values must be Tensor or None, got {type(tensor).__name__}"
                )

        req = {"op": OP_PREPARE, "send": send_entries, "recv": list(recv)}
        resp_hdr, resp_payload = self._run_coro(
            self._engine.rpc(req, b"".join(payloads))
        )
        if resp_hdr.get("block"):
            return None
        return self._decode_recv(resp_hdr.get("recv") or [], resp_payload)

    def release_gpu(self) -> dict[int, Any]:
        """Checkpoint, wait for the coordinator to schedule this rank, then
        restore and return the data this rank's pending prepare was waiting
        on. Must follow a :meth:`prepare` that returned ``None``.
        """
        from cuda_checkpoint import checkpoint_self, restore_self

        import torch

        torch.cuda.synchronize()
        torch.cuda.empty_cache()
        checkpoint_self()
        self._checkpointed = True
        try:
            resp_hdr, resp_payload = self._run_coro(
                self._engine.rpc({"op": OP_RELEASE_GPU})
            )
        finally:
            # Whether server replied ok or error, we need to restore to
            # keep the process usable.
            restore_self()
            self._checkpointed = False
        return self._decode_recv(resp_hdr.get("recv") or [], resp_payload)

    def done(self) -> None:
        try:
            self._run_coro(self._engine.rpc({"op": OP_DONE}))
        finally:
            self.close()

    def close(self) -> None:
        if self._loop is None:
            return
        try:
            self._run_coro(self._engine.close(), timeout=2.0)
        except Exception:
            pass
        self._loop.call_soon_threadsafe(self._loop.stop)
        self._thread.join(timeout=2.0)
        self._loop = None

    # ---- Helpers ----

    def _decode_recv(self, entries: list[dict], payload: bytes) -> dict[int, Any]:
        out: dict[int, Any] = {}
        offset = 0
        for entry in entries:
            src = entry["src"]
            hdr = entry.get("tensor")
            if hdr is None:
                out[src] = None
            else:
                nbytes = hdr["nbytes"]
                body = payload[offset : offset + nbytes]
                offset += nbytes
                out[src] = _deserialize_tensor(hdr, body)
        return out
