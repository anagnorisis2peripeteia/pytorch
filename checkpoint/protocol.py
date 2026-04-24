"""Wire protocol for the checkpoint coordinator.

Single source of truth for the op names, error codes, framing, and header
shapes shared between ``coordinator.py`` and ``coord_client.py``.

High-level model
----------------

N torch processes cooperate on one or more GPUs, coordinated by a central
coordinator. Each process registers under an integer ``rank``. Exactly one
rank holds the "baton" (is actively running on GPU) at a time; the others
are checkpointed to host RAM via ``cuCheckpointProcess*``.

Data exchange uses a single primitive, :func:`prepare`, that expresses any
point-to-point or collective pattern. A ``prepare`` call declares:

  * ``send``:  what tensors (or notifications) this rank is depositing for
    specific peer ranks.
  * ``recv``:  which peer ranks this rank expects tensors (or notifications)
    from.

If all ``recv`` sources have already deposited, the coordinator returns the
data synchronously (the "fast path"); the caller keeps the baton. If any
recv source has not yet deposited, the coordinator replies ``block``; the
caller must call :func:`release_gpu` next, which checkpoints the process,
waits for the coordinator to schedule it, and restores + returns the data
when it is this rank's turn.

Per-pair FIFO ordering is guaranteed: rank A's nth send to B pairs with
rank B's nth recv from A, matching PyTorch's "user ensures collective call
order" contract.

Wire format per message
-----------------------
::

    [ 4 bytes big-endian u32 : header_len ]
    [ header_len bytes       : UTF-8 JSON header ]
    [ 8 bytes big-endian u64 : payload_len ]    # 0 if no payload
    [ payload_len bytes      : concatenated tensor bodies ]

Every response header includes ``{"ok": bool, "error": str | None, ...}``.
When ``ok`` is false, ``error`` is a machine-readable string — see the
``ERR_*`` constants below.

Ops
---
``OP_REGISTER``
    Request:  ``{"op": "register", "rank": int}``
    Response: ``{"ok": true}``
    Registers this connection under ``rank``. Error if already registered.

``OP_WAIT_FOR_TURN``
    Request:  ``{"op": "wait_for_turn"}``
    Response: ``{"ok": true}``
    Bootstrap only. Block until this rank holds the baton. Returns error
    ``"no_peers"`` if all other peers have disconnected.

``OP_PREPARE``
    Request::

        {
            "op": "prepare",
            "send": [{"dsts": [int, ...], "tensor": TensorHeader | null}, ...],
            "recv": [int, ...],  # source ranks
        }

    followed by concatenated tensor bodies (only for non-null send entries,
    in ``send`` order).

    Response — fast path (all recvs already deposited), caller keeps baton::

        {
            "ok": true,
            "block": false,
            "recv": [{"src": int, "tensor": TensorHeader | null}, ...],
        }

    followed by concatenated tensor bodies (only non-null, in ``recv`` order).

    Response — slow path, caller must call release_gpu next::

        {"ok": true, "block": true}

    Errors: ``"mismatch"`` if a peer's pending prepare is inconsistent
    (sent to a peer that didn't declare the recv, or recv declared from a
    peer that isn't sending).

``OP_RELEASE_GPU``
    Request:  ``{"op": "release_gpu"}``
    Response: ``{"ok": true, "recv": [...]}`` + payloads, same shape as
    the fast-path branch of ``prepare``.

    Valid only after a ``prepare`` that returned ``block: true``. Blocks
    server-side until this rank's prepare becomes satisfiable (all recv
    deposits present) AND this rank is scheduled to run. Returns the recv
    data. Client is expected to have checkpointed before sending this
    request and restore immediately on receiving the response.

``OP_DONE``
    Request:  ``{"op": "done"}``
    Response: ``{"ok": true}``
    Client is exiting; coordinator cleans up its state, drops any stale
    mailboxes, and fails any pending prepares on peers with ``peer_gone``.

(See ``CoordClient.prepare`` in ``coord_client.py`` for how to express each
PyTorch collective using this primitive.)

Error codes
-----------
``ERR_NO_PEERS``   — all other clients disconnected; the waiter can't be granted.
``ERR_PEER_GONE``  — a peer we were waiting on (recv src) disappeared.
``ERR_MISMATCH``   — a peer's pending prepare is inconsistent with ours.
"""

import asyncio
import json
import struct
from typing import TypedDict


# ---- Op names ----

OP_REGISTER = "register"
OP_WAIT_FOR_TURN = "wait_for_turn"
OP_PREPARE = "prepare"
OP_RELEASE_GPU = "release_gpu"
OP_DONE = "done"


# ---- Error codes ----

ERR_NO_PEERS = "no_peers"
ERR_PEER_GONE = "peer_gone"
ERR_MISMATCH = "mismatch"


# ---- Header shapes ----


class TensorHeader(TypedDict):
    shape: list[int]
    dtype: str  # e.g. "float32", "bfloat16"
    device: str  # e.g. "cuda:0", "cpu"
    nbytes: int  # payload size for this tensor


class SendEntry(TypedDict):
    dsts: list[int]
    tensor: TensorHeader | None


class RecvEntry(TypedDict):
    src: int
    tensor: TensorHeader | None


# ---- Framing ----


async def read_exact(reader: asyncio.StreamReader, n: int) -> bytes:
    """Read exactly n bytes or raise IncompleteReadError."""
    return await reader.readexactly(n)


async def read_message(reader: asyncio.StreamReader) -> tuple[dict, bytes]:
    """Read one framed message. Returns ``(header_dict, payload_bytes)``."""
    (hdr_len,) = struct.unpack(">I", await read_exact(reader, 4))
    header: dict = json.loads((await read_exact(reader, hdr_len)).decode("utf-8"))
    (payload_len,) = struct.unpack(">Q", await read_exact(reader, 8))
    payload: bytes = await read_exact(reader, payload_len) if payload_len else b""
    return header, payload


async def write_message(
    writer: asyncio.StreamWriter, header: dict, payload: bytes = b""
) -> None:
    """Write one framed message and drain."""
    hdr_bytes = json.dumps(header).encode("utf-8")
    writer.write(struct.pack(">I", len(hdr_bytes)))
    writer.write(hdr_bytes)
    writer.write(struct.pack(">Q", len(payload)))
    if payload:
        writer.write(payload)
    await writer.drain()
