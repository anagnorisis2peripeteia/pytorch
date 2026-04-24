"""File-token Baton: a simple two-process GPU rendezvous used by the
demo/bench scripts that predate the coordinator.

Two processes agree on a shared directory and their own/peer names. A
single ``token`` file in that directory names the process currently
entitled to the GPU. The non-holder polls the token; when the holder
calls :meth:`Baton.release`, the current process is checkpointed
(VRAM freed) and the token is atomically handed to the peer. The peer's
:meth:`Baton.acquire` observes the token flip, restores its own CUDA
state, and continues.

This is a DEMO abstraction. The real system uses the coordinator
(:mod:`coord_client`) which has no user-visible "baton" concept.
"""

import os
import pathlib
import sys
import time


sys.path.insert(
    0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "checkpoint")
)
from cuda_checkpoint import checkpoint_self, restore_self


_DONE_TOKEN = "__DONE__"


class Baton:
    """Token-file rendezvous. Exactly one process holds the token at a time.

    Invariants:
      - ``token`` file contains the name of the process currently entitled
        to use the GPU.
      - A process may call :meth:`acquire` to block until it holds the
        token.
      - A process holding the token calls :meth:`release` to checkpoint
        itself and hand the token to its peer.
      - A process can call :meth:`done` to signal it is exiting; the
        peer's :meth:`acquire` returns ``False`` in that case.
    """

    def __init__(
        self,
        baton_dir,
        my_name: str,
        peer_name: str,
        *,
        poll_interval: float = 0.05,
    ) -> None:
        self.dir = pathlib.Path(baton_dir)
        self.dir.mkdir(parents=True, exist_ok=True)
        self.token = self.dir / "token"
        self.my_name = my_name
        self.peer_name = peer_name
        self.poll = poll_interval
        self._checkpointed = False

    def _read_token(self):
        try:
            return self.token.read_text().strip()
        except FileNotFoundError:
            return None

    def _write_token(self, name: str) -> None:
        tmp = self.token.with_suffix(".tmp")
        tmp.write_text(name)
        tmp.replace(self.token)

    def acquire(self, timeout=None) -> bool:
        """Block until this process holds the token. Returns False if peer signaled done."""
        deadline = None if timeout is None else time.monotonic() + timeout
        while True:
            cur = self._read_token()
            if cur == self.my_name:
                break
            if cur == _DONE_TOKEN:
                return False
            if deadline is not None and time.monotonic() > deadline:
                raise TimeoutError(f"Baton.acquire timed out (token={cur!r})")
            time.sleep(self.poll)
        if self._checkpointed:
            restore_self()
            self._checkpointed = False
        return True

    def release(self) -> None:
        """Checkpoint this process and hand the token to the peer."""
        import torch

        torch.cuda.synchronize()
        torch.cuda.empty_cache()
        checkpoint_self()
        self._checkpointed = True
        self._write_token(self.peer_name)

    def done(self) -> None:
        """Signal peer to stop. Call before exit when this process is finished."""
        self._write_token(_DONE_TOKEN)

    def init_as_holder(self) -> None:
        """Seed the token with this process's name. Call once, before both
        processes start their loop."""
        self._write_token(self.my_name)
