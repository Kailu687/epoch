"""Run an experiment plan in a separate worker process with a typed event stream and real cancellation."""
from __future__ import annotations

import multiprocessing as mp
import queue
import time

import psutil

from .runner import run_plan

TERMINAL = ("finished", "failed", "cancelled")


def _worker(plan, q, cancel_evt):
    run_plan(plan, q.put, cancel_evt.is_set)


class Job:
    def __init__(self, plan: dict):
        ctx = mp.get_context("spawn")
        self.plan = plan
        self._q = ctx.Queue()
        self._cancel = ctx.Event()
        self._proc = ctx.Process(target=_worker, args=(plan, self._q, self._cancel), daemon=True)
        self.done = False
        self._last_res = 0.0
        self._ps = None

    def start(self):
        self._proc.start()
        self._ps = psutil.Process(self._proc.pid)
        psutil.cpu_percent(None)
        return self

    def cancel(self, grace: float = 1.5):
        self._cancel.set()
        t0 = time.time()
        while self._proc.is_alive() and time.time() - t0 < grace:
            time.sleep(0.05)
        killed = self._proc.is_alive()
        if killed:
            self._terminate_tree()
        self._proc.join(2)
        events = self._drain()
        if killed or not any(e["type"] in TERMINAL for e in events):
            events.append({"type": "cancelled", "ts": time.time()})
        self.done = True
        return events

    def _terminate_tree(self):
        try:
            for ch in self._ps.children(recursive=True):
                ch.kill()
        except psutil.Error:
            pass
        self._proc.kill()
        self._proc.join(2)

    def _drain(self):
        out = []
        while True:
            try:
                out.append(self._q.get_nowait())
            except queue.Empty:
                return out

    def poll(self, timeout: float = 0.2) -> list[dict]:
        """Return new events (non-blocking after `timeout`). Includes periodic resource samples."""
        events = []
        try:
            events.append(self._q.get(timeout=timeout))
            events += self._drain()
        except queue.Empty:
            pass
        now = time.time()
        if not self.done and now - self._last_res > 0.5:
            self._last_res = now
            try:
                rss = self._ps.memory_info().rss + sum(c.memory_info().rss for c in self._ps.children(recursive=True))
                events.append({"type": "resource", "ts": now, "cpu_percent": psutil.cpu_percent(None),
                               "ram_percent": psutil.virtual_memory().percent, "worker_rss_mb": round(rss / 1048576, 1)})
            except psutil.Error:
                pass
        if any(e["type"] in TERMINAL for e in events):
            self.done = True
        elif not self._proc.is_alive() and self._q.empty() and not self.done:
            events += self._drain()
            if not any(e["type"] in TERMINAL for e in events):
                events.append({"type": "failed", "ts": now, "reason": "Worker process exited unexpectedly."})
            self.done = True
        return events

    def wait(self, on_event=None, poll: float = 0.2) -> list[dict]:
        allev = []
        while not self.done:
            for e in self.poll(poll):
                allev.append(e)
                if on_event:
                    on_event(e)
        self._proc.join(5)
        return allev
