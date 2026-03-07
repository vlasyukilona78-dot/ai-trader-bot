from __future__ import annotations

import asyncio
import time
from concurrent.futures import ThreadPoolExecutor
from typing import Callable


class AsyncRetrainScheduler:
    def __init__(
        self,
        train_callable: Callable[[int, bool, str], bool],
        reload_models_callable: Callable[[], bool],
        retrain_interval: int,
        model_dir: str,
    ):
        self._train_callable = train_callable
        self._reload_models_callable = reload_models_callable
        self._retrain_interval = int(retrain_interval)
        self._model_dir = model_dir
        self._executor = ThreadPoolExecutor(max_workers=1)
        self._future = None
        self._last_check_ts = 0.0
        self._check_interval = max(30, min(300, int(self._retrain_interval / 4) if self._retrain_interval > 0 else 60))

    async def tick(self):
        now_ts = time.time()

        if self._future is not None and self._future.done():
            try:
                retrained = bool(self._future.result())
                if retrained:
                    self._reload_models_callable()
            except Exception:
                pass
            self._future = None

        if self._future is None and (now_ts - self._last_check_ts) >= self._check_interval:
            loop = asyncio.get_running_loop()
            self._future = loop.run_in_executor(
                self._executor,
                self._train_callable,
                self._retrain_interval,
                False,
                self._model_dir,
            )
            self._last_check_ts = now_ts

    def shutdown(self):
        try:
            self._executor.shutdown(wait=False, cancel_futures=True)
        except Exception:
            pass
