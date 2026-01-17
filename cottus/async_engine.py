import asyncio
import threading
import queue
import uuid
from dataclasses import dataclass, field
from typing import Optional, List, Dict, AsyncIterator, Any
from enum import Enum

try:
    import _cottus_C
except ImportError:
    _cottus_C = None


class RequestStatus(Enum):
    PENDING = "pending"
    RUNNING = "running"
    FINISHED = "finished"
    CANCELLED = "cancelled"


@dataclass
class AsyncRequest:
    request_id: str
    prompt_ids: List[int]
    max_tokens: int
    temperature: float = 0.0
    top_k: int = 50
    stream: bool = False
    status: RequestStatus = RequestStatus.PENDING
    output_ids: List[int] = field(default_factory=list)
    stop_reason: int = 0
    result_future: Optional[asyncio.Future] = None
    output_queue: Optional[asyncio.Queue] = None


class AsyncEngine:
    def __init__(self, config: Any, weight_ptrs: Dict[str, int]):
        if _cottus_C is None:
            raise RuntimeError("_cottus_C extension not available")

        self._engine = _cottus_C.Engine(config, weight_ptrs)
        self._config = config

        self._pending_queue: queue.Queue = queue.Queue()
        self._active_requests: Dict[str, AsyncRequest] = {}
        self._lock = threading.Lock()

        self._running = False
        self._step_thread: Optional[threading.Thread] = None
        self._loop: Optional[asyncio.AbstractEventLoop] = None

    def start(self):
        if self._running:
            return
        self._running = True
        self._loop = asyncio.get_event_loop()
        self._step_thread = threading.Thread(target=self._step_loop, daemon=True)
        self._step_thread.start()

    def stop(self):
        self._running = False
        if self._step_thread:
            self._step_thread.join(timeout=5.0)
            self._step_thread = None

    def _step_loop(self):
        while self._running:
            self._process_pending()
            self._engine.step()
            self._collect_finished()
            if not self._engine.has_active_requests() and self._pending_queue.empty():
                threading.Event().wait(timeout=0.001)

    def _process_pending(self):
        while not self._pending_queue.empty():
            try:
                req = self._pending_queue.get_nowait()
                internal_req = _cottus_C.Request(
                    int(req.request_id.split("-")[-1], 16) % (2**31),
                    req.prompt_ids,
                    req.max_tokens
                )
                internal_req.temperature = req.temperature
                internal_req.top_k = req.top_k
                self._engine.add_request(internal_req)
                req.status = RequestStatus.RUNNING
            except queue.Empty:
                break

    def _collect_finished(self):
        finished = self._engine.pull_finished_requests()
        for internal_req in finished:
            req_id = None
            with self._lock:
                for rid, req in self._active_requests.items():
                    internal_id = int(rid.split("-")[-1], 16) % (2**31)
                    if internal_id == internal_req.id:
                        req_id = rid
                        break

            if req_id:
                with self._lock:
                    req = self._active_requests.get(req_id)
                    if req:
                        req.output_ids = list(internal_req.generated_ids)
                        req.stop_reason = internal_req.stop_reason
                        req.status = RequestStatus.FINISHED

                        if req.stream and req.output_queue:
                            self._loop.call_soon_threadsafe(
                                req.output_queue.put_nowait, None
                            )
                        elif req.result_future:
                            self._loop.call_soon_threadsafe(
                                req.result_future.set_result, req.output_ids
                            )

    async def generate(
        self,
        prompt_ids: List[int],
        max_tokens: int,
        temperature: float = 0.0,
        top_k: int = 50
    ) -> List[int]:
        request_id = str(uuid.uuid4())
        future = self._loop.create_future()

        req = AsyncRequest(
            request_id=request_id,
            prompt_ids=prompt_ids,
            max_tokens=max_tokens,
            temperature=temperature,
            top_k=top_k,
            stream=False,
            result_future=future
        )

        with self._lock:
            self._active_requests[request_id] = req

        self._pending_queue.put(req)

        try:
            result = await future
            return result
        finally:
            with self._lock:
                self._active_requests.pop(request_id, None)

    async def generate_stream(
        self,
        prompt_ids: List[int],
        max_tokens: int,
        temperature: float = 0.0,
        top_k: int = 50
    ) -> AsyncIterator[int]:
        request_id = str(uuid.uuid4())
        output_queue: asyncio.Queue = asyncio.Queue()

        req = AsyncRequest(
            request_id=request_id,
            prompt_ids=prompt_ids,
            max_tokens=max_tokens,
            temperature=temperature,
            top_k=top_k,
            stream=True,
            output_queue=output_queue
        )

        with self._lock:
            self._active_requests[request_id] = req

        self._pending_queue.put(req)

        try:
            while True:
                token = await output_queue.get()
                if token is None:
                    break
                yield token
        finally:
            with self._lock:
                self._active_requests.pop(request_id, None)

    async def cancel(self, request_id: str) -> bool:
        with self._lock:
            req = self._active_requests.get(request_id)
            if req and req.status == RequestStatus.PENDING:
                req.status = RequestStatus.CANCELLED
                self._active_requests.pop(request_id, None)
                return True
        return False

    def get_num_pending(self) -> int:
        return self._pending_queue.qsize()

    def get_num_active(self) -> int:
        with self._lock:
            return len(self._active_requests)
