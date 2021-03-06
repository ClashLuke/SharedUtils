import datetime
import multiprocessing
import threading
import time
import traceback
import typing
import uuid
from multiprocessing.shared_memory import SharedMemory

import numpy as np


def try_except(default: typing.Optional[typing.Any] = None):
    """
    Function decorator to stop exceptions from stopping the program. Instead, the error is printed and default value
    returned.
    As it's a used as a function decorator, it doesn't alter typehints while still working with all arguments.

    :param default: Default value to return in case of an error
    :return: decorator which can be applied to any function
    """

    def _decorator(fn: typing.Callable):
        def _fn(*args, **kwargs):
            try:
                return fn(*args, **kwargs)
            except Exception as exc:
                print(r"IGNORED EXCEPTION \/\/\/")
                print(fn, exc)
                traceback.print_exc()
                print("IGNORED EXCEPTION /\\/\\/\\")

            return default

        return _fn

    return _decorator


def call_with(contexts: list, fn: typing.Callable[[], None], cond_fn: typing.Callable[[], bool]):
    """
    Used to call a function with multiple context objects (for example, 4 multiprocessing locks) while ensuring a
    condition stays false before entering the next context.
    This is useful for when slowly acquiring multiple locks in a highly distributed setting. Sometimes it can be the
    case that the reason those locks should've been acquired was already fixed by another process,
    so that slowly acquiring the rest of the locks isn't necessary anymore.
    :param contexts: A list of objects that can be entered using a with statement (needs __enter__ and __exit__,
    such as tf.control_dependencies and multiprocessing.Lock())
    :param fn: Function that will be called once all contexts are entered
    :param cond_fn: Callback called whenever a context is acquired, to ensure the function still has to run.
    :return: either none (-> cond = True) or output of fn
    """
    if not contexts:
        return fn()
    if cond_fn():
        return
    with contexts.pop(0):
        return call_with(contexts, fn, cond_fn)


class ListQueue:
    """
    A reimplementation of multiprocessing's Queue with a public list attribute which can be inspected by any process
    without forcibly locking the whole thing.

    It's used the same way as a normal queue, with the slight difference that ListQueue has a `.list` attribute which
    displays tbe entire queue in order.
    """

    def __init__(self):
        manager = multiprocessing.Manager()
        self.list = manager.list()
        self.write_lock = manager.RLock()
        self.read_lock = manager.RLock()
        self.cond = manager.Condition(manager.Lock())

    def get(self):
        with self.read_lock:
            if not self.list:
                with self.cond:
                    self.cond.wait()
            return self.list.pop(0)

    def put(self, obj):
        with self.write_lock:
            self.list.append(obj)
            with self.cond:
                self.cond.notify_all()


class FiFoSemaphore:
    """
    A semaphore that processes items in a FiFo fashion with optional values to increment or decrement by.
    This can be useful to process things in order and have cleaner retry-loops.
    Additionally, it's used by `SharedSequentialQueue` to "enqueue" that one worker can't have any other worker to run.
    Once all other workers are finished, this main worker would be started immediately and does its task. Without
    values, this would require many calls to .acquire() where any of the other workers could intervene and cause a
    deadlock. Without the in-order execution, it's possible that tasks don't get executed in the FiFo way they were
    intended by the programmer.

    Usage:
    >>> def worker(semaphore: FiFoSemaphore):
    ...     with semaphore(5):
    ...         print("Hello World")
    ...         semaphore.acquire(10)
    ...         print("Fully acquired")
    ...         semaphore.release(5)
    ...         print("Releasing 5")
    ...         semaphore.release(5)
    ...         print("Releasing 5 more")
    ...     print("Exit")
    >>> sem = FiFoSemaphore(15)
    >>> [multiprocessing.Process(target=worker, args=(sem,)).start() for _ in range(3)]
    Hello World
    Fully acquired
    Releasing 5
    Hello World
    Releasing 5 more
    Hello World
    Exit
    Fully acquired
    Releasing 5
    Fully acquired
    Releasing 5 more
    Releasing 5
    Exit
    Exit
    """

    def __init__(self, value: int = 1):
        manager = multiprocessing.Manager()
        self._cond = multiprocessing.Condition(multiprocessing.Lock())
        self._value = manager.list([value])
        self._queue = ListQueue()
        self._value_lock = multiprocessing.Lock()
        self.max_value = value

    def __call__(self, val: int = 0):
        return FiFoSemaphoreContext(self, val)

    def acquire(self, val: int = 0):
        job_id = uuid.uuid4()
        self._queue.put(job_id)
        if val < 1:
            val = self.max_value + val
        with self._cond:
            while self._queue.list[0] != job_id or self._value[0] < val:
                self._cond.wait()
            with self._value_lock:
                self._value[0] -= val
            self._queue.get()
        return True

    def release(self, val: int = 0):
        if val < 1:
            val = self.max_value + val
        with self._cond:
            with self._value_lock:
                self._value[0] += val
            self._cond.notify_all()


class FiFoSemaphoreContext:
    def __init__(self, semaphore: FiFoSemaphore, val: int):
        self.semaphore = semaphore
        self.val = val

    def __enter__(self):
        self.semaphore.acquire(self.val)

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.semaphore.release(self.val)


class SharedEXTQueue:
    """
    A non-FiFo queue using shared memory.
    It can be used if the read order doesn't matter and speed in a multiple-producer multiple-consumer setting
    is the most important.
    Note that replicating the queue to multiple workers does not replicate its memory and sending messages does not
    incur the pickling overhead of normal python transfers. This way, all workers can access the same area in RAM at
    high speeds.

    The memory allocation is similar to https://en.wikipedia.org/wiki/Extended_file_system, so there are no guarantees
    which file gets "popped". However, insertion is very fast and it support multiple producers and multiple consumers.
    Internally, It uses a shared list to synchronise "file pointers" and adds new "files" to a shared numpy array.
    Files get inserted wherever there's free space, while the readers only read at the start.
    Note that there might be fragmentation if the memory is too full and files too large. With sufficiently small
    entries or large queues, fragmentation won't occur. Allocating memory for 1000 items seems to be enough to avoid
    fragmentation, even with slow consumers

    There are five core functions.
    1) from_shape(), which creates a new queue of the specified shape. The first dimension is used to index over items,
    while all other dimensions have to be shared across all entries.
        >>> queue = SharedEXTQueue.from_shape(27, 32, 32, 3)
        >>> assert queue.data.shape == (27, 32, 32, 3)
    2) export() and from_export(), which are necessary to use the queue in a new process without replicating its memory:
        >>> export = queue.export()
        >>> multiprocessing.Process(target=lambda x: print(queue.from_export(x).data.shape), args=(export,)).start()
        (27, 32, 32, 3)
    3) get() and put(), which act similar to multiprocessing queues
        >>> queue.put(np.ones((1, 32, 32, 3)))
        >>> queue.put(np.ones((3, 32, 32, 3)))
        >>> queue.get().shape
        (1, 32, 32, 3)
        >>> queue.get().shape
        (3, 32, 32, 3)
    However, keep in mind that the order cannot be depended upon. When producing and consuming items of the queue
    simulatenously, it's possible that the queue won't end up being (3, 4, 5, 6) 1..6 were appended and 2 consumed,
    but instead it could also look like (5, 3, 4, 6).
    """
    data_mem: SharedMemory
    data: np.ndarray
    indices: list
    write_index_lock: threading.Lock
    read_index_lock: threading.Lock
    dtype: np.dtype

    @classmethod
    def from_shape(cls, *shape: int, dtype: np.dtype = np.uint8):
        self = cls()
        self.data_mem = SharedMemory(create=True, size=np.zeros(shape, dtype=dtype).nbytes)
        self.data = np.ndarray(shape, dtype=dtype, buffer=self.data_mem.buf)
        manager = multiprocessing.Manager()
        self.indices = manager.list()
        self.data[:] = 0
        self.write_index_lock = manager.Lock()
        self.read_index_lock = manager.Lock()
        self.dtype = dtype
        return self

    @classmethod
    def from_export(cls, data_name, data_shape, indices, write_index_lock, read_index_lock, dtype):
        self = cls()
        self.data_mem = SharedMemory(create=False, name=data_name)
        self.data = np.ndarray(data_shape, dtype=dtype, buffer=self.data_mem.buf)
        self.indices = indices
        self.write_index_lock = write_index_lock
        self.read_index_lock = read_index_lock
        return self

    def export(self):
        return self.data_mem.name, self.data.shape, self.indices, self.write_index_lock, self.read_index_lock, \
               self.dtype

    def get(self):
        while True:
            with self.read_index_lock:
                while not self:
                    time.sleep(1)
                start, end = self.indices.pop(0)
            return self.data[start:end].copy()  # local clone, so share can be safely edited

    def _free_memory(self, size: int) -> typing.Optional[typing.Tuple[int, int, int]]:
        if not self:
            return 0, 0, size
        local_indices = list(self.indices)
        itr = zip([[None, 0]] + local_indices, local_indices + [[self.data.shape[0], None]])
        for i, ((_, prev_end), (start, _)) in enumerate(itr):
            if start - prev_end > size:
                return i, prev_end, prev_end + size

    def _put_item(self, obj: np.ndarray):
        batches = obj.shape[0]
        with self.write_index_lock:
            indices = self._free_memory(batches)
            if indices is None:
                return
            idx, start, end = indices
            self.indices.insert(idx, (start, end))
        self.data[start:end] = obj[:]  # we simply assume that the synchronisation overheads make the reader slower

    def put(self, obj: np.ndarray):
        batches = obj.shape[0]
        max_size = self.data.shape[0] // 4  # unrealistic that it'll fit if it takes up 25% of the memory
        if batches > max_size:
            for idx in range(0, batches, max_size):  # ... so we slice it up and feed in many smaller videos
                self.put(obj[idx:idx + max_size])
            return

        def _fits():
            return bool(self._free_memory(batches))

        # until new data fits into memory
        waiting = 12
        while not _fits():
            time.sleep(5)
            waiting -= 1
        if not waiting:
            print("Warning: waited for one minute for space to free up, but none found. Increase memory size to avoid "
                  "fragmentation or implement defragmentation. Timestamp:", datetime.datetime.now(), flush=True)
            return

        self._put_item(obj)

    def __bool__(self):
        return bool(self.indices)


class SharedFiFoQueue:
    """
    A FiFo Queue using shared memory.
    While still supporting multiple-producer multiple-consumer setups, this queue has to lock more aggressively than
    `SharedEXTQueue`, which could lead to potential slowdowns. Otherwise, the usage is identical, so refer to the docs
    of `SharedEXTQueue`.
    """
    data_mem: SharedMemory
    data: np.ndarray
    index_queue: ListQueue
    read_memory: FiFoSemaphore
    write_memory: FiFoSemaphore
    read_timeout: int
    dtype: np.dtype

    @classmethod
    def from_shape(cls, *shape: int, exclusive: int = 128, read_timeout: int = 3, dtype: np.dtype= np.uint8):
        self = cls()
        self.data_mem = SharedMemory(create=True, size=np.zeros(shape, dtype=dtype).nbytes)
        self.data = np.ndarray(shape, dtype=dtype, buffer=self.data_mem.buf)
        self.index_queue = ListQueue()
        self.data[:] = 0
        self.read_memory = FiFoSemaphore(exclusive)
        self.write_memory = FiFoSemaphore(exclusive)
        self.read_timeout = read_timeout
        self.dtype = dtype
        return self

    @classmethod
    def from_export(cls, data_name, data_shape, index_queue, read_from_memory, write_to_memory, read_timeout, dtype):
        self = cls()
        self.data_mem = SharedMemory(create=False, name=data_name)
        self.data = np.ndarray(data_shape, dtype=dtype, buffer=self.data_mem.buf)
        self.index_queue = index_queue
        self.read_memory = read_from_memory
        self.write_memory = write_to_memory
        self.read_timeout = read_timeout
        return self

    def export(self):
        return self.data_mem.name, self.data.shape, self.index_queue, self.read_memory, self.write_memory, \
               self.read_timeout, self.dtype

    def get(self):
        while True:
            with self.read_memory(1):
                start, end = self.index_queue.get()
                return self.data[start:end].copy()  # local clone, so share can be safely edited

    def _shift_left(self):
        local_list = list(self.index_queue.list)
        min_start = local_list[0][0]
        max_end = local_list[-1][1]
        self.data[:max_end - min_start] = self.data[min_start:max_end]
        self.index_queue.list[:] = [(start - min_start, end - min_start) for start, end in local_list]

    def _put_item(self, obj: np.ndarray):
        batches = obj.shape[0]
        with self.index_queue.write_lock:
            if self.index_queue.list:
                _, start = self.index_queue.list[-1]
                start += 1
            else:
                start = 0
            end = start + batches
            self.index_queue.put((start, end))
        with self.write_memory(1):
            self.data[start:end] = obj[:]  # we simply assume that the synchronisation overheads make the reader slower

    def put(self, obj: np.ndarray):
        batches = obj.shape[0]
        max_size = self.data.shape[0] // 4  # unrealistic that it'll fit if it takes up 25% of the memory
        if batches > max_size:
            for idx in range(0, batches, max_size):  # ... so we slice it up and feed in many smaller videos
                self.put(obj[idx:idx + max_size])
            return

        def _fits():
            return not self or self.index_queue.list[-1][1] + batches < self.data.shape[0]

        # until new data fits into memory
        while not _fits():
            while self and self.index_queue.list[0][0] == 0:  # wait for anything to be read
                time.sleep(2)
            # ensure _nothing_ else is reading or writing
            call_with([self.write_memory(), self.read_memory(), self.index_queue.read_lock,
                       self.index_queue.write_lock], self._shift_left, _fits)
        self._put_item(obj)

    def __bool__(self):
        return bool(self.index_queue.list)
