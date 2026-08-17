/**
 * A FIFO lock, serializing whatever shares one instance of it.
 *
 * `ChatSession` holds one per session, so overlapping `respond()` calls on a
 * session queue up — while two sessions run concurrently, which is safe because
 * each owns its own context and KV cache. Pass a shared lock via
 * `GenerateOptions.lock` to serialize more broadly than that.
 */

export class InferenceLock {
  #tail: Promise<void> = Promise.resolve();

  /**
   * Acquire exclusive access. Resolves to a `release` function.
   *
   * If `signal` aborts while queued, rejects immediately — without waiting for the
   * current holder, which may be decoding for minutes — and without taking the lock.
   */
  acquire(signal?: AbortSignal): Promise<() => void> {
    signal?.throwIfAborted();

    let release!: () => void;
    const done = new Promise<void>((r) => {
      release = r;
    });
    const prev = this.#tail;
    // Chain rather than replace: our successor waits for `prev` *and* for us, so
    // bailing out early on abort releases only our own slot — it can never let a
    // later waiter overtake the holder that is still running.
    this.#tail = prev.then(() => done);

    if (!signal) return prev.then(() => release);

    return new Promise<() => void>((resolve, reject) => {
      const onAbort = () => {
        // Give up our slot so the queue keeps advancing behind us, then reject
        // now instead of when our turn would have come.
        release();
        reject(signal.reason);
      };
      signal.addEventListener("abort", onAbort, { once: true });
      prev.then(() => {
        // Already aborted => onAbort has run and rejected; nothing left to do.
        if (signal.aborted) return;
        // Stop listening before handing the lock over, or a later abort would
        // release it out from under the caller.
        signal.removeEventListener("abort", onAbort);
        resolve(release);
      });
    });
  }

  /** Run `fn` exclusively under this lock. */
  async run<T>(fn: () => Promise<T>, signal?: AbortSignal): Promise<T> {
    const release = await this.acquire(signal);
    try {
      return await fn();
    } finally {
      release();
    }
  }
}
