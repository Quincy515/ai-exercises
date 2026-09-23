import {
  Instant,
  timeResponseCleared,
  timeResponseDurationElapsed,
  timeResponseInstantArrived,
  timeResponseNow,
  type TimeRequest,
  type TimeResponse,
  type TimerId,
} from "shared_types/app.js";

const maxDelay = 2_147_483_647n;
const nanosPerMillisecond = 1_000_000n;

export class Time {
  private readonly timers = new Map<bigint, ReturnType<typeof setTimeout>>();
  private disposed = false;

  request(
    request: TimeRequest,
    respond: (response: TimeResponse) => void,
  ): void {
    if (this.disposed) return;

    switch (request.kind) {
      case "Now": {
        const now = BigInt(Date.now());
        respond(
          timeResponseNow(new Instant(now / 1000n, Number(now % 1000n) * 1e6)),
        );
        break;
      }
      case "NotifyAt":
        this.schedule(
          request.id,
          request.instant.seconds * 1000n +
            (BigInt(request.instant.nanos) + nanosPerMillisecond - 1n) /
              nanosPerMillisecond,
          timeResponseInstantArrived(request.id),
          respond,
        );
        break;
      case "NotifyAfter":
        this.schedule(
          request.id,
          BigInt(Date.now()) +
            (request.duration.nanos + nanosPerMillisecond - 1n) /
              nanosPerMillisecond,
          timeResponseDurationElapsed(request.id),
          respond,
        );
        break;
      case "Clear":
        clearTimeout(this.timers.get(request.id.value));
        this.timers.delete(request.id.value);
        respond(timeResponseCleared(request.id));
        break;
    }
  }

  private schedule(
    id: TimerId,
    deadline: bigint,
    response: TimeResponse,
    respond: (response: TimeResponse) => void,
  ): void {
    clearTimeout(this.timers.get(id.value));

    // 浏览器长延时会溢出；按截止时间分段等待。
    const wait = () => {
      const remaining = deadline - BigInt(Date.now());
      const delay =
        remaining <= 0n ? 0n : remaining > maxDelay ? maxDelay : remaining;
      const handle = setTimeout(() => {
        if (this.disposed) return;
        if (BigInt(Date.now()) < deadline) {
          wait();
          return;
        }
        this.timers.delete(id.value);
        respond(response);
      }, Number(delay));
      this.timers.set(id.value, handle);
    };
    wait();
  }

  dispose(): void {
    this.disposed = true;
    for (const handle of this.timers.values()) clearTimeout(handle);
    this.timers.clear();
  }
}
