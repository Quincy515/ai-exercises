import type { CoreFfi } from "shared";
import type { Effect, Event } from "shared_types/app.js";
import {
  Requests,
  ViewModel,
  serializeEvent,
  serializeHttpResult,
  serializeKeyValueResult,
  serializeSseResponse,
  serializeTimeResponse,
  sseResponseDone,
} from "shared_types/app.js";
import {
  BincodeDeserializer,
  BincodeSerializer,
} from "shared_types/bincode/index.js";
import { request as http } from "./http.js";
import { request as keyValue } from "./key-value.js";
import { request as sse } from "./sse.js";
import { Time } from "./time.js";

/** ViewModel 的数据字段；序列化方法由生成代码管理。 */
export type CruxViewModel = {
  [
    K in keyof ViewModel as ViewModel[K] extends (...args: never[]) => unknown
      ? never
      : K
  ]: ViewModel[K];
};

/** Shell 桥接：业务状态留在 Rust，浏览器执行 effect。 */
export class Core {
  private core: CoreFfi | null = null;
  private initializing: Promise<void> | null = null;
  private disposed = false;
  private readonly controller = new AbortController();
  private readonly time = new Time();

  constructor(
    private readonly onView: (view: CruxViewModel) => void,
    private readonly onError: (error: Error) => void = console.error,
  ) {}

  get ready(): boolean {
    return this.core !== null && !this.disposed;
  }

  initialize(): Promise<void> {
    if (this.disposed)
      return Promise.reject(new Error("Core has been disposed"));
    if (this.core) return Promise.resolve();
    if (!this.initializing) {
      this.initializing = this.load().catch((error: unknown) => {
        this.initializing = null;
        throw error;
      });
    }
    return this.initializing;
  }

  private async load(): Promise<void> {
    // 客户端挂载后按需加载生成包，并等待 WASM 初始化。
    const wasm = await import("shared");
    await (wasm as typeof wasm & { initialized: Promise<void> }).initialized;
    if (this.disposed) return;

    this.core = wasm.CoreFfi.new({
      processEffects: (bytes) => {
        // 离开 FFI 回调栈后再处理，避免在 Rust 调用过程中重入 resolve。
        const copy = bytes.slice();
        queueMicrotask(() => this.processEffects(copy));
      },
    });
    this.onView(this.view());
  }

  view(): CruxViewModel {
    if (!this.ready)
      throw new Error("Call initialize() before reading the view");
    return {
      ...ViewModel.deserialize(new BincodeDeserializer(this.core!.view())),
    };
  }

  update(event: Event): void {
    if (!this.ready) throw new Error("Call initialize() before sending events");
    const serializer = new BincodeSerializer();
    serializeEvent(event, serializer);
    this.processEffects(this.core!.update(serializer.getBytes()));
  }

  private processEffects(bytes: Uint8Array): void {
    if (this.disposed || bytes.length === 0) return;
    try {
      const requests = Requests.deserialize(
        new BincodeDeserializer(bytes),
      ).value;
      for (const { id, effect } of requests) {
        // HTTP 与长连接独立执行，Render 可立即刷新乐观更新。
        void this.processEffect(id, effect).catch((error: unknown) =>
          this.report(error),
        );
      }
    } catch (error) {
      this.report(error);
    }
  }

  private async processEffect(id: number, effect: Effect): Promise<void> {
    switch (effect.kind) {
      case "Render":
        this.onView(this.view());
        return;
      case "Http": {
        const result = await http(effect.value, {
          signal: this.controller.signal,
        });
        if (result.kind === "Err") {
          this.report(
            new Error(
              result.value.kind === "Timeout"
                ? "HTTP request cancelled or timed out"
                : result.value.value,
            ),
          );
        }
        this.respond(id, result, serializeHttpResult);
        return;
      }
      case "KeyValue":
        this.respond(id, keyValue(effect.value), serializeKeyValueResult);
        return;
      case "Time":
        this.time.request(effect.value, (response) =>
          this.respond(id, response, serializeTimeResponse),
        );
        return;
      case "ServerSentEvents":
        try {
          for await (const response of sse(effect.value, {
            signal: this.controller.signal,
          })) {
            this.respond(id, response, serializeSseResponse);
          }
        } catch (error) {
          // 网络异常同样结束 Rust 中挂起的流。
          this.respond(id, sseResponseDone(), serializeSseResponse);
          throw error;
        }
        return;
      default: {
        const unreachable: never = effect;
        throw new Error(`Unhandled effect: ${String(unreachable)}`);
      }
    }
  }

  private respond<T>(
    id: number,
    response: T,
    serialize: (value: T, serializer: BincodeSerializer) => void,
  ): void {
    if (!this.ready) return;
    try {
      const serializer = new BincodeSerializer();
      serialize(response, serializer);
      this.processEffects(this.core!.resolve(id, serializer.getBytes()));
    } catch (error) {
      this.report(error);
    }
  }

  private report(error: unknown): void {
    if (!this.disposed) {
      this.onError(error instanceof Error ? error : new Error(String(error)));
    }
  }

  dispose(): void {
    if (this.disposed) return;
    this.disposed = true;
    this.controller.abort();
    this.time.dispose();
    this.core?.dispose();
    this.core = null;
  }
}
