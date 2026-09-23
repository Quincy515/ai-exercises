import { useEffect } from "react";
import { Button } from "./components/ui/button";
import { useCrux } from "./lib/crux/use-crux.js";

function Chat() {
  const { view, events, dispatch, ready, error } = useCrux();

  useEffect(() => {
    if (ready) dispatch(events.LoadState());
  }, [ready, dispatch, events]);

  return (
    <section className="flex min-h-svh flex-col items-center justify-center gap-6 px-6">
      <div className="space-y-2 text-center">
        <h1 className="text-2xl font-semibold">共享计数器</h1>
        <p className="text-sm text-muted-foreground">
          Web 与桌面端使用同一套 Rust 业务逻辑
        </p>
      </div>
      <output aria-live="polite" className="text-xl tabular-nums">
        {ready ? view.text : "正在加载…"}
      </output>
      <div className="flex flex-wrap justify-center gap-3">
        <Button disabled={!ready} onClick={() => dispatch(events.Decrement())}>
          减一
        </Button>
        <Button
          variant="outline"
          disabled={!ready}
          onClick={() => dispatch(events.Reset())}
        >
          重置
        </Button>
        <Button disabled={!ready} onClick={() => dispatch(events.Increment())}>
          加一
        </Button>
        <Button
          variant="outline"
          disabled={!ready}
          onClick={() => dispatch(events.Get())}
        >
          同步服务端
        </Button>
      </div>
      <p className="text-sm text-muted-foreground">
        {ready ? (view.confirmed ? "已同步" : "本地状态") : "初始化中"}
      </p>
      {error && (
        <p
          role="alert"
          className="max-w-lg text-center text-sm text-destructive"
        >
          {error.message}
        </p>
      )}
    </section>
  );
}
export default Chat;
