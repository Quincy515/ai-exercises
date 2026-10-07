import { Button } from "../../components/ui/button";

interface ConfigFeedbackProps {
  config: {
    ready: boolean;
    loading: boolean;
    saving: boolean;
    dirty: boolean;
    saved: boolean;
    error: string | null;
    refresh: () => void;
    reset: () => void;
  };
  hasData: boolean;
  name?: string;
  successText?: string;
}

/** 设置资源共用的读取、撤销与错误反馈，业务状态由 Rust 提供。 */
export function ConfigFeedback({
  config,
  hasData,
  name = "配置",
  successText = "保存成功",
}: ConfigFeedbackProps) {
  const { ready, loading, saving, dirty, saved, error, refresh, reset } =
    config;
  const busy = loading || (!ready && !error);
  return (
    <>
      <div className="flex flex-wrap items-center justify-between gap-3">
        <p role="status" className="text-sm text-muted-foreground">
          {saving
            ? `正在保存${name}…`
            : busy
              ? `正在加载${name}…`
              : error
                ? "请检查配置提示"
                : saved
                  ? successText
                  : dirty
                    ? "有未保存的修改"
                    : hasData
                      ? `已读取服务器${name}`
                      : `等待加载${name}`}
        </p>
        <div className="flex gap-2">
          {dirty && (
            <Button
              type="button"
              size="sm"
              variant="ghost"
              disabled={loading || saving}
              onClick={reset}
            >
              撤销修改
            </Button>
          )}
          <Button
            type="button"
            size="sm"
            variant="outline"
            disabled={!ready || loading || saving || dirty}
            onClick={refresh}
          >
            {error && !hasData ? "重试" : "刷新"}
          </Button>
        </div>
      </div>
      {error && (
        <p role="alert" className="text-sm text-destructive">
          {error}
        </p>
      )}
    </>
  );
}
