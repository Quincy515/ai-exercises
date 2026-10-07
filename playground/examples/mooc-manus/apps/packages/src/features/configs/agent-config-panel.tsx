import { Button } from "../../components/ui/button";
import {
  Field,
  FieldDescription,
  FieldGroup,
  FieldLabel,
  FieldLegend,
  FieldSet,
} from "../../components/ui/field";
import { Input } from "../../components/ui/input";
import { Kbd } from "../../components/ui/kbd";
import { agentConfigFields } from "./events";
import type { AgentConfigController } from "./use-agent-config";

export const AGENT_CONFIG_FORM_ID = "agent-config-form";

export function AgentConfigPanel({
  config,
}: {
  config: AgentConfigController;
}) {
  const {
    data,
    draft,
    ready,
    loading,
    saving,
    dirty,
    saved,
    error,
    edit,
    refresh,
    reset,
    save,
  } = config;
  const busy = loading || saving || (!ready && !error);

  return (
    <form
      id={AGENT_CONFIG_FORM_ID}
      className="w-full px-1"
      aria-busy={busy}
      noValidate
      onSubmit={(event) => {
        event.preventDefault();
        save();
      }}
    >
      <FieldGroup>
        <FieldSet>
          <FieldLegend className="font-bold text-foreground data-[variant=legend]:text-lg">
            通用配置
          </FieldLegend>
          <FieldDescription className="text-sm">
            修改 Agent 的运行参数，点击保存后生效。
          </FieldDescription>
          <div className="flex flex-wrap items-center justify-between gap-3">
            <p role="status" className="text-sm text-muted-foreground">
              {saving
                ? "正在保存配置…"
                : busy
                  ? "正在加载配置…"
                  : error
                    ? "请检查配置提示"
                    : saved
                      ? "保存成功"
                      : dirty
                        ? "有未保存的修改"
                        : data
                          ? "已读取服务器配置"
                          : "等待加载配置"}
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
                {error && !data ? "重试" : "刷新"}
              </Button>
            </div>
          </div>
          {error && (
            <p role="alert" className="text-sm text-destructive">
              {error}
            </p>
          )}
          {/* 原始输入交给 Rust 校验；noValidate 保证错误通过统一 ViewModel 展示。 */}
          <FieldGroup>
            <Field>
              <FieldLabel className="flex-wrap" htmlFor="max_iterations">
                最大迭代次数<Kbd>max_iterations</Kbd>
              </FieldLabel>
              <Input
                id="max_iterations"
                type="number"
                inputMode="numeric"
                min={1}
                max={999}
                step={1}
                value={draft.max_iterations}
                disabled={!data || loading || saving}
                onChange={(event) =>
                  edit(agentConfigFields.max_iterations, event.target.value)
                }
              />
              <FieldDescription className="text-xs">
                Agent 循环调用工具的最大次数，范围 1–999。
              </FieldDescription>
            </Field>
            <Field>
              <FieldLabel className="flex-wrap" htmlFor="max_retries">
                最大重试次数<Kbd>max_retries</Kbd>
              </FieldLabel>
              <Input
                id="max_retries"
                type="number"
                inputMode="numeric"
                min={2}
                max={9}
                step={1}
                value={draft.max_retries}
                disabled={!data || loading || saving}
                onChange={(event) =>
                  edit(agentConfigFields.max_retries, event.target.value)
                }
              />
              <FieldDescription className="text-xs">
                LLM 或工具调用的最大重试次数，范围 2–9。
              </FieldDescription>
            </Field>
            <Field>
              <FieldLabel className="flex-wrap" htmlFor="max_search_results">
                最大搜索结果<Kbd>max_search_results</Kbd>
              </FieldLabel>
              <Input
                id="max_search_results"
                type="number"
                inputMode="numeric"
                min={2}
                max={29}
                step={1}
                value={draft.max_search_results}
                disabled={!data || loading || saving}
                onChange={(event) =>
                  edit(agentConfigFields.max_search_results, event.target.value)
                }
              />
              <FieldDescription className="text-xs">
                每个搜索步骤返回的最大结果数，范围 2–29。
              </FieldDescription>
            </Field>
          </FieldGroup>
        </FieldSet>
      </FieldGroup>
    </form>
  );
}
