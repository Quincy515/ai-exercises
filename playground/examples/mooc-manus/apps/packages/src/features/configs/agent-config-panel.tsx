import { ConfigFeedback } from "./config-feedback";
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
import type { AgentConfigController } from "./use-configs";

export const AGENT_CONFIG_FORM_ID = "agent-config-form";

export function AgentConfigPanel({
  config,
}: {
  config: AgentConfigController;
}) {
  const { data, draft, ready, loading, saving, error, edit, save } = config;
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
          <ConfigFeedback config={config} hasData={data !== null} />
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
