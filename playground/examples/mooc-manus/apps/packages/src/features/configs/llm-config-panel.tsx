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
import { llmConfigFields } from "./events";
import type { LlmConfigController } from "./use-configs";

export const LLM_CONFIG_FORM_ID = "llm-config-form";

export function LlmConfigPanel({ config }: { config: LlmConfigController }) {
  const {
    data,
    draft,
    ready,
    loading,
    saving,
    error,
    edit,
    save,
    apiKey,
    editApiKey,
  } = config;
  const busy = loading || saving || (!ready && !error);
  const disabled = !data || loading || saving;

  return (
    <form
      id={LLM_CONFIG_FORM_ID}
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
            模型提供商
          </FieldLegend>
          <FieldDescription className="text-sm">
            配置 Agent 使用的基础 LLM 模型，需兼容 OpenAI
            格式。修改后点击保存生效。
          </FieldDescription>
          <ConfigFeedback
            config={config}
            hasData={data !== null}
            name="模型配置"
            successText="模型配置保存成功"
          />
          <FieldGroup>
            <Field>
              <FieldLabel className="flex-wrap" htmlFor="base_url">
                提供商基础地址<Kbd>base_url</Kbd>
              </FieldLabel>
              <Input
                id="base_url"
                type="url"
                value={draft.base_url}
                disabled={disabled}
                placeholder="https://api.example.com/v1"
                onChange={(event) =>
                  edit(llmConfigFields.base_url, event.target.value)
                }
              />
              <FieldDescription className="text-xs">
                填写兼容 OpenAI 格式的 HTTP(S) 地址；留空使用服务端默认配置。
              </FieldDescription>
            </Field>
            <Field>
              <FieldLabel className="flex-wrap" htmlFor="api_key">
                提供商密钥<Kbd>api_key</Kbd>
              </FieldLabel>
              <Input
                id="api_key"
                type="password"
                autoComplete="new-password"
                spellCheck={false}
                value={apiKey}
                disabled={disabled}
                placeholder={
                  data?.api_key_configured
                    ? "已配置，留空保留原密钥"
                    : "填写新的 API 密钥"
                }
                onChange={(event) => editApiKey(event.target.value)}
              />
              <FieldDescription className="text-xs">
                {data &&
                  (data.api_key_configured
                    ? "服务器已配置密钥。"
                    : "服务器尚未配置密钥。")}
                填写后替换，留空保留；保存成功后清空输入。
              </FieldDescription>
            </Field>
            <Field>
              <FieldLabel className="flex-wrap" htmlFor="model_name">
                模型名<Kbd>model_name</Kbd>
              </FieldLabel>
              <Input
                id="model_name"
                type="text"
                value={draft.model_name}
                disabled={disabled}
                placeholder="填写模型名称"
                onChange={(event) =>
                  edit(llmConfigFields.model_name, event.target.value)
                }
              />
              <FieldDescription className="text-xs">
                模型需支持工具调用、图像识别等功能；留空使用服务端默认模型。
              </FieldDescription>
            </Field>
            <Field>
              <FieldLabel className="flex-wrap" htmlFor="temperature">
                温度<Kbd>temperature</Kbd>
              </FieldLabel>
              <Input
                id="temperature"
                type="number"
                min={-2}
                max={2}
                step="any"
                value={draft.temperature}
                disabled={disabled}
                placeholder="使用服务端默认值"
                onChange={(event) =>
                  edit(llmConfigFields.temperature, event.target.value)
                }
              />
              <FieldDescription className="text-xs">
                接口允许 −2 到
                2，具体支持范围以提供商为准；留空使用服务端默认值。
              </FieldDescription>
            </Field>
            <Field>
              <FieldLabel className="flex-wrap" htmlFor="max_tokens">
                最大输出 token 数<Kbd>max_tokens</Kbd>
              </FieldLabel>
              <Input
                id="max_tokens"
                type="number"
                inputMode="numeric"
                min={0}
                step={1}
                value={draft.max_tokens}
                disabled={disabled}
                placeholder="使用服务端默认值"
                onChange={(event) =>
                  edit(llmConfigFields.max_tokens, event.target.value)
                }
              />
              <FieldDescription className="text-xs">
                填写非负整数；留空使用服务端默认值。
              </FieldDescription>
            </Field>
          </FieldGroup>
        </FieldSet>
      </FieldGroup>
    </form>
  );
}
