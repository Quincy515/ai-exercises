import {
  eventConfigs,
  configsEventAgent,
  configsEventLlm,
  configsEventA2a,
  a2aConfigEventGet,
  a2aConfigEventEditUrl,
  a2aConfigEventResetDraft,
  a2aConfigEventCreate,
  a2aConfigEventSetEnabled,
  a2aConfigEventDelete,
  agentConfigEventGet,
  agentConfigEventEdit,
  agentConfigEventReset,
  agentConfigEventSave,
  agentConfigFieldMaxIterations,
  agentConfigFieldMaxRetries,
  agentConfigFieldMaxSearchResults,
  llmConfigEventGet,
  llmConfigEventEdit,
  llmConfigEventReset,
  llmConfigEventSave,
  llmConfigFieldBaseUrl,
  llmConfigFieldApiKey,
  llmConfigFieldModelName,
  llmConfigFieldTemperature,
  llmConfigFieldMaxTokens,
} from "shared_types/app.js";
import type { AgentConfigField, LlmConfigField } from "shared_types/app.js";

export const agentConfigFields = {
  max_iterations: agentConfigFieldMaxIterations(),
  max_retries: agentConfigFieldMaxRetries(),
  max_search_results: agentConfigFieldMaxSearchResults(),
} as const;

export const llmConfigFields = {
  base_url: llmConfigFieldBaseUrl(),
  api_key: llmConfigFieldApiKey(),
  model_name: llmConfigFieldModelName(),
  temperature: llmConfigFieldTemperature(),
  max_tokens: llmConfigFieldMaxTokens(),
} as const;

// 事件包装只连接生成类型；请求、校验与状态更新留在 Rust 子模块。
export const configEvents = {
  GetAgentConfig: (baseUrl: string) =>
    eventConfigs(configsEventAgent(agentConfigEventGet(baseUrl))),
  EditAgentConfig: (field: AgentConfigField, value: string) =>
    eventConfigs(configsEventAgent(agentConfigEventEdit(field, value))),
  ResetAgentConfig: () =>
    eventConfigs(configsEventAgent(agentConfigEventReset())),
  SaveAgentConfig: (baseUrl: string) =>
    eventConfigs(configsEventAgent(agentConfigEventSave(baseUrl))),
} as const;

export const llmConfigEvents = {
  Get: (baseUrl: string) =>
    eventConfigs(configsEventLlm(llmConfigEventGet(baseUrl))),
  Edit: (field: LlmConfigField, value: string) =>
    eventConfigs(configsEventLlm(llmConfigEventEdit(field, value))),
  Reset: () => eventConfigs(configsEventLlm(llmConfigEventReset())),
  Save: (baseUrl: string) =>
    eventConfigs(configsEventLlm(llmConfigEventSave(baseUrl))),
} as const;

export const a2aConfigEvents = {
  Get: (baseUrl: string) =>
    eventConfigs(configsEventA2a(a2aConfigEventGet(baseUrl))),
  EditUrl: (value: string) =>
    eventConfigs(configsEventA2a(a2aConfigEventEditUrl(value))),
  ResetDraft: () => eventConfigs(configsEventA2a(a2aConfigEventResetDraft())),
  Create: (baseUrl: string) =>
    eventConfigs(configsEventA2a(a2aConfigEventCreate(baseUrl))),
  SetEnabled: (baseUrl: string, id: string, enabled: boolean) =>
    eventConfigs(
      configsEventA2a(a2aConfigEventSetEnabled(baseUrl, id, enabled)),
    ),
  Delete: (baseUrl: string, id: string) =>
    eventConfigs(configsEventA2a(a2aConfigEventDelete(baseUrl, id))),
} as const;
