import {
  eventConfigs,
  configsEventGetAgentConfig,
  configsEventEditAgentConfig,
  configsEventResetAgentConfig,
  configsEventSaveAgentConfig,
  agentConfigFieldMaxIterations,
  agentConfigFieldMaxRetries,
  agentConfigFieldMaxSearchResults,
} from "shared_types/app.js";
import type { AgentConfigField } from "shared_types/app.js";

export const agentConfigFields = {
  max_iterations: agentConfigFieldMaxIterations(),
  max_retries: agentConfigFieldMaxRetries(),
  max_search_results: agentConfigFieldMaxSearchResults(),
} as const;

// 模块事件包装只负责连接生成类型，业务校验与状态更新留在 Rust。
export const configEvents = {
  GetAgentConfig: (baseUrl: string) =>
    eventConfigs(configsEventGetAgentConfig(baseUrl)),
  EditAgentConfig: (field: AgentConfigField, value: string) =>
    eventConfigs(configsEventEditAgentConfig(field, value)),
  ResetAgentConfig: () => eventConfigs(configsEventResetAgentConfig()),
  SaveAgentConfig: (baseUrl: string) =>
    eventConfigs(configsEventSaveAgentConfig(baseUrl)),
} as const;
