/// <reference types="vite/client" />
import { useEffect, useState } from "react";
import type { AgentConfigField, LlmConfigField } from "shared_types/app.js";
import { resolveApiBaseUrl } from "../../lib/crux/api-config.js";
import { useCrux } from "../../lib/crux/use-crux.js";
import { configEvents, llmConfigEvents, llmConfigFields } from "./events.js";

const emptyAgent = {
  max_iterations: "",
  max_retries: "",
  max_search_results: "",
};
const emptyLlm = {
  base_url: "",
  model_name: "",
  temperature: "",
  max_tokens: "",
};

function baseUrl() {
  return resolveApiBaseUrl({
    configured: import.meta.env?.VITE_API_BASE_URL,
    location: window.location,
  });
}

/** 一个设置弹窗持有一个 Core，Agent/LLM 的草稿与请求状态由各自 Rust 子模块管理。 */
export function useConfigs(active: string) {
  const { view, dispatch, ready, error } = useCrux();
  const agent = view?.agent_config;
  const llm = view?.llm_config;
  // 密码框仅保留用户正在输入的值，服务器密钥始终只读配置状态。
  const [apiKey, setApiKey] = useState("");
  const clientError = error ? "客户端加载失败，请关闭设置后重试。" : null;

  useEffect(() => {
    if (!ready) return;
    if (active === "common-setting")
      dispatch(configEvents.GetAgentConfig(baseUrl()));
    if (active === "llm-setting") dispatch(llmConfigEvents.Get(baseUrl()));
  }, [ready, active, dispatch]);

  useEffect(() => {
    if (!llm?.api_key_changed) setApiKey("");
  }, [llm?.api_key_changed]);

  return {
    agent: {
      data: agent?.data ?? null,
      draft: agent?.draft ?? emptyAgent,
      loading: agent?.loading ?? false,
      saving: agent?.saving ?? false,
      saved: agent?.saved ?? false,
      dirty: agent?.dirty ?? false,
      canSave: agent?.can_save ?? false,
      error: agent?.error ?? clientError,
      ready,
      refresh: () => dispatch(configEvents.GetAgentConfig(baseUrl())),
      edit: (field: AgentConfigField, value: string) =>
        dispatch(configEvents.EditAgentConfig(field, value)),
      reset: () => dispatch(configEvents.ResetAgentConfig()),
      save: () => dispatch(configEvents.SaveAgentConfig(baseUrl())),
    },
    llm: {
      data: llm?.data ?? null,
      draft: llm?.draft ?? emptyLlm,
      loading: llm?.loading ?? false,
      saving: llm?.saving ?? false,
      saved: llm?.saved ?? false,
      dirty: llm?.dirty ?? false,
      canSave: llm?.can_save ?? false,
      error: llm?.error ?? clientError,
      ready,
      apiKey,
      refresh: () => dispatch(llmConfigEvents.Get(baseUrl())),
      edit: (field: LlmConfigField, value: string) =>
        dispatch(llmConfigEvents.Edit(field, value)),
      editApiKey: (value: string) => {
        setApiKey(value);
        dispatch(llmConfigEvents.Edit(llmConfigFields.api_key, value));
      },
      reset: () => {
        setApiKey("");
        dispatch(llmConfigEvents.Reset());
      },
      save: () => dispatch(llmConfigEvents.Save(baseUrl())),
    },
  };
}

export type AgentConfigController = ReturnType<typeof useConfigs>["agent"];
export type LlmConfigController = ReturnType<typeof useConfigs>["llm"];
