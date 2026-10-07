/// <reference types="vite/client" />
import { useCallback, useEffect } from "react";
import type { AgentConfigField } from "shared_types/app.js";
import { resolveApiBaseUrl } from "../../lib/crux/api-config.js";
import { useCrux } from "../../lib/crux/use-crux.js";
import { configEvents } from "./events.js";

const emptyDraft = {
  max_iterations: "",
  max_retries: "",
  max_search_results: "",
};

function baseUrl() {
  return resolveApiBaseUrl({
    configured: import.meta.env?.VITE_API_BASE_URL,
    location: window.location,
  });
}

/** 弹窗内容持有一个 Core，表单和底部保存按钮共享这份状态。 */
export function useAgentConfig(active = true) {
  const { view, dispatch, ready, error } = useCrux();
  const state = view?.agent_config;
  const refresh = useCallback(
    () => dispatch(configEvents.GetAgentConfig(baseUrl())),
    [dispatch],
  );
  const edit = useCallback(
    (field: AgentConfigField, value: string) => {
      dispatch(configEvents.EditAgentConfig(field, value));
    },
    [dispatch],
  );
  const reset = useCallback(
    () => dispatch(configEvents.ResetAgentConfig()),
    [dispatch],
  );
  const save = useCallback(
    () => dispatch(configEvents.SaveAgentConfig(baseUrl())),
    [dispatch],
  );

  useEffect(() => {
    if (ready && active) refresh();
  }, [ready, active, refresh]);

  return {
    data: state?.data ?? null,
    draft: state?.draft ?? emptyDraft,
    loading: state?.loading ?? false,
    saving: state?.saving ?? false,
    saved: state?.saved ?? false,
    dirty: state?.dirty ?? false,
    canSave: state?.can_save ?? false,
    error:
      state?.error ?? (error ? "客户端加载失败，请刷新页面后重试。" : null),
    ready,
    refresh,
    edit,
    reset,
    save,
  };
}

export type AgentConfigController = ReturnType<typeof useAgentConfig>;
