/// <reference types="vite/client" />
import { useCallback, useEffect } from "react";
import { resolveApiBaseUrl } from "./api-config.js";
import { useCrux } from "./use-crux.js";

/** 页面挂载后读取配置；业务状态与重复请求判断由 Rust 管理。 */
export function useAgentConfig() {
  const { view, events, dispatch, ready, error } = useCrux();
  const refresh = useCallback(() => {
    const baseUrl = resolveApiBaseUrl({
      configured: import.meta.env?.VITE_API_BASE_URL,
      location: window.location,
    });
    dispatch(events.GetAgentConfig(baseUrl));
  }, [dispatch, events]);

  useEffect(() => {
    if (ready) refresh();
  }, [ready, refresh]);

  return {
    ...view.agent_config,
    ready,
    // Core 初始化/桥接错误使用单独的提示，HTTP 业务错误来自 Rust。
    error:
      view.agent_config.error ??
      (error ? "客户端加载失败，请重新打开设置。" : null),
    refresh,
  };
}
