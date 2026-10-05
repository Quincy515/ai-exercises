import { LOCAL_API_BASE_URL } from "./packages/src/lib/crux/api-config.ts";

// Web 与 Electron 的开发页面共用同源代理，后端地址只在共享配置中维护。
export const apiDevProxy = {
  "/api": { target: LOCAL_API_BASE_URL, changeOrigin: true },
};
