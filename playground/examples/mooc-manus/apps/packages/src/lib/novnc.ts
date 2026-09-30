import RFB from "@novnc/novnc/lib/rfb";

// 兼容不同打包器对 CommonJS 默认导出的包装。
const RFBModule = RFB as typeof RFB & { default?: typeof RFB };
export default RFBModule.default ?? RFBModule;
