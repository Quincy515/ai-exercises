/// <reference types="novnc__novnc" />
import type RFB from "@novnc/novnc/lib/rfb";
import { useEffect, useRef } from "react";

interface VNCViewerProps {
  url: string;
  viewOnly?: boolean;
}

export function VNCViewer({ url, viewOnly = false }: VNCViewerProps) {
  const displayRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    // 1.检查引用是否存在
    const display = displayRef.current;
    if (!display) return;

    let disposed = false;
    let rfb: RFB | undefined;

    // 2.创建代理连接；浏览器挂载后加载，避免构建时访问 window。
    void import("../lib/novnc")
      .then(({ default: RFBClient }) => {
        if (disposed) return;
        rfb = new RFBClient(display, url, {
          credentials: { password: "", username: "", target: "" },
        });

        // 3.配置基础属性
        rfb.viewOnly = viewOnly;
        rfb.scaleViewport = true;
        rfb.background = "#000";
        rfb.addEventListener("connect", () => console.log("Connected"));
        rfb.addEventListener("disconnect", () => console.log("Disconnected"));
      })
      .catch((error: unknown) => {
        if (!disposed) console.error("Failed to initialize VNC", error);
      });

    // 卸载或参数变化时断开连接；加载完成前卸载则跳过创建。
    return () => {
      disposed = true;
      rfb?.disconnect();
    };
  }, [url, viewOnly]);

  return (
    <div
      ref={displayRef}
      role="region"
      aria-label="远程桌面"
      className="h-full w-full bg-black"
    />
  );
}
