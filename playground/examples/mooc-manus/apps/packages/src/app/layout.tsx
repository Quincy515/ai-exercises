import { useEffect, useRef, useState } from "react";
import type { ReactNode } from "react";
import { usePanelRef } from "react-resizable-panels";
import { ChatHeader } from "../components/chat-header";
import { LeftPanel } from "../components/left-panel";
import type { LeftPanelProps } from "../components/left-panel";
import { NavigationRail } from "../components/navigation-rail";
import type { NavigationTarget } from "../components/navigation-rail";
import {
  ResizableHandle,
  ResizablePanel,
  ResizablePanelGroup,
} from "../components/ui/resizable";
import { SidebarProvider, useSidebar } from "../components/ui/sidebar";

function ChatLayout({
  children,
  ...navigation
}: LeftPanelProps & { children: ReactNode }) {
  const { open, setOpen, isMobile } = useSidebar();
  const panelRef = usePanelRef();
  const expandedWidth = useRef(280);
  // defaultSize 仅用于初始化；固定它，保留组件记住的上次展开宽度。
  const [defaultSize] = useState(open ? "280px" : "0px");

  // 现有按钮和快捷键继续使用 Sidebar 状态，由面板执行展开与收起。
  useEffect(() => {
    // 按像素恢复，避免跨断点后用组件缓存的百分比展开到不同宽度。
    const panel = panelRef.current;
    if (open && !isMobile) {
      if (panel?.isCollapsed()) panel.resize(expandedWidth.current);
    } else panel?.collapse();
  }, [open, isMobile, panelRef]);

  return (
    // 保持内容组件树稳定，切换窄屏时保留页面状态。
    <ResizablePanelGroup
      orientation="horizontal"
      disabled={isMobile}
      className="h-svh min-w-0 flex-1"
    >
      {/* 左侧的面板：拖过最小宽度 40px 后，自动收起到 0。 */}
      <ResizablePanel
        id="chat-sidebar"
        panelRef={panelRef}
        defaultSize={defaultSize}
        minSize="220px"
        maxSize="420px"
        collapsible
        collapsedSize="0px"
        collapsedThreshold="40px"
        groupResizeBehavior="preserve-pixel-size"
        onResize={(size, _id, previousSize) => {
          // 拖动结果同步回按钮状态；初次挂载沿用 Sidebar 的状态。
          const nextOpen = size.inPixels > 0;
          if (!isMobile) {
            if (nextOpen) expandedWidth.current = size.inPixels;
            if (previousSize && nextOpen !== open) setOpen(nextOpen);
          }
        }}
      >
        <div className="h-full" inert={!open && !isMobile}>
          <LeftPanel {...navigation} />
        </div>
      </ResizablePanel>
      {!isMobile && <ResizableHandle aria-label="调整会话列表宽度" />}
      <ResizablePanel
        minSize={isMobile ? "0px" : "320px"}
        className="flex min-w-0"
      >
        {children}
      </ResizablePanel>
    </ResizablePanelGroup>
  );
}

export default function RootLayout({
  children,
  pathname,
  onNavigate,
  ...navigation
}: LeftPanelProps & {
  children: ReactNode;
  pathname: string;
  onNavigate: (to: NavigationTarget) => void;
}) {
  // noVNC 是独立全屏查看页，两端统一跳过三栏布局。
  if (/^\/sessions\/[^/]+\/novnc\/?$/.test(pathname)) {
    return <>{children}</>;
  }

  const isChat = pathname === "/" || pathname.startsWith("/sessions/");
  const content = (
    // 右侧的内容
    <main className="relative flex min-h-0 min-w-0 flex-1 flex-col overflow-auto bg-chat-background">
      {/* 首页顶部header；详情页使用自己的 SessionHeader，保留侧栏展开入口。 */}
      {pathname === "/" && (
        <ChatHeader onNavigateHome={navigation.onNewSession} />
      )}
      {/* 中间对话框 */}
      {children}
    </main>
  );

  return (
    <SidebarProvider className="h-svh">
      {/* 一级功能栏常驻，二级列表按当前模块显示。 */}
      <NavigationRail pathname={pathname} onNavigate={onNavigate} />
      {isChat ? <ChatLayout {...navigation}>{content}</ChatLayout> : content}
    </SidebarProvider>
  );
}
