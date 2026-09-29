import type { CSSProperties, ReactNode } from "react";
import { ChatHeader } from "../components/chat-header";
import { LeftPanel } from "../components/left-panel";
import type { LeftPanelProps } from "../components/left-panel";
import {
  SidebarProvider,
  SidebarTrigger,
  useSidebar,
} from "../components/ui/sidebar";

function Content({
  children,
  showChatHeader,
  onNavigateHome,
}: {
  children: ReactNode;
  showChatHeader: boolean;
  onNavigateHome: () => void;
}) {
  const { isMobile, state } = useSidebar();

  return (
    <main className="relative flex min-w-0 flex-1 flex-col bg-chat-background">
      {/* 顶部header */}
      {showChatHeader && <ChatHeader onNavigateHome={onNavigateHome} />}
      {/* 收起后和窄屏下保留展开入口。 */}
      {!showChatHeader && (isMobile || state === "collapsed") && (
        <SidebarTrigger
          className="absolute left-2 top-2 cursor-pointer"
          aria-label="展开会话列表"
        />
      )}
      {/* 中间对话框 */}
      {children}
    </main>
  );
}

export default function RootLayout({
  children,
  showChatHeader = false,
  ...navigation
}: LeftPanelProps & { children: ReactNode; showChatHeader?: boolean }) {
  return (
    <SidebarProvider
      style={
        {
          "--sidebar-width": "300px",
          "--sidebar-width-icon": "300px",
        } as CSSProperties
      }
    >
      {/* 左侧的面板 */}
      <LeftPanel {...navigation} />
      {/* 右侧的内容 */}
      <Content
        showChatHeader={showChatHeader}
        onNavigateHome={navigation.onNewSession}
      >
        {children}
      </Content>
    </SidebarProvider>
  );
}
