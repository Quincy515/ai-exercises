import type { CSSProperties, ReactNode } from "react";
import { LeftPanel } from "../components/left-panel";
import type { LeftPanelProps } from "../components/left-panel";
import {
  SidebarProvider,
  SidebarTrigger,
  useSidebar,
} from "../components/ui/sidebar";

function Content({ children }: { children: ReactNode }) {
  const { isMobile, state } = useSidebar();

  return (
    <main className="relative min-w-0 flex-1">
      {/* 收起后和窄屏下保留展开入口。 */}
      {(isMobile || state === "collapsed") && (
        <SidebarTrigger
          className="absolute left-2 top-2 cursor-pointer"
          aria-label="展开会话列表"
        />
      )}
      {children}
    </main>
  );
}

export default function RootLayout({
  children,
  ...navigation
}: LeftPanelProps & { children: ReactNode }) {
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
      <Content>{children}</Content>
    </SidebarProvider>
  );
}
