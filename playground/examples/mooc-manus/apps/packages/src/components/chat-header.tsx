import { SidebarTrigger, useSidebar } from "./ui/sidebar";

export function ChatHeader({ onNavigateHome }: { onNavigateHome: () => void }) {
  const { open, isMobile } = useSidebar();

  return (
    <header className="flex w-full shrink-0 items-center justify-between px-4 py-2">
      {/* 左侧操作&logo */}
      <div className="flex items-center gap-2">
        {/* 面板操作按钮: 关闭面板&移动端下会显示 */}
        {(!open || isMobile) && (
          <SidebarTrigger
            className="cursor-pointer"
            aria-label="展开会话列表"
          />
        )}
        {/* Logo占位符 */}
        <button
          type="button"
          aria-label="返回首页"
          className="block h-9 w-[80px] cursor-pointer rounded-md bg-background outline-none focus-visible:ring-2 focus-visible:ring-ring"
          onClick={onNavigateHome}
        />
      </div>
    </header>
  );
}
