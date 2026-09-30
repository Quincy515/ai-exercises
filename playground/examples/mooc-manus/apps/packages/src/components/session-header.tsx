import { FileSearchCorner } from "lucide-react";
import { Button } from "./ui/button";
import { SidebarTrigger, useSidebar } from "./ui/sidebar";

export function SessionHeader() {
  const { open, isMobile } = useSidebar();

  return (
    <header className="sticky top-0 z-10 flex shrink-0 items-center justify-between gap-1 bg-chat-background pt-3 pb-2">
      {/* 左侧操作按钮 */}
      <div className="flex flex-1 items-center">
        {(!open || isMobile) && (
          <SidebarTrigger
            className="cursor-pointer"
            aria-label="展开会话列表"
          />
        )}
      </div>
      {/* 中间会话标题区 */}
      <div className="flex w-full min-w-0 max-w-[768px] items-center justify-between gap-1 overflow-hidden">
        {/* 左侧标题 */}
        <h1 className="truncate text-lg text-foreground/80">
          编写Python冒泡排序算法编写Python冒泡排序算法编写Python冒泡排序算法编写Python冒泡排序算法
        </h1>
        {/* 右侧按钮；会话文件列表在后续接入。 */}
        <Button
          type="button"
          variant="ghost"
          size="icon-sm"
          className="cursor-pointer"
          aria-label="查看会话文件"
        >
          <FileSearchCorner />
        </Button>
      </div>
      {/* 右侧占位 */}
      <div className="flex-1" />
    </header>
  );
}
