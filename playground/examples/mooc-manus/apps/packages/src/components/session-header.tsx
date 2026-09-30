import { Download, FileSearchCorner, FileText } from "lucide-react";
import { Avatar, AvatarFallback } from "./ui/avatar";
import { Button } from "./ui/button";
import {
  Dialog,
  DialogContent,
  DialogHeader,
  DialogTitle,
  DialogTrigger,
} from "./ui/dialog";
import {
  Item,
  ItemActions,
  ItemContent,
  ItemDescription,
  ItemMedia,
  ItemTitle,
} from "./ui/item";
import { ScrollArea } from "./ui/scroll-area";
import { SidebarTrigger, useSidebar } from "./ui/sidebar";

export function SessionHeader() {
  const { open, isMobile } = useSidebar();
  // 本课使用演示文件，真实列表与下载在后续接入。
  const files = [
    { id: 1, extension: "pdf", filename: "go+java.pdf", size: "2.52MB" },
    { id: 2, extension: "png", filename: "全家福.png", size: "2.52MB" },
    {
      id: 3,
      extension: "docx",
      filename: "2025年年中汇报.docx",
      size: "2.52MB",
    },
    {
      id: 4,
      extension: "xlsx",
      filename: "数据分析可视化看板.xsx",
      size: "2.52MB",
    },
    {
      id: 5,
      extension: "gif",
      filename: "数据看板动态演示.gif",
      size: "2.52MB",
    },
    { id: 6, extension: "py", filename: "ReActAgent.py", size: "2.52MB" },
  ];

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
        {/* 右侧按钮 */}
        <Dialog>
          {/* 模态窗触发器 */}
          <DialogTrigger
            render={
              <Button
                type="button"
                variant="ghost"
                size="icon-sm"
                className="cursor-pointer"
                aria-label="查看会话文件"
              />
            }
          >
            <FileSearchCorner />
          </DialogTrigger>
          {/* 模态窗内容 */}
          <DialogContent>
            <DialogHeader>
              <DialogTitle>此任务中的所有文件</DialogTitle>
            </DialogHeader>
            <ScrollArea className="h-[500px] max-h-[calc(100dvh-8rem)]">
              <div
                role="list"
                aria-label="任务文件"
                className="flex flex-col gap-1"
              >
                {files.map((file) => (
                  <Item
                    key={file.id}
                    role="listitem"
                    variant="default"
                    className="shrink-0 cursor-pointer flex-nowrap gap-2 p-2 hover:bg-gray-100"
                  >
                    {/* 左侧文件图标 */}
                    <ItemMedia>
                      <Avatar className="size-8">
                        <AvatarFallback>
                          <FileText className="size-4" />
                        </AvatarFallback>
                      </Avatar>
                    </ItemMedia>
                    {/* 文件信息 */}
                    <ItemContent className="min-w-0 gap-0">
                      <ItemTitle className="max-w-full text-sm text-gray-700">
                        <span className="truncate">{file.filename}</span>
                      </ItemTitle>
                      <ItemDescription className="text-xs">
                        {file.extension} · {file.size}
                      </ItemDescription>
                    </ItemContent>
                    <ItemActions className="shrink-0">
                      <Button
                        type="button"
                        variant="ghost"
                        size="icon-xs"
                        className="cursor-pointer"
                        aria-label={`下载 ${file.filename}`}
                      >
                        <Download />
                      </Button>
                    </ItemActions>
                  </Item>
                ))}
              </div>
            </ScrollArea>
          </DialogContent>
        </Dialog>
      </div>
      {/* 右侧占位 */}
      <div className="flex-1" />
    </header>
  );
}
