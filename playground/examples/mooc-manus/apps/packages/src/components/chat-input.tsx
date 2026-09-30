import { ArrowUp, FileText, Plus, XCircle } from "lucide-react";
import { cn } from "../lib/utils";
import { Avatar, AvatarFallback } from "./ui/avatar";
import { Button } from "./ui/button";
import {
  Item,
  ItemActions,
  ItemContent,
  ItemDescription,
  ItemMedia,
  ItemTitle,
} from "./ui/item";
import { ScrollArea } from "./ui/scroll-area";
import {
  Tooltip,
  TooltipContent,
  TooltipProvider,
  TooltipTrigger,
} from "./ui/tooltip";

interface ChatInputProps {
  className?: string;
}

export function ChatInput({ className }: ChatInputProps) {
  // 本课使用演示附件，上传、移除和发送在后续接入业务。
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
    <div
      className={cn(
        "flex w-full min-w-0 flex-col rounded-2xl bg-background py-3",
        className,
      )}
    >
      {/* 顶部的文件列表 */}
      <div className="mb-1 w-full min-w-0 px-4">
        <ScrollArea className="w-full whitespace-nowrap">
          <div
            role="list"
            aria-label="演示附件"
            className="flex w-max gap-4 pb-4"
          >
            {files.map((file) => (
              <Item
                key={file.id}
                role="listitem"
                variant="muted"
                className="w-auto shrink-0 flex-nowrap gap-2 p-2"
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
                <ItemContent className="gap-0">
                  <ItemTitle className="text-sm text-gray-700">
                    {file.filename}
                  </ItemTitle>
                  <ItemDescription className="text-xs">
                    {file.extension} · {file.size}
                  </ItemDescription>
                </ItemContent>
                <ItemActions>
                  <Button
                    type="button"
                    variant="ghost"
                    size="icon-xs"
                    className="cursor-pointer"
                    aria-label={`移除附件 ${file.filename}`}
                  >
                    <XCircle />
                  </Button>
                </ItemActions>
              </Item>
            ))}
          </div>
          {/* <ScrollBar orientation="horizontal" /> */}
        </ScrollArea>
      </div>
      {/* 中间输入框 */}
      <div className="mb-3 px-4">
        <textarea
          rows={2}
          aria-label="任务内容"
          placeholder="给MoocManus一个任务"
          className="scrollbar-hide h-[46px] min-h-[40px] w-full resize-none text-sm outline-none"
        />
      </div>
      {/* 底部上传&发送按钮 */}
      <footer className="flex w-full flex-row justify-between px-3">
        {/* 上传按钮 */}
        <div className="flex gap-2">
          <TooltipProvider>
            <Tooltip>
              <TooltipTrigger
                render={
                  <Button
                    type="button"
                    variant="ghost"
                    size="icon"
                    className="cursor-pointer rounded-full"
                    aria-label="上传附件"
                  />
                }
              >
                <Plus />
              </TooltipTrigger>
              <TooltipContent>添加文件等内容</TooltipContent>
            </Tooltip>
          </TooltipProvider>
        </div>
        {/* 发送按钮 */}
        <div className="flex gap-2">
          <Button
            type="button"
            variant="ghost"
            size="icon"
            className="cursor-pointer rounded-full"
            aria-label="发送消息"
          >
            <ArrowUp />
          </Button>
        </div>
      </footer>
    </div>
  );
}
