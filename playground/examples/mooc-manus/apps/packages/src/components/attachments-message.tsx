import { Eye, FileSearch, FileText } from "lucide-react";
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

interface AttachmentsMessageProps {
  className?: string;
  role: string;
}

export function AttachmentsMessage({
  className,
  role,
}: AttachmentsMessageProps) {
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

  // 1.判断角色是否为user，如果是则渲染用户附件列表
  const isUser = role === "user";
  if (!isUser && role !== "assistant") return null;

  return (
    <div
      role="group"
      aria-label={isUser ? "用户附件" : "AI附件"}
      className={cn(
        "flex w-full min-w-0 flex-col gap-2",
        isUser ? "items-end" : "items-start",
        className,
      )}
    >
      <div
        className={cn(
          "flex w-full max-w-[568px] flex-wrap gap-2",
          isUser && "justify-end",
        )}
      >
        {files.map((file) => (
          <Item
            key={file.id}
            variant="outline"
            className="w-[280px] max-w-full shrink-0 flex-nowrap gap-2 bg-background p-2"
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
                aria-label={`预览 ${file.filename}`}
              >
                <Eye />
              </Button>
            </ItemActions>
          </Item>
        ))}
        {/* 2.渲染AI附件列表；复用文件卡片，增加查看全部入口。 */}
        {!isUser && (
          <Button
            type="button"
            variant="outline"
            className="max-w-full cursor-pointer"
          >
            <FileSearch data-icon="inline-start" />
            <span className="text-sm text-gray-700">
              查看此任务中所有的文件
            </span>
          </Button>
        )}
      </div>
    </div>
  );
}
