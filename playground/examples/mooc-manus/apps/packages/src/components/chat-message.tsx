import { CheckIcon, ChevronDown, Languages } from "lucide-react";
import { cn } from "../lib/utils";
import { ManusIcon } from "./manus-icon";
import { ToolUse } from "./tool-use";
import { Button } from "./ui/button";

interface ChatMessageProps {
  className?: string;
  message: {
    type: string;
    role?: string;
  };
}

export function ChatMessage({ className, message }: ChatMessageProps) {
  // 1.消息为user时显示组件
  if (message.type === "user") {
    return (
      <div
        className={cn(
          "group mt-3 flex w-full flex-col items-end justify-end gap-1",
          className,
        )}
      >
        {/* 顶部时间 */}
        <div className="invisible flex items-center justify-end gap-1 text-xs text-gray-500 group-hover:visible">
          2个月前
        </div>
        {/* 底部用户消息 */}
        <div className="relative flex max-w-[90%] flex-col items-end gap-2">
          <div className="relative flex items-center overflow-hidden rounded-lg border bg-background p-3 text-gray-700">
            帮我写一个Python版本的冒泡排序
          </div>
        </div>
      </div>
    );
  } else if (message.type === "assistant") {
    // 2.消息为AI时显示组件
    return (
      <div className={cn("group mt-3 flex w-full flex-col gap-2", className)}>
        {/* AI图标&时间 */}
        <div className="flex h-7 items-center justify-between">
          <div className="flex items-center justify-center gap-1 text-gray-700">
            <Languages size={18} />
            <ManusIcon />
          </div>
          <div className="invisible flex items-center gap-[3px] text-xs text-gray-500 group-hover:visible">
            2个月前
          </div>
        </div>
        {/* AI消息 */}
        <div className="m-0 max-w-none p-0 text-gray-700">
          用户请求编写一个Python版本的冒泡排序算法。冒泡排序是一种简单的排序算法，通过重复遍历列表，比较相邻元素并交换它们的位置，直到列表完全排序。我将创建一个Python脚本来实现这个算法，包括必要的注释和示例使用。这个任务需要编写代码并保存到文件中，以便后续执行或修改。
        </div>
      </div>
    );
  } else if (message.type === "tool") {
    // 3.消息为工具时显示组件
    return <ToolUse />;
  } else if (message.type === "step") {
    // 4.消息为子步骤时显示组件
    return (
      <div className={cn("flex min-w-0 flex-col", className)}>
        {/* 步骤描述 */}
        <div className="group/header flex w-full min-w-0 justify-between gap-2 text-sm text-gray-700">
          <div className="flex min-w-0 items-center justify-center gap-2">
            {/* 已完成状态/未完成状态 */}
            <div
              role="img"
              aria-label="已完成"
              className="flex size-4 shrink-0 items-center justify-center rounded-[15px] border bg-gray-300"
            >
              <CheckIcon className="text-white" size={10} />
            </div>
            {/* 步骤描述 */}
            <div className="truncate font-medium">
              编写一个Golang程序文件，实现冒泡排序算法，包括必要的函数和主函数
            </div>
            {/* 展开or折叠icon；本课保留入口，交互在后续接入。 */}
            <Button
              type="button"
              variant="ghost"
              size="icon-xs"
              className="cursor-pointer"
              aria-label="步骤展开或折叠"
            >
              <ChevronDown />
            </Button>
          </div>
        </div>
        {/* 步骤详情 */}
        <div className="flex min-w-0">
          <div className="relative w-6 shrink-0" aria-hidden="true">
            <div className="absolute start-[8px] top-0 h-[calc(100%+14px)] border-l border-dashed" />
          </div>
          {/* 调用工具列表信息 */}
          <div className="flex min-w-0 flex-1 flex-col gap-3 overflow-hidden pt-2 transition-[max-height,opacity] duration-150 ease-in-out">
            {[1, 2, 3, 4].map((item) => (
              <ToolUse key={item} />
            ))}
          </div>
        </div>
      </div>
    );
  } else if (message.type === "attachments") {
    return <div className={className}>附件消息</div>;
  }

  return null;
}
