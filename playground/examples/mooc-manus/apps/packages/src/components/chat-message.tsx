import { Languages } from "lucide-react";
import { cn } from "../lib/utils";
import { ManusIcon } from "./manus-icon";
import { ToolUse } from "./tool-use";

interface ChatMessageProps {
  className?: string;
  message: {
    type: string;
    role?: string;
  };
}

export function ChatMessage({ className, message }: ChatMessageProps) {
  // 1.消息类型为user时
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
    return <ToolUse />;
  } else if (message.type === "step") {
    return <div className={className}>步骤/子任务消息</div>;
  } else if (message.type === "attachments") {
    return <div className={className}>附件消息</div>;
  }

  return null;
}
