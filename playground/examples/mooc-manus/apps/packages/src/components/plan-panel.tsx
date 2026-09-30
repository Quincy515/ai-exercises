import { Check, ChevronDown, ChevronUp, Clock } from "lucide-react";
import { useState } from "react";
import { cn } from "../lib/utils";
import { Button } from "./ui/button";

interface PlanPanelProps {
  className?: string;
}

export function PlanPanel({ className }: PlanPanelProps) {
  // 1.定义组件状态&函数
  const [isExpanded, setIsExpanded] = useState(false);
  const togglePanel = () => setIsExpanded((expanded) => !expanded);
  // 本课使用演示步骤和固定进度，真实计划由后续业务提供。
  const steps = [
    {
      id: 1,
      status: "completed",
      description:
        "使用搜索工具查找第四季度（假设为当前或最近一年的10月至12月）国内人工智能的最新新闻和发展动态，使用中文关键词如'第四季度 AI 发展 新闻 中国'。",
    },
    {
      id: 2,
      status: "running",
      description:
        "使用搜索工具查找第四季度国际人工智能的最新新闻和发展动态，使用英文关键词如'Q4 AI developments 2023 international'或类似，以覆盖全球范围。",
    },
    {
      id: 3,
      status: "running",
      description:
        "从搜索结果中选择多个相关URL，并使用浏览器工具访问这些页面，提取页面内容以获取详细信息，包括新闻标题、日期和来源。",
    },
  ];

  return (
    <div className={cn("rounded-xl border bg-background", className)}>
      {/* 顶部留白+按钮；保留同一个按钮，切换时键盘焦点继续可用。 */}
      <div
        className={cn(isExpanded && "mb-4 flex w-full justify-end px-4 pt-4")}
      >
        <Button
          type="button"
          variant="ghost"
          size={isExpanded ? "icon-xs" : "default"}
          className={cn(
            "cursor-pointer",
            !isExpanded &&
              "h-auto w-full justify-between gap-2 rounded-xl py-0 pr-3 pl-4",
          )}
          aria-label={isExpanded ? "收起任务计划" : "展开任务计划"}
          aria-expanded={isExpanded}
          onClick={togglePanel}
        >
          {isExpanded ? (
            <ChevronDown />
          ) : (
            <>
              {/* 折叠状态 */}
              {/* 左侧的最新计划 */}
              <span className="flex min-w-0 flex-1 items-center gap-2.5 py-2 text-muted-foreground">
                <Clock />
                <span className="truncate text-sm">
                  使用搜索工具查找第四季度（假设为当前或最近一年的10月至12月）国内人工智能的最新新闻和发展动态，例如
                  xxx 等
                </span>
              </span>
              {/* 右侧操作按钮&步骤信息 */}
              <span className="flex shrink-0 items-center gap-2 py-2.5">
                <span className="text-xs text-muted-foreground">1 / 5</span>
                <ChevronUp />
              </span>
            </>
          )}
        </Button>
      </div>
      {/* 展开状态 */}
      {isExpanded && (
        <div className="rounded-xl px-4 pb-4">
          {/* 底部的计划列表 */}
          <div className="rounded-lg bg-muted/50 px-2 py-3">
            {/* 任务进度信息 */}
            <div className="flex w-full items-center justify-between gap-3 px-4">
              <span className="font-bold text-foreground/80">任务进度</span>
              <span className="text-xs text-muted-foreground">1 / 5</span>
            </div>
            {/* 任务列表 */}
            <ul
              aria-label="任务步骤"
              className="max-h-[clamp(72px,calc(100svh-360px),400px)] overflow-y-auto"
            >
              {steps.map((step) => (
                <li
                  key={step.id}
                  className="flex w-full items-center gap-2.5 px-4 py-2 text-sm text-muted-foreground"
                >
                  {/* 图标 */}
                  {step.status === "completed" ? (
                    <Check size={16} className="shrink-0" />
                  ) : (
                    <Clock size={16} className="shrink-0" />
                  )}
                  {/* 任务描述 */}
                  <span className="truncate">{step.description}</span>
                </li>
              ))}
            </ul>
          </div>
        </div>
      )}
    </div>
  );
}
