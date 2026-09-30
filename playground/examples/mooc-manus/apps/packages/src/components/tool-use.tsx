import { SquareChevronRight } from "lucide-react";

export function ToolUse() {
  return (
    <>
      <p className="overflow-hidden text-sm text-ellipsis whitespace-pre-line text-gray-500">
        我将使用文件写入工具来创建一个包含冒泡排序算法的Golang程序文件。
      </p>
      <div className="group flex cursor-pointer items-center gap-2">
        {/* 左侧工具信息 */}
        <div className="min-w-0 flex-1">
          <div className="inline-flex max-w-full items-center gap-2 rounded-[15px] border bg-gray-100 px-[10px] py-[3px]">
            {/* 图标信息 */}
            <SquareChevronRight size={21} className="shrink-0 text-gray-700" />
            {/* 工具信息 */}
            <div className="flex min-w-0 items-center text-xs text-gray-700">
              <span className="shrink-0">正在写入文件</span>
              <span className="ml-1 min-w-0 truncate rounded-[6px] px-1 font-mono text-gray-500">
                <code>bubble_sort.go</code>
              </span>
            </div>
          </div>
        </div>
        {/* 右侧时间 */}
        <div className="invisible shrink-0 text-xs whitespace-nowrap text-gray-500 transition group-hover:visible">
          2个月前
        </div>
      </div>
    </>
  );
}
