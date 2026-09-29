import { Gift, Languages, LayoutGrid, Settings } from "lucide-react";
import { useState } from "react";
import { Button } from "./ui/button";
import {
  Dialog,
  DialogClose,
  DialogContent,
  DialogDescription,
  DialogFooter,
  DialogHeader,
  DialogTitle,
  DialogTrigger,
} from "./ui/dialog";
import { Separator } from "./ui/separator";

export function ManusSettings() {
  const [activatedSetting, setActivatedSetting] = useState("common-setting");
  const settingMenus = [
    {
      key: "common-setting",
      icon: Settings,
      title: "通用配置",
      childComponent: null,
    },
    {
      key: "llm-setting",
      icon: Languages,
      title: "模型提供商",
      childComponent: null,
    },
    {
      key: "a2a-setting",
      icon: LayoutGrid,
      title: "A2A Agent配置",
      childComponent: null,
    },
    {
      key: "mcp-setting",
      icon: Gift,
      title: "MCP 服务器",
      childComponent: null,
    },
  ];

  return (
    <Dialog>
      {/* 模态窗触发器 */}
      <DialogTrigger
        render={
          <Button variant="outline" size="icon-sm" className="cursor-pointer" />
        }
        aria-label="打开设置"
      >
        <Settings />
      </DialogTrigger>
      {/* 模态窗本身 */}
      <DialogContent className="max-h-[calc(100dvh-2rem)] w-[calc(100%-2rem)] !max-w-[850px] grid-rows-[auto_minmax(0,1fr)_auto]">
        {/* 模态窗header */}
        <DialogHeader className="border-b pb-4">
          <DialogTitle className="text-gray-700">MoocManus 设置</DialogTitle>
          <DialogDescription className="text-gray-500">
            在此管理您的 MoocManus 设置。
          </DialogDescription>
        </DialogHeader>
        {/* 模态窗中间内容 */}
        <div className="flex min-h-0 flex-row gap-4">
          {/* 左侧快捷菜单 */}
          <div className="scrollbar-hide max-w-[180px] shrink-0 overflow-y-auto">
            <div className="flex flex-col gap-0">
              {settingMenus.map((setting) => (
                <Button
                  variant={
                    activatedSetting === setting.key ? "default" : "ghost"
                  }
                  key={setting.key}
                  className="cursor-pointer justify-start"
                  aria-pressed={activatedSetting === setting.key}
                  onClick={() => setActivatedSetting(setting.key)}
                >
                  <setting.icon data-icon="inline-start" />
                  {setting.title}
                </Button>
              ))}
            </div>
          </div>
          {/* 分隔符 */}
          <Separator orientation="vertical" />
          {/* 右侧表单内容 */}
          <div className="scrollbar-hide h-[500px] max-h-full min-w-0 flex-1 overflow-y-auto wrap-break-word">
            {activatedSetting}
          </div>
        </div>
        {/* 模态窗footer */}
        <DialogFooter className="mx-0 mb-0 bg-transparent p-0 pt-4">
          <DialogClose
            render={<Button variant="outline" className="cursor-pointer" />}
          >
            取消
          </DialogClose>
          <Button className="cursor-pointer">保存</Button>
        </DialogFooter>
      </DialogContent>
    </Dialog>
  );
}
