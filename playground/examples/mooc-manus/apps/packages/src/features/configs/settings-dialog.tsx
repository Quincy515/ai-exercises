import { Gift, Languages, LayoutGrid, Settings } from "lucide-react";
import { useCallback, useLayoutEffect, useRef, useState } from "react";
import { Button } from "../../components/ui/button";
import {
  Dialog,
  DialogClose,
  DialogContent,
  DialogDescription,
  DialogFooter,
  DialogHeader,
  DialogTitle,
  DialogTrigger,
} from "../../components/ui/dialog";
import { Separator } from "../../components/ui/separator";
import { AGENT_CONFIG_FORM_ID, AgentConfigPanel } from "./agent-config-panel";
import { MCPSetting } from "./other-settings-panels";
import { useConfigs } from "./use-configs";
import { A2aConfigPanel } from "./a2a-config-panel";
import { LLM_CONFIG_FORM_ID, LlmConfigPanel } from "./llm-config-panel";

const menus = [
  { key: "common-setting", icon: Settings, title: "通用配置" },
  { key: "llm-setting", icon: Languages, title: "模型提供商" },
  { key: "a2a-setting", icon: LayoutGrid, title: "A2A Agent配置" },
  { key: "mcp-setting", icon: Gift, title: "MCP 服务器" },
] as const;
type SettingKey = (typeof menus)[number]["key"];

function SettingsContent({
  active,
  onSelect,
  onSavingChange,
}: {
  active: SettingKey;
  onSelect: (key: SettingKey) => void;
  onSavingChange: (saving: boolean) => void;
}) {
  const configs = useConfigs(active);
  const config = active === "llm-setting" ? configs.llm : configs.agent;
  const saving =
    configs.agent.saving || configs.llm.saving || configs.a2a.saving;
  useLayoutEffect(() => {
    onSavingChange(saving);
    return () => onSavingChange(false);
  }, [saving, onSavingChange]);

  return (
    <>
      <DialogHeader className="border-b pb-4">
        <DialogTitle className="text-gray-700">MoocManus 设置</DialogTitle>
        <DialogDescription className="text-gray-500">
          在此管理您的 MoocManus 设置。
        </DialogDescription>
      </DialogHeader>
      <div className="flex min-h-0 min-w-0 flex-col gap-4 sm:flex-row">
        <div className="scrollbar-hide shrink-0 overflow-x-auto sm:max-w-[180px] sm:overflow-y-auto">
          <div className="flex gap-0 sm:flex-col">
            {menus.map((menu) => (
              <Button
                key={menu.key}
                variant={active === menu.key ? "default" : "ghost"}
                className="cursor-pointer justify-start"
                aria-pressed={active === menu.key}
                disabled={saving}
                onClick={() => onSelect(menu.key)}
              >
                <menu.icon data-icon="inline-start" />
                {menu.title}
              </Button>
            ))}
          </div>
        </div>
        <Separator orientation="vertical" className="hidden sm:block" />
        <div className="scrollbar-hide h-[500px] max-h-full min-h-0 min-w-0 flex-1 overflow-y-auto wrap-break-word">
          {active === "common-setting" && (
            <AgentConfigPanel config={configs.agent} />
          )}
          {active === "llm-setting" && <LlmConfigPanel config={configs.llm} />}
          {active === "a2a-setting" && <A2aConfigPanel config={configs.a2a} />}
          {active === "mcp-setting" && <MCPSetting />}
        </div>
      </div>
      <DialogFooter className="mx-0 mb-0 bg-transparent p-0 pt-4">
        <DialogClose
          disabled={saving}
          render={<Button variant="outline" className="cursor-pointer" />}
        >
          {active === "a2a-setting" ? "关闭" : "取消"}
        </DialogClose>
        {active !== "a2a-setting" && (
          <Button
            type="submit"
            form={
              active === "llm-setting"
                ? LLM_CONFIG_FORM_ID
                : AGENT_CONFIG_FORM_ID
            }
            className="cursor-pointer"
            disabled={
              (active !== "common-setting" && active !== "llm-setting") ||
              !config.ready ||
              !config.canSave
            }
          >
            {saving ? "保存中…" : "保存"}
          </Button>
        )}
      </DialogFooter>
    </>
  );
}

export function ManusSettings() {
  const [active, setActive] = useState<SettingKey>("common-setting");
  const saving = useRef(false);
  const onSavingChange = useCallback((next: boolean) => {
    saving.current = next;
  }, []);

  return (
    <Dialog
      onOpenChange={(_open, details) => {
        // 保存中的关闭（含 Escape、遮罩与右上角按钮）统一等待结果。
        if (saving.current) details.cancel();
      }}
    >
      <DialogTrigger
        render={
          <Button variant="ghost" size="icon-lg" className="cursor-pointer" />
        }
        aria-label="打开设置"
        title="设置"
      >
        <Settings />
      </DialogTrigger>
      <DialogContent className="max-h-[calc(100dvh-2rem)] w-[calc(100%-2rem)] !max-w-[850px] grid-rows-[auto_minmax(0,1fr)_auto]">
        <SettingsContent
          active={active}
          onSelect={setActive}
          onSavingChange={onSavingChange}
        />
      </DialogContent>
    </Dialog>
  );
}
