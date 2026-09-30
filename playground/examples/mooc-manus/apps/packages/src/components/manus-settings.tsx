import {
  Gift,
  Languages,
  LayoutGrid,
  LayoutList,
  Settings,
  Trash,
  Wrench,
} from "lucide-react";
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

import {
  Field,
  FieldDescription,
  FieldGroup,
  FieldLabel,
  FieldLegend,
  FieldSet,
} from "./ui/field";
import { Kbd } from "./ui/kbd";
import { Input } from "./ui/input";
import {
  Item,
  ItemContent,
  ItemDescription,
  ItemGroup,
  ItemTitle,
} from "./ui/item";
import { Badge } from "./ui/badge";
import { Switch } from "./ui/switch";
import { Textarea } from "./ui/textarea";

// 本课只展示配置 UI；表单阻止默认提交，业务保存后续接入。

export function CommonSetting() {
  return (
    <form className="w-full px-1" onSubmit={(event) => event.preventDefault()}>
      <FieldGroup>
        <FieldSet>
          {/* 顶部表单标题 */}
          <FieldLegend className="font-bold text-gray-700 data-[variant=legend]:text-lg">
            通用配置
          </FieldLegend>
          <FieldDescription className="text-sm">
            配置MoocManus系统的通用配置信息
          </FieldDescription>
          {/* 中间表单内容 */}
          <FieldGroup>
            <Field>
              <FieldLabel className="flex-wrap" htmlFor="max_iterations">
                最大迭代次数
                <Kbd>max_iterations</Kbd>
              </FieldLabel>
              <Input
                id="max_iterations"
                type="number"
                placeholder="Agent最大迭代次数"
                defaultValue={100}
                min={0}
                max={200}
                required
              />
              <FieldDescription className="text-xs">
                执行Agent最大能迭代循环调用工具的次数, 默认为100
              </FieldDescription>
            </Field>
            <Field>
              <FieldLabel className="flex-wrap" htmlFor="max_retries">
                最大重试次数
                <Kbd>max_retries</Kbd>
              </FieldLabel>
              <Input
                id="max_retries"
                type="number"
                placeholder="LLM/Tool最大重试次数"
                defaultValue={3}
                min={0}
                max={10}
                required
              />
              <FieldDescription className="text-xs">
                默认情况下，最大重试次数为3
              </FieldDescription>
            </Field>
            <Field>
              <FieldLabel className="flex-wrap" htmlFor="max_search_results">
                最大搜索结果
                <Kbd>max_search_results</Kbd>
              </FieldLabel>
              <Input
                id="max_search_results"
                type="number"
                placeholder="搜索工具返回的最大结果数"
                defaultValue={10}
                min={0}
                max={30}
                required
              />
              <FieldDescription className="text-xs">
                默认情况下, 每个搜索步骤包含10个结果
              </FieldDescription>
            </Field>
          </FieldGroup>
        </FieldSet>
      </FieldGroup>
    </form>
  );
}

export function LLMSetting() {
  return (
    <form className="w-full px-1" onSubmit={(event) => event.preventDefault()}>
      <FieldGroup>
        <FieldSet>
          {/* 顶部表单标题 */}
          <FieldLegend className="font-bold text-gray-700 data-[variant=legend]:text-lg">
            模型提供商
          </FieldLegend>
          <FieldDescription className="text-sm">
            配置Agent使用的基础LLM模型(兼容OpenAI格式)
          </FieldDescription>
          {/* 中间表单内容 */}
          <FieldGroup>
            <Field>
              <FieldLabel className="flex-wrap" htmlFor="base_url">
                提供商基础地址
                <Kbd>base_url</Kbd>
              </FieldLabel>
              <Input
                id="base_url"
                type="url"
                placeholder="请填写LLM基础URL地址, 不带/"
                defaultValue="https://api.deepseek.com"
                required
              />
              <FieldDescription className="text-xs">
                请填写模型提供商的基础 url 地址, 需兼容 OpenAI 格式
              </FieldDescription>
            </Field>
            <Field>
              <FieldLabel className="flex-wrap" htmlFor="api_key">
                提供商秘钥
                <Kbd>api_key</Kbd>
              </FieldLabel>
              <Input
                id="api_key"
                type="text"
                placeholder="请填写提供商API秘钥"
                required
              />
              <FieldDescription className="text-xs">
                请填写模型提供商秘钥信息
              </FieldDescription>
            </Field>
            <Field>
              <FieldLabel className="flex-wrap" htmlFor="model_name">
                模型名
                <Kbd>model_name</Kbd>
              </FieldLabel>
              <Input
                id="model_name"
                type="text"
                placeholder="请填写需要使用的模型名字"
                required
              />
              <FieldDescription className="text-xs">
                请填写 MoocManus 调用的模型名字,
                模型必须支持工具调用、图像识别等功能。
              </FieldDescription>
            </Field>
            <Field>
              <FieldLabel className="flex-wrap" htmlFor="temperature">
                温度
                <Kbd>temperature</Kbd>
              </FieldLabel>
              <Input
                id="temperature"
                type="number"
                placeholder="请填写模型温度信息"
                defaultValue={0.7}
                required
              />
              <FieldDescription className="text-xs">
                温度越低, 模型输出内容越确定, 创意性越差, 默认配置0.7
              </FieldDescription>
            </Field>
            <Field>
              <FieldLabel className="flex-wrap" htmlFor="max_tokens">
                最大输出token数
                <Kbd>max_tokens</Kbd>
              </FieldLabel>
              <Input
                id="max_tokens"
                type="number"
                placeholder="请填写模型最大输出token数"
                defaultValue={8192}
                min={0}
                required
              />
              <FieldDescription className="text-xs">
                设置模型的最大输出token数
              </FieldDescription>
            </Field>
          </FieldGroup>
        </FieldSet>
      </FieldGroup>
    </form>
  );
}

export function A2ASetting() {
  return (
    <div className="w-full px-1">
      <FieldGroup>
        <FieldSet>
          {/* 顶部标题 */}
          <FieldLegend className="flex w-full flex-wrap items-center justify-between gap-2 font-bold text-gray-700 data-[variant=legend]:text-lg">
            A2A Agent配置
            <Dialog>
              {/* 模态窗触发按钮 */}
              <DialogTrigger
                render={
                  <Button type="button" size="xs" className="cursor-pointer" />
                }
              >
                新增远程Agent
              </DialogTrigger>
              {/* 新增A2A服务器模态窗 */}
              <DialogContent className="max-h-[calc(100dvh-2rem)] grid-rows-[auto_minmax(0,1fr)_auto]">
                <DialogHeader>
                  <DialogTitle className="text-gray-700">
                    添加远程Agent
                  </DialogTitle>
                  <DialogDescription className="text-gray-500">
                    MoocManus 使用标准的 A2A 协议来连接远程 Agent。
                    <br />
                    请将您的配置粘贴到下方，然后点击“添加”即可添加 Agent。
                  </DialogDescription>
                </DialogHeader>
                <form
                  className="w-full min-h-0 overflow-y-auto"
                  onSubmit={(event) => event.preventDefault()}
                >
                  <FieldGroup>
                    <FieldSet>
                      <Field>
                        <Input
                          id="a2a_base_url"
                          aria-label="远程Agent地址"
                          type="url"
                          placeholder="Example: https://mooc-manus.com/weather-agent"
                        />
                      </Field>
                    </FieldSet>
                  </FieldGroup>
                </form>
                <DialogFooter>
                  <DialogClose
                    render={
                      <Button variant="outline" className="cursor-pointer" />
                    }
                  >
                    取消
                  </DialogClose>
                  <Button type="button" className="cursor-pointer">
                    添加
                  </Button>
                </DialogFooter>
              </DialogContent>
            </Dialog>
          </FieldLegend>
          <FieldDescription className="text-sm">
            模型A2A协议 (Agent to Agent Protocol) 通过集成外部 Agent 来增强
            MoocManus 的性能。
          </FieldDescription>
          {/* 中间列表内容 */}
          <ItemGroup>
            <Item variant="outline" role="listitem">
              <ItemContent className="min-w-0">
                <ItemTitle className="flex w-full flex-wrap items-center justify-between gap-2 font-bold text-gray-700">
                  {/* 左侧Agent名称 */}
                  <div className="flex min-w-0 flex-wrap items-center gap-2">
                    天气Agent
                    <Badge>禁用</Badge>
                  </div>
                  {/* 右侧基础操作 */}
                  <div className="flex items-center justify-center gap-2">
                    <Button
                      type="button"
                      variant="ghost"
                      size="icon-xs"
                      className="cursor-pointer"
                      aria-label="删除天气Agent"
                    >
                      <Trash />
                    </Button>
                    <Switch aria-label="启用天气Agent" />
                  </div>
                </ItemTitle>
                <ItemDescription>提供天气查询相关功能</ItemDescription>
                <ItemDescription className="flex flex-wrap items-center gap-x-2 gap-y-1">
                  <LayoutList size={12} />
                  <Badge variant="secondary" className="text-gray-500">
                    输入: text
                  </Badge>
                  <Badge variant="secondary" className="text-gray-500">
                    输出: text
                  </Badge>
                  <Badge variant="secondary" className="text-gray-500">
                    流式输出
                  </Badge>
                  <Badge variant="secondary" className="text-gray-500">
                    推送通知
                  </Badge>
                </ItemDescription>
              </ItemContent>
            </Item>
          </ItemGroup>
        </FieldSet>
      </FieldGroup>
    </div>
  );
}

export function MCPSetting() {
  const mcpConfigPlaceholder = `{
  "mcpServers": {
    "qiniu": {
      "command": "uvx",
      "args": [
        "qiniu-mcp-server"
      ],
      "env": {
        "QINIU_ACCESS_KEY": "YOUR_ACCESS_KEY",
        "QINIU_SECRET_KEY": "YOUR_SECRET_KEY",
        "QINIU_REGION_NAME": "YOUR_REGION_NAME",
        "QINIU_ENDPOINT_URL": "YOUR_ENDPOINT_URL",
        "QINIU_BUCKETS": ""
      },
      "disabled": false
    }
  }
}`;

  return (
    <div className="w-full px-1">
      <FieldGroup>
        <FieldSet>
          {/* 顶部标题 */}
          <FieldLegend className="flex w-full flex-wrap items-center justify-between gap-2 font-bold text-gray-700 data-[variant=legend]:text-lg">
            MCP 服务器
            <Dialog>
              {/* 模态窗触发按钮 */}
              <DialogTrigger
                render={
                  <Button type="button" size="xs" className="cursor-pointer" />
                }
              >
                新增服务器
              </DialogTrigger>
              {/* 新增MCP服务器模态窗 */}
              <DialogContent className="max-h-[calc(100dvh-2rem)] grid-rows-[auto_minmax(0,1fr)_auto]">
                <DialogHeader>
                  <DialogTitle className="text-gray-700">
                    添加新的 MCP 服务器
                  </DialogTitle>
                  <DialogDescription className="text-gray-500">
                    MoocManus 使用标准的 JSON MCP 配置来创建新服务器。
                    请将您的配置粘贴到下方，然后点击“添加”即可添加新服务器。
                  </DialogDescription>
                </DialogHeader>
                <form
                  className="w-full min-h-0 overflow-y-auto"
                  onSubmit={(event) => event.preventDefault()}
                >
                  <FieldGroup>
                    <FieldSet>
                      <Field>
                        <Textarea
                          id="mcp_config"
                          aria-label="MCP服务器配置"
                          placeholder={mcpConfigPlaceholder}
                          required
                        />
                      </Field>
                    </FieldSet>
                  </FieldGroup>
                </form>
                <DialogFooter>
                  <DialogClose
                    render={
                      <Button variant="outline" className="cursor-pointer" />
                    }
                  >
                    取消
                  </DialogClose>
                  <Button type="button" className="cursor-pointer">
                    添加
                  </Button>
                </DialogFooter>
              </DialogContent>
            </Dialog>
          </FieldLegend>
          <FieldDescription className="text-sm">
            模型上下文协议 (MCP) 通过集成外部工具来增强 MoocManus
            的性能，例如私有域搜索、网页浏览、订餐、PPT 生成等任务。
          </FieldDescription>
          {/* 中间列表内容 */}
          <ItemGroup>
            <Item variant="outline" role="listitem">
              <ItemContent className="min-w-0">
                <ItemTitle className="flex w-full flex-wrap items-center justify-between gap-2 font-bold text-gray-700">
                  {/* 左侧MCP名称 */}
                  <div className="flex min-w-0 flex-wrap items-center gap-2">
                    bilibili-video-info-mcp
                    <Badge>stdio</Badge>
                    <Badge>禁用</Badge>
                  </div>
                  {/* 右侧基础操作 */}
                  <div className="flex items-center justify-center gap-2">
                    <Button
                      type="button"
                      variant="ghost"
                      size="icon-xs"
                      className="cursor-pointer"
                      aria-label="删除bilibili-video-info-mcp"
                    >
                      <Trash />
                    </Button>
                    <Switch aria-label="启用bilibili-video-info-mcp" />
                  </div>
                </ItemTitle>
                <ItemDescription className="flex flex-wrap items-center gap-x-2 gap-y-1">
                  <Wrench size={12} />
                  <Badge variant="secondary" className="text-gray-500">
                    get_subtitles
                  </Badge>
                  <Badge variant="secondary" className="text-gray-500">
                    get_danmuku
                  </Badge>
                  <Badge variant="secondary" className="text-gray-500">
                    get_comments
                  </Badge>
                  <Badge variant="secondary" className="text-gray-500">
                    version
                  </Badge>
                  <Badge variant="secondary" className="text-gray-500">
                    list_buckets
                  </Badge>
                  <Badge variant="secondary" className="text-gray-500">
                    get_object
                  </Badge>
                  <Badge variant="secondary" className="text-gray-500">
                    get_images
                  </Badge>
                  <Badge variant="secondary" className="text-gray-500">
                    get_pngs
                  </Badge>
                </ItemDescription>
              </ItemContent>
            </Item>
          </ItemGroup>
        </FieldSet>
      </FieldGroup>
    </div>
  );
}

export function ManusSettings() {
  const [activatedSetting, setActivatedSetting] = useState("common-setting");
  const settingMenus = [
    {
      key: "common-setting",
      icon: Settings,
      title: "通用配置",
      childComponent: CommonSetting,
    },
    {
      key: "llm-setting",
      icon: Languages,
      title: "模型提供商",
      childComponent: LLMSetting,
    },
    {
      key: "a2a-setting",
      icon: LayoutGrid,
      title: "A2A Agent配置",
      childComponent: A2ASetting,
    },
    {
      key: "mcp-setting",
      icon: Gift,
      title: "MCP 服务器",
      childComponent: MCPSetting,
    },
  ];
  const ActiveSetting = settingMenus.find(
    (setting) => setting.key === activatedSetting,
  )?.childComponent;

  return (
    <Dialog>
      {/* 模态窗触发器 */}
      <DialogTrigger
        render={
          <Button variant="ghost" size="icon-lg" className="cursor-pointer" />
        }
        aria-label="打开设置"
        title="设置"
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
        <div className="flex min-h-0 min-w-0 flex-col gap-4 sm:flex-row">
          {/* 左侧快捷菜单 */}
          <div className="scrollbar-hide shrink-0 overflow-x-auto sm:max-w-[180px] sm:overflow-y-auto">
            <div className="flex gap-0 sm:flex-col">
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
          <Separator orientation="vertical" className="hidden sm:block" />
          {/* 右侧表单内容 */}
          <div className="scrollbar-hide h-[500px] max-h-full min-h-0 min-w-0 flex-1 overflow-y-auto wrap-break-word">
            {ActiveSetting && <ActiveSetting />}
          </div>
        </div>
        {/* 模态窗footer */}
        <DialogFooter className="mx-0 mb-0 bg-transparent p-0 pt-4">
          <DialogClose
            render={<Button variant="outline" className="cursor-pointer" />}
          >
            取消
          </DialogClose>
          <Button type="button" className="cursor-pointer">
            保存
          </Button>
        </DialogFooter>
      </DialogContent>
    </Dialog>
  );
}
