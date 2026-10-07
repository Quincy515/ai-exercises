import { useEffect, useRef, useState } from "react";
import { Wrench, Trash } from "lucide-react";
import { Badge } from "../../components/ui/badge";
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
import {
  Field,
  FieldDescription,
  FieldGroup,
  FieldLabel,
  FieldLegend,
  FieldSet,
} from "../../components/ui/field";
import { Textarea } from "../../components/ui/textarea";
import {
  Item,
  ItemContent,
  ItemDescription,
  ItemGroup,
  ItemTitle,
} from "../../components/ui/item";
import { Switch } from "../../components/ui/switch";
import type { McpConfigController } from "./use-configs";

type McpServer = McpConfigController["servers"][number];

const mcpConfigExample = JSON.stringify(
  {
    mcpServers: {
      example: {
        transport: "streamable_http",
        url: "https://example.com/mcp",
        enabled: true,
        headers: {},
      },
    },
  },
  null,
  2,
);

function DeleteServer({
  server,
  config,
  disabled,
  open,
  onOpenChange,
  onClosed,
  returnFocus,
}: {
  server: McpServer | null;
  config: McpConfigController;
  disabled: boolean;
  open: boolean;
  onOpenChange: (open: boolean) => void;
  onClosed: () => void;
  returnFocus: () => HTMLElement | null;
}) {
  return (
    <Dialog
      open={open}
      onOpenChange={(next, details) => {
        if (config.saving) {
          details.cancel();
          return;
        }
        onOpenChange(next);
      }}
      onOpenChangeComplete={(next) => {
        if (!next) onClosed();
      }}
    >
      <DialogContent finalFocus={returnFocus}>
        <DialogHeader>
          <DialogTitle>删除 MCP 服务器</DialogTitle>
          <DialogDescription>
            将从 MoocManus 移除「{server?.server_name}」的配置。
          </DialogDescription>
        </DialogHeader>
        {config.error && (
          <p role="alert" className="text-sm text-destructive">
            {config.error}
          </p>
        )}
        {config.writeUncertain && (
          <Button
            type="button"
            variant="outline"
            disabled={config.loading}
            onClick={config.refresh}
          >
            刷新列表核对
          </Button>
        )}
        <DialogFooter>
          <DialogClose
            disabled={config.saving}
            render={<Button variant="outline" />}
          >
            取消
          </DialogClose>
          <Button
            type="button"
            variant="destructive"
            disabled={disabled}
            onClick={() => {
              if (server) config.remove(server.server_name);
            }}
          >
            {config.saving ? "正在删除…" : "确认删除"}
          </Button>
        </DialogFooter>
      </DialogContent>
    </Dialog>
  );
}

export function McpConfigPanel({ config }: { config: McpConfigController }) {
  const [addOpen, setAddOpen] = useState(false);
  const [deleteTarget, setDeleteTarget] = useState<McpServer | null>(null);
  const [deleteOpen, setDeleteOpen] = useState(false);
  const panel = useRef<HTMLDivElement>(null);
  const refreshButton = useRef<HTMLButtonElement>(null);
  const busy = config.loading || config.saving;
  const disabled =
    !config.ready || !config.loaded || busy || config.writeUncertain;

  useEffect(() => {
    if (config.created) setAddOpen(false);
  }, [config.created]);

  // 删除行后保持弹窗挂载，等写后刷新结束再关闭，焦点回到稳定的父级元素。
  useEffect(() => {
    if (
      deleteOpen &&
      deleteTarget &&
      !busy &&
      !config.servers.some(
        (server) => server.server_name === deleteTarget.server_name,
      )
    )
      setDeleteOpen(false);
  }, [deleteOpen, deleteTarget, busy, config.servers]);

  return (
    <div
      ref={panel}
      tabIndex={-1}
      className="w-full px-1 outline-none"
      aria-busy={busy}
    >
      <FieldGroup>
        <FieldSet>
          <FieldLegend className="flex w-full flex-wrap items-center justify-between gap-2 font-bold text-foreground data-[variant=legend]:text-lg">
            MCP 服务器
            <Dialog
              open={addOpen}
              onOpenChange={(open, details) => {
                if (config.saving) {
                  details.cancel();
                  return;
                }
                config.resetDraft();
                setAddOpen(open);
              }}
            >
              <DialogTrigger
                render={
                  <Button
                    type="button"
                    size="xs"
                    disabled={disabled}
                    className="cursor-pointer"
                  />
                }
              >
                添加配置
              </DialogTrigger>
              <DialogContent
                finalFocus={() => panel.current}
                className="max-h-[calc(100dvh-2rem)] sm:max-w-xl grid-rows-[auto_minmax(0,1fr)_auto]"
              >
                <DialogHeader>
                  <DialogTitle>添加或更新 MCP 服务器</DialogTitle>
                  <DialogDescription>
                    粘贴 mcpServers JSON，支持一次添加多个服务器。
                    同名服务器将整体替换，请提供完整配置，包括所需的环境变量和请求头。
                  </DialogDescription>
                </DialogHeader>
                <form
                  id="mcp-create-form"
                  className="min-h-0 overflow-y-auto"
                  noValidate
                  onSubmit={(event) => {
                    event.preventDefault();
                    config.create();
                  }}
                >
                  <FieldGroup>
                    <Field data-invalid={Boolean(config.error)}>
                      <FieldLabel htmlFor="mcp_config">
                        MCP服务器配置
                      </FieldLabel>
                      <Textarea
                        id="mcp_config"
                        aria-label="MCP服务器配置"
                        aria-describedby="mcp-config-help"
                        aria-invalid={Boolean(config.error)}
                        autoComplete="off"
                        spellCheck={false}
                        className="min-h-64 font-mono text-xs"
                        placeholder={mcpConfigExample}
                        value={config.draftJson}
                        disabled={busy || config.writeUncertain}
                        onChange={(event) =>
                          config.editJson(event.target.value)
                        }
                      />
                      <FieldDescription id="mcp-config-help">
                        transport 支持 stdio 和 streamable_http，enabled
                        控制启用状态。 stdio 命令在后端所在机器执行；HTTP
                        服务填写完整 MCP 地址。
                        输入仅在本次弹窗中保留，关闭后清空。
                      </FieldDescription>
                    </Field>
                    {config.error && (
                      <p role="alert" className="text-sm text-destructive">
                        {config.error}
                      </p>
                    )}
                    {config.writeUncertain && (
                      <Button
                        type="button"
                        variant="outline"
                        disabled={config.loading}
                        onClick={config.refresh}
                      >
                        刷新列表核对
                      </Button>
                    )}
                    {config.notice && (
                      <p className="text-sm text-muted-foreground">
                        {config.notice}
                      </p>
                    )}
                  </FieldGroup>
                </form>
                <DialogFooter>
                  <DialogClose
                    disabled={config.saving}
                    render={<Button variant="outline" />}
                  >
                    取消
                  </DialogClose>
                  <Button
                    type="submit"
                    form="mcp-create-form"
                    disabled={disabled || !config.draftJson.trim()}
                  >
                    {config.saving ? "正在保存…" : "保存配置"}
                  </Button>
                </DialogFooter>
              </DialogContent>
            </Dialog>
          </FieldLegend>
          <FieldDescription>
            通过 MCP 接入外部工具。添加、更新、启停和删除会立即保存。
          </FieldDescription>
          <div className="flex flex-wrap items-center justify-between gap-3">
            <p role="status" className="text-sm text-muted-foreground">
              {config.saving
                ? "正在保存 MCP 服务器…"
                : config.loading
                  ? "正在读取 MCP 服务器…"
                  : (config.notice ??
                    (config.loaded
                      ? "已读取 MCP 服务器列表"
                      : "等待加载 MCP 服务器"))}
            </p>
            <Button
              ref={refreshButton}
              type="button"
              size="sm"
              variant="outline"
              disabled={!config.ready || busy}
              onClick={config.refresh}
            >
              刷新
            </Button>
          </div>
          {!addOpen && config.error && (
            <p role="alert" className="text-sm text-destructive">
              {config.error}
            </p>
          )}
          {config.loaded && !config.servers.length && !busy && (
            <p className="py-6 text-sm text-muted-foreground">
              暂无 MCP 服务器，点击“添加配置”接入工具。
            </p>
          )}
          <ItemGroup role="list" aria-label="MCP服务器列表">
            {config.servers.map((server) => (
              <Item
                key={server.server_name}
                data-mcp-name={server.server_name}
                data-mcp-transport={server.transport}
                variant="outline"
                role="listitem"
              >
                <ItemContent className="min-w-0">
                  <ItemTitle className="flex w-full flex-wrap items-center justify-between gap-2 font-bold text-foreground">
                    <div className="flex min-w-0 flex-wrap items-center gap-2">
                      <span className="wrap-break-word">
                        {server.server_name}
                      </span>
                      <Badge variant="outline">
                        {server.transport === "streamable_http"
                          ? "Streamable HTTP"
                          : "stdio"}
                      </Badge>
                      <Badge variant={server.enabled ? "default" : "secondary"}>
                        {server.enabled ? "已启用" : "已禁用"}
                      </Badge>
                    </div>
                    <div className="flex items-center gap-2">
                      <Button
                        type="button"
                        variant="ghost"
                        size="icon-xs"
                        disabled={disabled}
                        className="cursor-pointer"
                        aria-label={`删除 ${server.server_name}`}
                        onClick={() => {
                          setDeleteTarget(server);
                          setDeleteOpen(true);
                        }}
                      >
                        <Trash />
                      </Button>
                      <Switch
                        aria-label={`启用 ${server.server_name}`}
                        checked={server.enabled}
                        disabled={disabled}
                        onCheckedChange={(enabled) =>
                          config.setEnabled(server.server_name, enabled)
                        }
                      />
                    </div>
                  </ItemTitle>
                  <ItemDescription className="flex flex-wrap items-center gap-x-2 gap-y-1">
                    <Wrench size={12} />
                    {server.tools.length ? (
                      server.tools.map((tool, index) => (
                        <Badge
                          key={`${tool}-${index}`}
                          variant="secondary"
                          className="max-w-full whitespace-normal wrap-break-word"
                        >
                          {tool}
                        </Badge>
                      ))
                    ) : (
                      <span>暂未读取到工具，可检查连接后刷新。</span>
                    )}
                  </ItemDescription>
                </ItemContent>
              </Item>
            ))}
          </ItemGroup>
          <DeleteServer
            server={deleteTarget}
            config={config}
            disabled={disabled}
            open={deleteOpen}
            onOpenChange={setDeleteOpen}
            onClosed={() => setDeleteTarget(null)}
            returnFocus={() =>
              refreshButton.current?.disabled
                ? panel.current
                : refreshButton.current
            }
          />
        </FieldSet>
      </FieldGroup>
    </div>
  );
}
