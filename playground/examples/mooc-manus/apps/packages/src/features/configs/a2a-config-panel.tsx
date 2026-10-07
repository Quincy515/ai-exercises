import { useEffect, useRef, useState } from "react";
import { LayoutList, Trash } from "lucide-react";
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
import { Input } from "../../components/ui/input";
import {
  Item,
  ItemContent,
  ItemDescription,
  ItemGroup,
  ItemTitle,
} from "../../components/ui/item";
import { Switch } from "../../components/ui/switch";
import type { A2aConfigController } from "./use-configs";

type A2aServer = A2aConfigController["servers"][number];

function DeleteAgent({
  server,
  config,
  disabled,
  open,
  onOpenChange,
  onClosed,
  returnFocus,
}: {
  server: A2aServer | null;
  config: A2aConfigController;
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
          <DialogTitle>删除远程Agent</DialogTitle>
          <DialogDescription>
            将从 MoocManus 移除「{server?.name || "未命名 Agent"}」的配置。
          </DialogDescription>
        </DialogHeader>
        <p className="text-xs text-muted-foreground wrap-break-word">
          标识：{server?.id}
        </p>
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
              if (server) config.remove(server.id);
            }}
          >
            {config.saving ? "正在删除…" : "确认删除"}
          </Button>
        </DialogFooter>
      </DialogContent>
    </Dialog>
  );
}

export function A2aConfigPanel({ config }: { config: A2aConfigController }) {
  const [addOpen, setAddOpen] = useState(false);
  const [deleteTarget, setDeleteTarget] = useState<A2aServer | null>(null);
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
      !config.servers.some((server) => server.id === deleteTarget.id)
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
            A2A Agent配置
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
                新增远程Agent
              </DialogTrigger>
              <DialogContent
                finalFocus={() => panel.current}
                className="max-h-[calc(100dvh-2rem)] grid-rows-[auto_minmax(0,1fr)_auto]"
              >
                <DialogHeader>
                  <DialogTitle>添加远程Agent</DialogTitle>
                  <DialogDescription>
                    填写远程 Agent 的基础地址，添加后由后端读取 Agent Card。
                  </DialogDescription>
                </DialogHeader>
                <form
                  id="a2a-create-form"
                  className="min-h-0 overflow-y-auto"
                  noValidate
                  onSubmit={(event) => {
                    event.preventDefault();
                    config.create();
                  }}
                >
                  <FieldGroup>
                    <Field>
                      <FieldLabel htmlFor="a2a_base_url">
                        远程Agent地址
                      </FieldLabel>
                      <Input
                        id="a2a_base_url"
                        aria-label="远程Agent地址"
                        type="url"
                        placeholder="https://example.com/weather-agent"
                        value={config.draftUrl}
                        disabled={busy || config.writeUncertain}
                        onChange={(event) => config.editUrl(event.target.value)}
                      />
                      <FieldDescription>
                        请确保此地址下的 /.well-known/agent-card.json 可访问。
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
                    form="a2a-create-form"
                    disabled={disabled || !config.draftUrl.trim()}
                  >
                    {config.saving ? "正在添加…" : "添加"}
                  </Button>
                </DialogFooter>
              </DialogContent>
            </Dialog>
          </FieldLegend>
          <FieldDescription>
            通过 A2A 协议连接外部 Agent。新增、启停和删除会立即保存。
          </FieldDescription>
          <div className="flex flex-wrap items-center justify-between gap-3">
            <p role="status" className="text-sm text-muted-foreground">
              {config.saving
                ? "正在执行远程 Agent 操作…"
                : config.loading
                  ? "正在读取远程 Agent…"
                  : (config.notice ??
                    (config.loaded
                      ? "已读取远程 Agent 列表"
                      : "等待加载远程 Agent"))}
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
              暂无可展示的远程
              Agent。列表仅包含后端成功读取卡片的服务；已添加的服务未出现时，请检查地址和连接后刷新。
            </p>
          )}
          <ItemGroup role="list" aria-label="远程Agent列表">
            {config.servers.map((server) => (
              <Item
                key={server.id}
                data-a2a-id={server.id}
                variant="outline"
                role="listitem"
              >
                <ItemContent className="min-w-0">
                  <ItemTitle className="flex w-full flex-wrap items-center justify-between gap-2 font-bold text-foreground">
                    <div className="flex min-w-0 flex-wrap items-center gap-2">
                      <span className="wrap-break-word">
                        {server.name || "未命名 Agent"}
                      </span>
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
                        aria-label={`删除 ${server.name || "未命名 Agent"}`}
                        onClick={() => {
                          setDeleteTarget(server);
                          setDeleteOpen(true);
                        }}
                      >
                        <Trash />
                      </Button>
                      <Switch
                        aria-label={`启用 ${server.name || "未命名 Agent"}`}
                        checked={server.enabled}
                        disabled={disabled}
                        onCheckedChange={(enabled) =>
                          config.setEnabled(server.id, enabled)
                        }
                      />
                    </div>
                  </ItemTitle>
                  <ItemDescription className="whitespace-pre-wrap wrap-break-word">
                    {server.description || "此 Agent 暂未提供描述。"}
                  </ItemDescription>
                  <ItemDescription className="flex flex-wrap items-center gap-x-2 gap-y-1">
                    <LayoutList size={12} />
                    <Badge variant="secondary">
                      输入: {server.input_modes.join(", ") || "未提供"}
                    </Badge>
                    <Badge variant="secondary">
                      输出: {server.output_modes.join(", ") || "未提供"}
                    </Badge>
                    {server.streaming && (
                      <Badge variant="secondary">流式输出</Badge>
                    )}
                    {server.push_notifications && (
                      <Badge variant="secondary">推送通知</Badge>
                    )}
                  </ItemDescription>
                </ItemContent>
              </Item>
            ))}
          </ItemGroup>
          <DeleteAgent
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
