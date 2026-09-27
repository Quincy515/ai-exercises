import { MessageSquareText, MoreHorizontal, Plus, Trash } from "lucide-react";
import { cn } from "../lib/utils";
import { Avatar, AvatarFallback } from "./ui/avatar";
import { Button } from "./ui/button";
import {
  DropdownMenu,
  DropdownMenuContent,
  DropdownMenuGroup,
  DropdownMenuItem,
  DropdownMenuTrigger,
} from "./ui/dropdown-menu";
import {
  Item,
  ItemActions,
  ItemContent,
  ItemDescription,
  ItemGroup,
  ItemMedia,
  ItemTitle,
} from "./ui/item";
import { Kbd, KbdGroup } from "./ui/kbd";
import {
  Sidebar,
  SidebarContent,
  SidebarHeader,
  SidebarTrigger,
  useSidebar,
} from "./ui/sidebar";

export type LeftPanelProps = {
  sessionId?: string;
  onNewSession: () => void;
  onSelectSession: (id: string) => void;
};

export function LeftPanel({
  sessionId,
  onNewSession,
  onSelectSession,
}: LeftPanelProps) {
  // 本课使用静态会话，后续再接入业务数据。
  const sessions = ["1", "2", "3"];
  const { setOpenMobile } = useSidebar();

  return (
    <Sidebar>
      {/* 顶部的切换按钮 */}
      <SidebarHeader>
        <SidebarTrigger className="cursor-pointer" aria-label="收起会话列表" />
      </SidebarHeader>
      {/* 中间内容 */}
      <SidebarContent className="p-2">
        {/*新建会话按钮*/}
        <Button
          variant="outline"
          className="mb-3 cursor-pointer"
          aria-label="新建任务"
          onClick={() => {
            onNewSession();
            setOpenMobile(false);
          }}
        >
          <Plus data-icon="inline-start" />
          新建任务
          <KbdGroup>
            <Kbd>⌘</Kbd>
            <Kbd>K</Kbd>
          </KbdGroup>
        </Button>
        {/*会话列表*/}
        <ItemGroup className="gap-1" aria-label="会话列表">
          {sessions.map((session) => (
            <Item
              key={session}
              role="listitem"
              className={cn(
                "flex-nowrap gap-2 p-2 hover:bg-background",
                session === sessionId && "bg-background",
              )}
            >
              <button
                type="button"
                className="flex min-w-0 flex-1 cursor-pointer items-center gap-2 rounded-sm text-left outline-none focus-visible:ring-2 focus-visible:ring-ring"
                aria-label={`打开会话 ${session}：图片合并为PDF的操作计划`}
                aria-current={session === sessionId ? "page" : undefined}
                onClick={() => {
                  onSelectSession(session);
                  setOpenMobile(false);
                }}
              >
                {/* 左侧图标 */}
                <ItemMedia>
                  <Avatar>
                    <AvatarFallback>
                      <MessageSquareText className="size-4" />
                    </AvatarFallback>
                  </Avatar>
                </ItemMedia>
                {/* 中间内容 */}
                <ItemContent className="min-w-0 gap-0">
                  <ItemTitle className="block max-w-full truncate">
                    图片合并为PDF的操作计划
                  </ItemTitle>
                  <ItemDescription className="truncate text-xs">
                    您上传的四张图片已成功合并成为一个PDF文件
                  </ItemDescription>
                </ItemContent>
              </button>
              {/* 右侧操作区 */}
              <ItemActions className="flex-col gap-0 pt-1">
                <ItemDescription className="text-xs">周一</ItemDescription>
                <DropdownMenu>
                  <DropdownMenuTrigger
                    render={
                      <Button
                        size="icon-xs"
                        variant="ghost"
                        className="cursor-pointer"
                      />
                    }
                    aria-label={`会话 ${session} 的操作`}
                  >
                    <MoreHorizontal />
                  </DropdownMenuTrigger>
                  <DropdownMenuContent align="center" side="bottom">
                    <DropdownMenuGroup>
                      {/* 本课保留删除入口，后续接入删除操作。 */}
                      <DropdownMenuItem
                        variant="destructive"
                        className="cursor-pointer"
                      >
                        <Trash />
                        删除
                      </DropdownMenuItem>
                    </DropdownMenuGroup>
                  </DropdownMenuContent>
                </DropdownMenu>
              </ItemActions>
            </Item>
          ))}
        </ItemGroup>
      </SidebarContent>
    </Sidebar>
  );
}
