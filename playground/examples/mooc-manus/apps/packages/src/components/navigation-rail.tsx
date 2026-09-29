import { LibraryBig, UserRound } from "lucide-react";
import { motion, MotionConfig } from "motion/react";
import { cn } from "cn";
import { ManusSettings } from "./manus-settings";
import { Avatar, AvatarFallback } from "./ui/avatar";
import { Button } from "./ui/button";
import { ClockIcon } from "./ui/clock";
import { HomeIcon } from "./ui/home";
import { useSidebar } from "./ui/sidebar";

// 官方动画库暂未提供书架，沿用 Lucide 图形和 Motion 的悬停动效。
function LibraryIcon({ className }: { className?: string }) {
  return (
    <motion.div
      aria-hidden
      className={className}
      whileHover={{ rotate: [0, -8, 8, 0] }}
      transition={{ duration: 0.4 }}
    >
      <LibraryBig />
    </motion.div>
  );
}

const items = [
  {
    to: "/",
    title: "首页",
    icon: HomeIcon,
    activeClassName:
      "[&_path:first-child]:fill-current [&_path:last-child]:fill-secondary [&_path:last-child]:stroke-secondary",
  },
  {
    to: "/schedules",
    title: "定时任务",
    icon: ClockIcon,
    activeClassName:
      "[&_circle]:fill-current [&_line]:stroke-secondary [&_path]:stroke-secondary",
  },
  {
    to: "/library",
    title: "资料库",
    icon: LibraryIcon,
    activeClassName:
      "[&_svg]:fill-current [&_path:first-of-type]:stroke-secondary",
  },
] as const;

export type NavigationTarget = (typeof items)[number]["to"];

export function NavigationRail({
  pathname,
  onNavigate,
}: {
  pathname: string;
  onNavigate: (to: NavigationTarget) => void;
}) {
  const { setOpenMobile } = useSidebar();

  return (
    // 一级栏独立于可拖动的二级列表，收起列表时保持可见。
    <aside
      data-slot="navigation-rail"
      className="sticky top-0 z-20 flex h-svh w-16 shrink-0 flex-col items-center border-r bg-sidebar py-4"
    >
      {/* 跟随系统减少动态效果的设置，关闭旋转等位移动画。 */}
      <MotionConfig reducedMotion="user">
        <nav aria-label="功能导航" className="flex flex-col gap-3">
          {items.map((item) => {
            const active =
              item.to === "/"
                ? pathname === "/" || pathname.startsWith("/sessions/")
                : pathname === item.to || pathname.startsWith(`${item.to}/`);
            return (
              <Button
                key={item.to}
                variant={active ? "secondary" : "ghost"}
                size="icon-lg"
                className={cn(
                  "size-11 cursor-pointer rounded-2xl text-muted-foreground [&_svg:not([class*='size-'])]:size-6",
                  active && "text-foreground hover:bg-secondary",
                  active && item.activeClassName,
                )}
                title={item.title}
                aria-label={item.title}
                aria-current={active ? "page" : undefined}
                onClick={() => {
                  setOpenMobile(false);
                  onNavigate(item.to);
                }}
              >
                <item.icon
                  aria-hidden
                  className="flex size-full items-center justify-center"
                />
              </Button>
            );
          })}
        </nav>
      </MotionConfig>
      <div className="mt-auto flex flex-col items-center gap-3">
        {/* 底部设置模态窗 */}
        <ManusSettings />
        {/* 账号位置暂作展示，后续接入账号业务。 */}
        <Avatar role="img" aria-label="账号" title="账号">
          <AvatarFallback>
            <UserRound className="size-4" />
          </AvatarFallback>
        </Avatar>
      </div>
    </aside>
  );
}
