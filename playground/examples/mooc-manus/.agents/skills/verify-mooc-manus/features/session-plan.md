# 会话导航与任务计划

用户从聊天列表进入会话、查看任务详情、展开或收起任务计划，并返回首页。当前会话内容是应用内置演示数据。

## Sub-features

- `session-sidebar`：侧栏点击进入会话，选中态与 URL 一致。
- `session-deep-link`：直接打开并刷新会话 URL，详情保持可见。
- `plan-toggle`：计划在折叠和展开状态之间切换。
- `session-return`：新聊天、首页导航和首页 Logo 能到达首页。

## How to get to it (user POV)

- 首页侧栏点击“打开会话 1：图片合并为PDF的操作计划”。
- 直接访问 `/sessions/2` 并刷新页面。
- 详情底部点击“展开任务计划”/“收起任务计划”。
- 侧栏“新聊天”、一级导航“首页”返回首页；首页 Logo 的标签为“返回首页”。

## Driving it with verify-mooc-manus

Preconditions: doctor 的 `eligible.sessions` 为 true，视口 1280×900，独立 browser context。

运行 `node .agents/skills/verify-mooc-manus/scripts/verify.mjs drive --run output/playwright/verify-mooc-manus/manual --features sessions`。

- **进入**：点击准确会话标签；确认 `/sessions/1`、`aria-current="page"` 和 `region[name="会话任务详情"]`。
- **展开**：点击“展开任务计划”；确认按钮 `aria-expanded=true`、任务进度可见，`list[name="任务步骤"]` 含 3 项。
- **收起**：点击“收起任务计划”；确认任务步骤列表移除，折叠按钮恢复。
- **其他入口**：“新聊天”回到 `region[name="新建会话任务"]`；打开 `/sessions/2` 并刷新，再核对第二会话选中；“首页”和“返回首页”保持路径 `/`。
- **证据**：保存首页、详情、计划展开/收起、深链接刷新后的截图、ARIA 与 action trace。

## Gotchas

- 固定进度显示 `1 / 5`，现有演示列表实际为 3 项；它们不代表后台任务执行。
- 新聊天当前只导航；发送消息、删除会话和消息内部步骤按钮仍是占位。本脚本阻止并报告意外 API 写请求。
- `⌘K` 是显示提示，此处没有键盘处理器；不把它作为已接通入口。
- 窄屏需要先“展开会话列表”，本次固定桌面视口；Electron 路径为 `#/sessions/1`，需单独验证。
