# Agent 配置读取与刷新

用户打开通用配置，看到服务器当前的三个 Agent 参数，并可刷新。当前页面只读。

## Sub-features

- `settings-open`：从共享布局打开设置，选择通用配置。
- `settings-read`：三个只读字段与本次真实 GET 响应一致。
- `settings-refresh`：点击刷新产生新的 GET，字段与新响应一致。
- `settings-close`：关闭弹窗回到原页面。

## How to get to it (user POV)

- 首页 `/`、会话 `/sessions/1`、定时任务 `/schedules`、资料库 `/library` 都有全局“打开设置”按钮。
- 打开“MoocManus 设置”后选“通用配置”。noVNC 全屏页跳过全局布局，没有这个入口。

## Driving it with verify-mooc-manus

Preconditions: doctor 的 `eligible.settings` 为 true；后端 `GET /api/app_configs/agent` 返回三项整数。

运行 `node .agents/skills/verify-mooc-manus/scripts/verify.mjs drive --run output/playwright/verify-mooc-manus/manual --features settings`。

- **打开**：四个页面逐一进入，点击 `button[name="打开设置"]`，Dialog 内点 `button[name="通用配置"]`。
- **读取**：等待状态“已读取服务器配置”；比较 `#max_iterations`、`#max_retries`、`#max_search_results` 与浏览器实际响应的同名字段；三个 input 均 readonly，“保存”禁用。
- **刷新**：点击 `button[name="刷新"]`，观察第二次成功 GET，再比对三个值；保存刷新前后截图与 API 结果。
- **关闭**：点击 Dialog 内“取消”，确认弹窗隐藏；记录四个入口的完整覆盖。
- **证据**：`settings/` 中保留操作日志、截图、ARIA、trace 和公开配置值；后端受阻时保存实际错误界面并标记 blocked。

## Gotchas

- 参数会变，断言使用本次响应，避免固定写死 `100/3/10`。
- i64 的浏览器展示是字符串；当前 doctor 对 JSON safe integers 做可信比对，超出范围时需要扩展无损解析后再宣称通过。
- 未启动的后端不能用成功 fixture 冒充；本轮技能不自动迁移开发数据库。
- 错误恢复、LLM/MCP/A2A 保存、移动端和 Electron 安装包是独立验证范围。
