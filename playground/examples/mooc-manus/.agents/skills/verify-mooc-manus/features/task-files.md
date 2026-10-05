# 查看任务文件列表

用户在会话页打开本任务文件列表，看到文件名与类型/大小，再关闭弹窗继续查看会话。当前列表由六项演示数据构成。

## Sub-features

- `files-open`：Header 文件按钮打开 Dialog。
- `files-list`：六个演示文件出现在任务文件列表中。
- `files-close`：关闭 Dialog 后会话详情继续可见。

## How to get to it (user POV)

- 从首页侧栏进入会话，点击标题栏“查看会话文件”。
- 消息附件区的“查看此任务中所有的文件”按钮当前未接事件；文件预览与下载也未接通。登记为 unimplemented，Header 的通过不代替这些入口。

## Driving it with verify-mooc-manus

Preconditions: doctor 的 `eligible.files` 为 true；桌面视口；后端无需启动。

运行 `node .agents/skills/verify-mooc-manus/scripts/verify.mjs drive --run output/playwright/verify-mooc-manus/manual --features files`。

- **进入**：点击会话 1，等待详情 region；保存打开前截图。
- **打开**：点击 `button[name="查看会话文件"]`，定位 `dialog[name="此任务中的所有文件"]`。
- **列表**：仅在 Dialog 内验证 `list[name="任务文件"]` 的 6 个 listitem；文件名为 `go+java.pdf`、`全家福.png`、`2025年年中汇报.docx`、`数据分析可视化看板.xsx`、`数据看板动态演示.gif`、`ReActAgent.py`。
- **关闭**：点击 Dialog 内 `button[name="Close"]`，确认弹窗隐藏且会话详情仍可见；保存关闭后截图。
- **证据**：`files/` 保留前后状态、完整列表、用户操作与 trace；没有下载文件或创建服务端记录。

## Gotchas

- 页面其它附件会重复这些文件名，断言必须限定 Dialog。
- `.xsx` 是当前演示数据的真实拼写，验证脚本不修正产品文本。
- 本次不点击下载/预览占位按钮，也不声称上传、下载或文件持久化完成。
- 后续接通下载时，需新增下载事件、文件字节/哈希和临时文件清理证据。
