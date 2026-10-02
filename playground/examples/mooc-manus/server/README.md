# Welcome to Loco :train:

[Loco](https://loco.rs) is a web and API framework running on Rust.

This is the **SaaS starter** which includes a `User` model and authentication based on JWT.
It also include configuration sections that help you pick either a frontend or a server-side template set up for your fullstack server.


## Quick Start

```sh
cargo loco start
```

```sh
$ cargo loco start
Finished dev [unoptimized + debuginfo] target(s) in 21.63s
    Running `target/debug/myapp start`

    :
    :
    :

controller/app_routes.rs:203: [Middleware] Adding log trace id

                      ▄     ▀
                                 ▀  ▄
                  ▄       ▀     ▄  ▄ ▄▀
                                    ▄ ▀▄▄
                        ▄     ▀    ▀  ▀▄▀█▄
                                          ▀█▄
▄▄▄▄▄▄▄  ▄▄▄▄▄▄▄▄▄   ▄▄▄▄▄▄▄▄▄▄▄ ▄▄▄▄▄▄▄▄▄ ▀▀█
 ██████  █████   ███ █████   ███ █████   ███ ▀█
 ██████  █████   ███ █████   ▀▀▀ █████   ███ ▄█▄
 ██████  █████   ███ █████       █████   ███ ████▄
 ██████  █████   ███ █████   ▄▄▄ █████   ███ █████
 ██████  █████   ███  ████   ███ █████   ███ ████▀
   ▀▀▀██▄ ▀▀▀▀▀▀▀▀▀▀  ▀▀▀▀▀▀▀▀▀▀  ▀▀▀▀▀▀▀▀▀▀ ██▀
       ▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀
                https://loco.rs

environment: development
   database: automigrate
     logger: debug
compilation: debug
      modes: server

listening on http://localhost:5150
```

## Full Stack Serving

You can check your [configuration](config/development.yaml) to pick either frontend setup or server-side rendered template, and activate the relevant configuration sections.


## Getting help

Check out [a quick tour](https://loco.rs/docs/getting-started/tour/) or [the complete guide](https://loco.rs/docs/getting-started/guide/).

## 17-4：启动沙箱与验证聊天事件流

以下命令在 `server` 目录执行。开发数据库、Redis 和模型配置沿用已有配置；
沙箱的 Rust API 端口为 3000，远程桌面的 WebSocket 端口为 5901。

```sh
# 首次构建镜像
docker build -t sandbox-dev ../sandbox

# 本地测试沙箱；端口仅绑定本机
docker run --rm -d --name sandbox-dev --shm-size=512m \
  -p 127.0.0.1:3000:3000 \
  -p 127.0.0.1:5900:5900 \
  -p 127.0.0.1:5901:5901 \
  -p 127.0.0.1:9222:9222 sandbox-dev

# 直连上述沙箱，服务默认端口为 5150
SANDBOX_ADDRESS=127.0.0.1 cargo loco start
```

同时启动 Web 时，在 `apps` 目录完成 `just install` 后，使用
`pnpm --filter tanstack-app exec vite --port 3001`，为沙箱 API 保留 3000 端口。

先创建会话，再将返回的 `session_id` 填入聊天请求：

```sh
curl -X POST http://localhost:5150/api/sessions

session_id='填写上一步返回的 session_id'
curl -N -X POST "http://localhost:5150/api/sessions/$session_id/chat" \
  -H 'Content-Type: application/json' \
  -d '{"message":"检查当前工作目录并汇报结果","attachments":[]}'
```

`timestamp` 可选，单位是 Unix 秒；省略时采用服务端当前时间。
`attachments` 是文件 ID 数组，省略或传入 `null` 都按空列表处理。
不传 `message` 时订阅已有任务；`event_id` 表示已收到的最后一个队列事件 ID。

响应为 SSE：`event` 是响应事件名称，`data` 是统一的展示载荷，公共字段为
`event_id` 与整数 Unix 秒 `created_at`。步骤另有业务 `id`，工具 `content`
使用截图、搜索条目、控制台记录或文件正文等展示内容。
响应转换集中在 `src/views/events.rs`，数据库和 Redis 保留完整领域事件格式。
收到 `done`、`error` 或 `wait` 后结束本轮订阅；关闭连接后后台任务继续执行。
Agent、Flow 和 Runner 使用等待式 `EventSink` 逐条交付事件。计划生成后立即
发布标题、初始消息和计划；步骤开始、工具 Calling 在后续动作前保存，Called
在工具结束后补全当时的展示快照并保存，然后继续执行。模型文字按完整回复返回。
每次交付后检查新输入：收到新消息就停止旧轮并重新规划；Wait 保存等待状态，
保留排队的用户回复。后续失败保留已发布历史，数据库仍使用连接池与短事务。
Runner 在正常完成、失败和 Wait 返回前清理 MCP/A2A；Tokio 取消回调补上取消出口
的清理，`on_done` 记录结束日志。沙箱保留供继续会话；显式销毁先销毁沙箱，
成功后清理远程工具。MCP/A2A 清理失败分别记录日志并继续后一项。

Planner 与 ReAct 共享工具实例，规划请求仍使用 `tool_choice=none`。
消息、消息时间和未读数通过仓库的单条 SQL 原子更新。截图的 `screenshot` 字段
是完整文件 URL，使用 Loco `server.full_url()` 和既有文件下载接口，客户端可直接加载。
文件上传保留原始文件名和 MIME；上传完成后保存元数据，保存失败保留已上传对象。
当前按课程传入 `filepath` 调用按 `id` 匹配的附件移除方法，因此同一路径生成的
多个版本会保留在会话附件列表中。模型和工具的快照仍按每次实际执行时的内容保存。

运行中的会话收到非空消息时，已有 Task 会被复用；Task 实例丢失时会重建，
保留会话历史并复用或按需创建沙箱。当前回归覆盖单请求编排，
同会话并发创建与跨资源部分成功的恢复需要单独验证。

```sh
# 隔离测试：各自创建临时 PostgreSQL / Redis，不调用真实模型或 Docker
cargo test --locked --lib \
  --test agent_service --test chat_stream_api --test session_api \
  --test event_response --test session_repository \
  --test file_api --test file_storage --test file_service
```

测试需要本机 `initdb`、`pg_ctl` 和 `redis-server`。
领域测试用 Notify 暂停模型和工具，验证阻塞期间队列与仓库已保存前置事件、
两次写入同一文件的快照分别为 v1/v2，以及新输入阻止旧轮下一工具。
`chat_stream_api` 使用随机真实 HTTP 端口增量读取 SSE，验证任务仍阻塞时
客户端已收到 Calling，PostgreSQL 已保存相同 `event_id`。
可通过 `FILE_REPOSITORY_PG_BIN` 指定 PostgreSQL 程序目录，
通过 `CHAT_TEST_REDIS_BIN` 指定 Redis 程序。完成手动演示后使用
`docker stop sandbox-dev` 关闭上述容器。
