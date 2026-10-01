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
当前运行器沿用批量 Flow：模型和工具产生一轮结果后，再逐条推送事件。
步骤开始、工具调用中等事件会延后到本轮返回后可见，工具快照与新输入检查
也受这个时序影响。逐事件传播需要后续贯通 Agent、Flow 和 Runner。

运行中的会话收到非空消息时，已有 Task 会被复用；Task 实例丢失时会重建，
保留会话历史并复用或按需创建沙箱。当前回归覆盖单请求编排，
同会话并发创建与跨资源部分成功的恢复需要单独验证。

```sh
# 隔离测试：各自创建临时 PostgreSQL / Redis，不调用真实模型或 Docker
cargo test --locked --lib \
  --test agent_service --test chat_stream_api --test session_api \
  --test event_response --test file_api --test file_storage
```

测试需要本机 `initdb`、`pg_ctl` 和 `redis-server`。
可通过 `FILE_REPOSITORY_PG_BIN` 指定 PostgreSQL 程序目录，
通过 `CHAT_TEST_REDIS_BIN` 指定 Redis 程序。完成手动演示后使用
`docker stop sandbox-dev` 关闭上述容器。
