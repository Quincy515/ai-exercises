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

## 17-8：读取会话详情与持续刷新会话列表

```sh
# 会话详情：session_id 使用创建会话接口返回的值
curl "http://localhost:5150/api/sessions/$session_id"

# 持续订阅侧栏需要的会话基础信息，Ctrl+C 结束本次订阅
curl -N -X POST http://localhost:5150/api/sessions/stream
```

详情返回 `session_id/title/status/events`，历史事件沿用聊天流的 `event + data` 格式。
列表流立即发送首帧，随后每五秒重新查询并发送 `event: sessions`，其 `data`
直接为 `{"sessions": [...]}`，空列表同样发送。读取详情和订阅列表均保留未读数，
需要清零时调用已有的 `clear-unread-message-count` 接口。

```sh
cargo test --locked --test session_api --test agent_service --test event_response --test chat_stream_api
```

接口测试使用私有数据库与随机 HTTP 端口，验证历史映射、查询错误、首帧、五秒间隔
及数据库更新后的快照。流等待期间可以继续读写会话，断开请求后释放本次查询流。

## 17-9：停止任务会话与读取会话文件

```sh
# session_id 使用已有会话 ID；停止请求无需 JSON 请求体
curl -X POST "http://localhost:5150/api/sessions/$session_id/stop"

# 读取人类上传和智能体生成的文件元数据
curl "http://localhost:5150/api/sessions/$session_id/files"
```

停止接口查找并取消已有任务，然后将会话标记为 `completed`，成功响应的 `data` 为
`null`。已有事件、文件和沙箱继续保留；无任务或重复停止时同样返回成功。运行中
任务通过现有 Tokio 取消监督流程保存 `done`、结束聊天 SSE 并清理远程工具。

文件列表响应为 `data: {"files": [...]}`，保留会话中的顺序和同路径版本，空列表为
`[]`。本节两个新增服务的会话缺失错误按课程运行错误出口返回 500，UUID 格式错误
沿用 400。接口测试覆盖真实 HTTP 停止、私有 Redis/PostgreSQL、Done 持久化及文件字段。

## 17-10：查看沙箱文件与 Shell 内容

```sh
# URL 使用任务会话 UUID；文件路径来自工具事件或会话文件列表
curl -X POST "http://localhost:5150/api/sessions/$session_id/file" \
  -H 'Content-Type: application/json' \
  -d '{"filepath":"/home/ubuntu/bubble_sort.py"}'

# 请求体中的 session_id 是 Shell 标识，如 manus-shell
curl -X POST "http://localhost:5150/api/sessions/$session_id/shell" \
  -H 'Content-Type: application/json' \
  -d '{"session_id":"manus-shell"}'
```

文件响应的 `data` 为 `{filepath, content}`；沿用沙箱默认的 10000 字符读取上限。
Shell 响应为 `{session_id, output, console_records}`，记录包含 `ps1/command/output`，
缺省记录为 `[]`。两个接口只读取已关联沙箱，保持会话状态、历史和文件记录。
会话缺失返回 500，无沙箱或沙箱已销毁返回 404，沙箱返回读取失败时以 500 传递提示。
UUID 格式错误返回 400，请求字段缺失或类型错误返回 422。

```sh
cargo test --locked --lib --test session_service --test session_api --test agent_service --test chat_stream_api --test event_response
```

## 17-11：VNC WebSocket 双向转发

服务端提供 `ws://localhost:5150/api/sessions/{session_id}/vnc`，由会话 ID 查找已关联
沙箱的 VNC 地址。客户端和服务端先完成握手，随后连接沙箱并双向转发二进制数据。
客户端关闭或沙箱关闭都会结束另一方向的转发并释放连接；VNC 断开只结束本次远程
桌面连接，会话、沙箱及后台 Agent 任务继续按原有生命周期运行。

子协议优先选择 `binary`，其次为 `base64`；未匹配时接受无子协议连接。课程中的
`base64` 只参与协商，载荷保持二进制，本节沿用这一行为。会话/沙箱查找或连接失败
在升级后使用关闭码 `1011`，完整错误记录在服务端日志中。

已有共享 `VNCViewer` 接收 `url` 属性，后续前端业务接入时可传入上述会话代理地址；
当前演示入口仍使用本地直连地址。测试使用随机本地端口与受控 WebSocket 上游，
验证协议协商、双向字节保持、两端关闭和错误关闭；真实桌面显示需结合运行中的沙箱验证。

```sh
cargo test --locked --lib controllers::sessions::vnc_tests
cargo test --locked --test session_service
```

## 17-12：应用关闭与 Agent 优雅退出

Loco 的 `App::on_shutdown` 先通知列表 SSE、聊天 SSE 和 VNC 结束，再等待
`AgentService::shutdown::<RedisStreamTask>(&shutdown)` 清理仍在注册表中的任务，最多等待
30 秒。任务执行取消后，监督器完成取消/结束回调，再销毁 Runner 的沙箱与工具。
关闭标记与任务准备读写锁配合，确保销毁快照包含已经进入准备过程的任务。
超时或清理错误会记录日志，部分资源可能未完成释放。连接池在收尾期间保持可用，
数据库、Redis 与存储句柄随后由 Loco 上下文和 Rust 所有权管理。

聊天流正常结束、异常或客户端断开时，独立 Tokio 任务执行一次未读数清零。
后台 Agent 继续运行；清零失败仅记录警告，清零完成后的新消息继续计入未读。
错误事件保存失败时仍返回原始错误事件。应用退出可使用运行服务终端的 Ctrl+C。

已有 `Memory::compact` 会移除消息的 `reasoning_content`，保留工具调用轮次中需要的
推理字段直到压缩发生；Runner 和 MCP 清理沿用既有幂等实现。SeaORM 默认启用
连接取用前探活，数据库继续采用连接池与短操作。

```sh
cargo test --locked --lib --test agent_service --test chat_stream_api --test session_api --test session_service --test event_response
```

## 17-13：在 Docker / Ubuntu 中运行 API

在 `server` 目录构建。镜像分为 Rust 编译阶段和 Ubuntu 运行阶段，最终携带
`server-cli`、`config/docker.yaml`、`run.sh` 和 Node/npm。Node 供 `npx` 类型的 MCP
工具使用；浏览器能力通过 Rust CDP 客户端连接沙箱。运行入口使用 `exec` 传递停止信号。

```sh
# 1.构建 API；sandbox-dev 使用 sandbox 目录中的 Dockerfile 预先构建
docker build -t manus-api-dev .

# 2.创建开发网络
docker network create manus-network-dev

# 3.启动 Redis 与 PostgreSQL，数据存放在独立 Docker volume
docker run -d --name manus-redis-dev --network manus-network-dev \
  -v manus_redis_data_dev:/data redis:8.2
docker run -d --name manus-db-dev --network manus-network-dev \
  -e POSTGRES_USER=postgres -e POSTGRES_PASSWORD=postgres -e POSTGRES_DB=manus \
  -v manus_postgres_data_dev:/var/lib/postgresql/data postgres:17.6

# 4.确认依赖已就绪，再启动 API
docker exec manus-db-dev pg_isready -U postgres
docker exec manus-redis-dev redis-cli ping
docker run -d --name manus-api-dev --network manus-network-dev \
  -p 5150:5150 --stop-timeout 40 \
  -v manus_api_files_dev:/app/storage \
  -v /var/run/docker.sock:/var/run/docker.sock:ro manus-api-dev

# 5.查看启动、自动迁移与请求日志
docker logs -f manus-api-dev
# 另开终端检查 API 文档
curl --fail http://localhost:5150/api-docs/openapi.json

# 6.应用关闭时预留 Agent 的30秒收尾时间
docker stop --timeout 40 manus-api-dev
```

API 使用 Docker 网络里的 `manus-db-dev` / `manus-redis-dev`，动态沙箱也加入
`manus-network-dev`。数据库和 Redis 默认只供这个网络访问；需要宿主机工具连接时，
可分别增加未占用的端口映射。现有同名容器重用 `docker start`，网络只需创建一次。
挂载 Docker socket 让 API 可以管理沙箱容器；`:ro` 限制挂载文件写入，Docker API
仍然具备容器管理能力，以上配置用于本地开发。

`config/docker.yaml` 启用 `auto_migrate`，保留已有数据库数据。环境变量可覆盖
`DATABASE_URL`、`REDIS_URL`、`SANDBOX_IMAGE`、`SANDBOX_NETWORK`、`SANDBOX_ADDRESS`。
`SERVER_HOST` 和 `SERVER_PORT` 组成截图下载 URL，应填写客户端可访问的地址；
更改端口时同步容器端口映射。开发配置中的 JWT 默认值可通过 `JWT_SECRET` 覆盖。
`.dockerignore` 排除 `.env`、本地配置、存储与编译产物，镜像中只复制容器配置。

沙箱查找每次重新 inspect，检查运行状态和 IP；获取失败会记录日志并返回 `None`，
由已有任务流程创建新沙箱。截图继续返回既有文件下载 URL，size 按真实字节数计算。
修改源码后重新 `docker build` 并重建 API 容器，命名数据卷继续保留文件和数据库内容。
