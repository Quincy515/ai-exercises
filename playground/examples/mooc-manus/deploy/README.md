# 云端 Web 单机部署

沿用课程的「API + PostgreSQL + Redis + 动态沙箱」结构，用 Compose 固定启动命令。
TanStack 已开启 SPA 模式，构建产物由 nginx 托管；Web 与 API 共用一个访问地址。

```text
浏览器 → HTTPS 入口 → Web nginx
                       ├── 静态页面、Crux WASM
                       └── /api/* → Loco Agent（单实例）
                                      ├── PostgreSQL
                                      ├── Redis
                                      ├── 文件数据卷
                                      └── 按会话创建的沙箱容器
```

## 1.准备 Ubuntu 主机

安装 Docker Engine 和 Compose 插件，将项目上传到服务器。镜像支持 Linux amd64/arm64；
首次构建需要下载 Rust、Node 和沙箱浏览器依赖，并预留足够磁盘、内存。
主机无需另外安装 Rust、pnpm、Node 或数据库。

在 `mooc-manus/deploy` 目录执行：

```sh
cp .env.example .env
# 分别运行两次，将结果填写到 .env 的 POSTGRES_PASSWORD、JWT_SECRET。
openssl rand -hex 32
openssl rand -hex 32
```

设置 `PUBLIC_BASE_URL` 为浏览器访问的完整地址，例如 `https://manus.example.com`。
使用 SSH 隧道验证时保持 `http://localhost:8080`。该值不带 `/api`，用于生成文件和截图 URL；
API 内部继续监听 5150。数据库密码采用上面生成的十六进制值，避免连接 URI 的转义问题。
`.env` 留在服务器，镜像和 Git 都排除它。

## 2.构建与启动

```sh
# 检查配置；缺少密码、JWT 密钥或公开地址会立即报错。
docker compose config --quiet

# 沙箱仅构建镜像，由 Agent 动态启动；不要执行 --profile build up。
docker compose --profile build build sandbox-image

# 构建 API / Web，按 PostgreSQL、Redis → API → Web 的顺序启动。
docker compose up -d --build --wait
docker compose ps
docker compose logs -f api web
```

`apps/Dockerfile` 从源码生成 Crux WASM 和类型，按锁文件安装依赖，再构建 TanStack 页面。
运行镜像只包含 nginx 和静态文件。`server/Dockerfile` 构建 Rust API，运行层包含 Node/npm，
供已配置的 MCP stdio 服务使用。

`sandbox-image` 是镜像构建入口；容器名称和生命周期由现有 `DockerSandbox` 管理。
`SANDBOX_NETWORK` 同时用于 Compose 网络和动态沙箱。数据库、Redis、API、VNC、CDP
均不直接发布宿主端口；沙箱内部端口仍为 3000 / 5901 / 9222。

数据库自动执行已有迁移；PostgreSQL、Redis 和上传文件分别使用命名卷。
API 停止窗口为 40 秒，为已有 30 秒 Agent 收尾预留时间。
Docker socket 的 `:ro` 挂载仍允许 API 创建和删除容器，应仅交给此受信任的 Agent 服务。

## 3.配置访问入口

默认只发布 `127.0.0.1:8080`。个人验证可在本机建立隧道：

```sh
ssh -N -L 8080:127.0.0.1:8080 用户@服务器
```

然后访问 `http://localhost:8080`。此时无需开放服务器的 8080 端口。

域名部署时，将已有 HTTPS 反向代理指向 `127.0.0.1:8080`，证书在入口管理。
若入口使用 nginx，可在已有 HTTPS `server` 块内使用：

```nginx
location / {
    proxy_pass http://127.0.0.1:8080;
    proxy_http_version 1.1;
    proxy_set_header Host $http_host;
    proxy_set_header X-Forwarded-Proto $scheme;
    proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
    proxy_set_header Upgrade $http_upgrade;
    proxy_set_header Connection "upgrade";
    proxy_buffering off;
    proxy_read_timeout 3600s;
    proxy_send_timeout 3600s;
    client_max_body_size 32m;
}
```

内层 `apps/deploy/nginx.conf` 负责 `/api/*` 转发、SSE 即时交付、VNC WebSocket 升级，
以及刷新 `/sessions/...` 时回退到 TanStack 的 `_shell.html`。
外层代理也需关闭缓冲并保留长连接，避免再次缓存事件。

当前课程接口仍需后续接入用户资源权限；先通过 SSH 隧道，或在 HTTPS 入口配置访问限制，
供自己的受控环境使用。`JWT_SECRET` 是框架配置，设置它不会自动为全部业务路由添加鉴权。

## 4.验证与更新

```sh
# 在服务器执行，确认 Web 页面、深链接和同源 API。
curl --fail http://127.0.0.1:8080/ -o /dev/null
curl --fail http://127.0.0.1:8080/sessions/example -o /dev/null
curl --fail -X POST http://127.0.0.1:8080/api/sessions

# 使用上一条响应中的 session_id；先通过已有配置 API 填好模型配置。
session_id='填写返回的 session_id'
curl -N -X POST "http://127.0.0.1:8080/api/sessions/$session_id/chat" \
  -H 'Content-Type: application/json' \
  -d '{"message":"检查当前工作目录并汇报结果","attachments":[]}'

# 文件与会话结果均从同一入口读取。
curl --fail "http://127.0.0.1:8080/api/sessions/$session_id"
curl --fail "http://127.0.0.1:8080/api/sessions/$session_id/files"

# 更新源码后重新构建和启动；沙箱源码变化时另行重建 sandbox-image。
docker compose up -d --build --wait
# 平时停止与重新启动，保留网络及数据卷。
docker compose stop
docker compose start
```

查看日志时同时核对逐事件 SSE、数据库中的事件和最终文件下载地址。截图地址应使用
`PUBLIC_BASE_URL`，不会带上内部的 `:5150`。切换公开域名后，新事件使用新地址。
命名卷保留数据库和文件；`down -v` 会删除这些数据。

本节交付部署入口和运行配置。现有 UI 继续展示当前课程的页面内容，聊天/上传等业务 API
接入仍按后续课程进行，noVNC 演示页也保留本地演示地址；后续应传入
`wss://你的域名/api/sessions/{session_id}/vnc`。Electron sidecar 单独安排。
