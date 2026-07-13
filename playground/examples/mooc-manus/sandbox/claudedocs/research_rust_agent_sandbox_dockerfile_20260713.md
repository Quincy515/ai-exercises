# Rust Agent 沙箱与 Dockerfile 调研

> 调研时间：2026-07-13  
> 范围：用户提供的项目及课程字幕；结论基于各仓库当日默认分支源码快照。

## 结论摘要

1. **Dockerfile 和 Agent 沙箱是两个不同问题。**
   - Dockerfile 负责把 Agent/API 构建成可部署镜像。
   - Agent 沙箱负责把模型触发的 shell、代码或 Skill 放进受限环境执行。
   - 一个项目有 Dockerfile，不代表工具调用已经隔离；反过来，Go 版 Mooc-Manus 没有沙箱镜像 Dockerfile，却通过 Docker SDK 创建运行时容器。

2. 给定项目中，最值得组合参考的是：
   - **部署镜像**：Hermes Agent Rust 的 \`cargo-chef + 非 root\`，ZeroClaw 的多阶段、distroless 和固定基础镜像摘要。
   - **服务容器加固**：OpenHuman Compose 的只读根文件系统、能力清空、禁止提权、tmpfs、CPU/内存限制。
   - **一次性工具沙箱**：OpenHuman、Octos、ZeroClaw 的 \`docker run --rm\`。
   - **会话级容器**：OpenFang 的创建、执行、池化、销毁流程。
   - **跨平台抽象**：ZeroClaw/Octos 的 \`Sandbox\` trait，根据平台选择 Landlock、Bubblewrap、Seatbelt 或 Docker。

3. 本节字幕构建的是**远程开发容器**，核心目标是统一 Linux、编码、依赖和调试环境。SSH、root 密码、源码读写挂载都服务于开发体验，不能直接当作生产 Agent 沙箱。

4. 当前仓库的 \`.devops/Dockerfile\` 是 Python/uv 版本，会复制不存在的 \`pyproject.toml\` 和 \`uv.lock\`；Rust 服务实际固定监听 \`127.0.0.1:3000\`，而容器暴露 8080。因此当前配置不能作为 Rust 项目的可用 Docker 方案。

## 两类容器的边界

\`\`\`mermaid
flowchart LR
    IDE["IDE / 开发者"] --> DEV["开发容器<br/>SSH + Rust 工具链 + 源码挂载"]
    CLIENT["API 调用方"] --> API["Axum Sandbox API<br/>非 root、最小运行镜像"]
    API --> BROKER["SandboxRuntime trait<br/>生命周期与策略"]
    BROKER --> TOOL["临时工具容器<br/>资源、网络、挂载、权限受限"]
\`\`\`

- 开发容器可信，允许编译器、调试工具和项目目录读写。
- API 服务半可信，处理外部请求，但自身不应拥有宿主机 root 能力。
- 工具容器不可信，执行模型生成的命令，应默认拒绝网络、宿主路径和额外权限。

## 代表项目对照

### 1. OpenHuman：最完整的“部署 + 单次执行”组合

OpenHuman 的[服务 Dockerfile](https://github.com/tinyhumansai/openhuman/blob/0d9c1269eb7548a37a68195145641ac9a13d05f6/Dockerfile)使用 Rust builder 与 Debian slim runtime；固定 UID/GID，入口脚本完成卷权限修正后通过 \`gosu\` 降权。其[Compose](https://github.com/tinyhumansai/openhuman/blob/0d9c1269eb7548a37a68195145641ac9a13d05f6/docker-compose.yml)增加：

- \`read_only: true\`
- \`no-new-privileges:true\`
- \`cap_drop: ALL\`
- \`/tmp\` tmpfs
- CPU、内存和健康检查

其[Docker 工具后端](https://github.com/tinyhumansai/openhuman/blob/0d9c1269eb7548a37a68195145641ac9a13d05f6/src/openhuman/sandbox/docker.rs)每次执行新建临时容器，默认：

- \`--network none\`
- \`--cap-drop ALL\`
- 512 MiB、1 CPU
- 只读根文件系统
- \`/tmp\`、\`/var/tmp\` 为带 \`noexec,nosuid\` 的 tmpfs
- \`no-new-privileges\`
- 只显式传递允许的环境变量
- 用 label 清理孤儿容器

不足：

- 没有设置 \`--pids-limit\` 和容器用户。
- \`String::truncate(1MB)\` 可能在中文 UTF-8 字符中间截断并 panic。
- 超时分支宣称已经 kill，但仅对 \`cmd.output()\` 使用 timeout；应以容器名或 ID 显式执行 \`docker rm -f\`。
- 服务镜像默认没有 Docker CLI，也没有挂载 Docker socket，因此“容器化部署服务”和“服务调用 Docker 沙箱”默认不能同时工作。

### 2. OpenFang：适合参考会话级容器生命周期

OpenFang 的[部署 Dockerfile](https://github.com/RightNow-AI/openfang/blob/acf2587e46be174c10200489c9a2d23a39a98aeb/Dockerfile)是多阶段构建，但运行阶段仍基于 Rust 镜像并以 root 运行，适合作为功能镜像，不是最佳生产加固模板。

其[Docker 沙箱实现](https://github.com/RightNow-AI/openfang/blob/acf2587e46be174c10200489c9a2d23a39a98aeb/crates/openfang-runtime/src/docker_sandbox.rs)采用“每个 Agent 创建常驻容器，再通过 \`docker exec\` 执行”的模式：

- 内存、CPU、PID 限制。
- 清空 capabilities，禁止新权限。
- 可配置只读根文件系统、网络和 tmpfs。
- 工作区默认只读挂载。
- 校验镜像名、容器名、挂载源，并阻止 Docker socket 等敏感路径。
- 支持容器池、回收和销毁。
- 输出截断使用 UTF-8 边界安全辅助方法。

适用场景：同一会话连续运行多条命令，需要保留安装依赖、临时文件和进程状态。代价是容器池的过期、并发、异常回收和租户隔离更复杂。

不足：

- \`docker exec\` 超时时同样需要显式终止容器内进程。
- 命令元字符黑名单容易同时阻止合法 shell 能力；真正的边界应主要依赖容器隔离和权限策略。

### 3. Octos：参数构造与路径防注入最值得参考

Octos 的[Docker 沙箱](https://github.com/octos-org/octos/blob/30dee489f8de62361a65ffefcae01160311d2332/crates/octos-agent/src/sandbox/docker.rs)也是每次命令创建临时容器，特点是：

- 默认断网，清空 capabilities，禁止提权。
- 支持 CPU、内存、PID 和工作区 \`rw/ro/none\` 三种挂载模式。
- 清理动态链接器、Python、Node、Shell 等代码注入环境变量。
- 拒绝带冒号、NUL、换行的挂载路径。
- 阻止 \`/etc\`、\`/proc\`、\`/sys\`、\`/dev\` 和 Docker socket。
- 参数构造、挂载模式和危险路径都有单元测试。

其[部署 Dockerfile](https://github.com/octos-org/octos/blob/30dee489f8de62361a65ffefcae01160311d2332/Dockerfile)使用 Rust Alpine builder 和包含 Chromium、LibreOffice、Node 的功能型 runtime，但 runtime 仍以 root 运行。

不足：

- CPU、内存、PID 默认值均为空，容易只“支持限制”而没有默认限制。
- 工具容器未设置只读根文件系统和非 root 用户。
- 自动检测不到后端时会警告后退到无沙箱；生产环境应 fail closed。

### 4. ZeroClaw：架构抽象好，Docker 默认加固不足

ZeroClaw 的[生产 Dockerfile](https://github.com/zeroclaw-labs/zeroclaw/blob/42fa19711e769d9aa142592bf7c52d43628277e2/Dockerfile)值得学习：

- 基础镜像固定 digest。
- 前端、Rust 编译、Debian 调试和 distroless 发布阶段分离。
- 构建缓存和跨架构处理完整。
- 最终使用 distroless nonroot，用户为 65534。

其[开发沙箱镜像](https://github.com/zeroclaw-labs/zeroclaw/blob/42fa19711e769d9aa142592bf7c52d43628277e2/dev/sandbox/Dockerfile)与课程容器接近：Ubuntu、Node、Python、Git、编译工具、非 root 开发用户，但为开发便利授予免密 sudo。

其[Docker Sandbox](https://github.com/zeroclaw-labs/zeroclaw/blob/42fa19711e769d9aa142592bf7c52d43628277e2/crates/zeroclaw-runtime/src/security/docker.rs)通过统一 trait 包装原命令，加入 \`--rm\`、512 MiB、1 CPU、\`--network none\` 和只读工作区。

优点是后端可替换；不足是 Docker 参数没有：

- \`--cap-drop ALL\`
- \`no-new-privileges\`
- \`--read-only\`
- \`--pids-limit\`
- 非 root 用户

默认镜像还是 \`alpine:latest\`。因此更适合学习抽象，不应原样作为生产安全基线。

### 5. Hermes Agent Rust：部署 Dockerfile 好，Docker backend 仍像原型

Hermes Agent Rust 的[Dockerfile](https://github.com/Lumio-Research/hermes-agent-rs/blob/9a145877cacfde43efa07115536b71c51344de75/Dockerfile)使用 \`cargo-chef\` 缓存依赖、Debian slim runtime 和非 root 用户，是当前项目可直接简化借鉴的部署模板。

但其[Docker terminal backend](https://github.com/Lumio-Research/hermes-agent-rs/blob/9a145877cacfde43efa07115536b71c51344de75/crates/hermes-environments/src/docker.rs)创建常驻 Ubuntu 容器时没有网络、资源、能力、用户、挂载和清理约束；输出通过字节索引切 String，也可能破坏 UTF-8。当前管理器创建 backend 后没有显式完成容器初始化，因此不宜作为安全沙箱实现参考。

### 6. small-rust-hermes 与 claw-code：分别代表部署容器和开发容器

- [small-rust-hermes Dockerfile](https://github.com/coder-brzhang/small-rust-hermes/blob/bdd400deb8ba56e30c87b6916348db73f828aded/Dockerfile)：简单清晰的 builder + Debian slim + 非 root + 数据卷，属于部署镜像，不是工具沙箱。
- [claw-code Containerfile](https://github.com/ultraworkers/claw-code/blob/4ea31c1bc91c4e9bcbd67d51c550c01e127e6d0d/Containerfile)：只提供 Rust 编译测试 shell，源码从宿主机挂载到 \`/workspace\`；这与本节课程的远程开发目标最接近，但不提供不可信命令隔离。

## 其余链接盘点

| 项目 | Docker/沙箱现状 | 结论 |
|---|---|---|
| Go 版 Mooc-Manus | 默认分支无 Dockerfile，但有 [Docker SDK Skill Executor](https://github.com/pw151294/mooc-manus/blob/d72feeaf2f2d747d5b7212fbe22bc338b9ba53b0/internal/domains/services/tools/skill_executor_docker.go) | 有一次性/池化容器、资源限制和输入/输出分挂载；缺少断网、cap drop、禁止提权、PID 限制，代码中的 Skill 挂载也未兑现设计文档的只读要求 |
| rust-daerwen-agent | 无 Dockerfile，无 Docker sandbox | 暂无可参考实现 |
| astrcodey | 无 Dockerfile，无 Docker sandbox | 暂无可参考实现 |
| KathaGPT | 无 Dockerfile | 本地 AI 桌面应用，不是本题沙箱参考 |
| AutoAgents | 仅测试构建 Dockerfile；有路径边界工具 | 不是容器执行沙箱 |
| learn-claude-code-rs | 无 Dockerfile；s18 是 Git worktree 隔离 | 文档明确说明 worktree 不是安全沙箱 |
| open-agent-sdk-rust | 无 Dockerfile；有 worktree 工具 | 不是容器执行沙箱 |
| hello-agents | Dockerfile 位于个别共创项目和 MCP 示例 | 不是核心 Agent 沙箱 |
| NousResearch/hermes-agent | Python 主项目有大型 Dockerfile、Compose 和 s6 进程监督 | 对后续 Supervisor/s6 部署有参考价值，但默认 Compose 使用 host network，不是 Rust Docker 沙箱模板 |
| CodeCrafters Claude Code | 课程支持 Rust，覆盖 read/write/bash/agent loop | 页面未提供 Docker 沙箱实现 |
| Learn Claude Code | 渐进课程包含 worktree isolation | 教学重点是 Agent 结构，不是 OS 安全边界 |
| Jake Goldsborough 文章 | Rust 重写关注工具 trait、循环和权限 | 文中没有 Docker 或 sandbox；其原型当时还会自动批准权限 |

## 对当前 Rust + Axum 项目的建议

### 1. 文件职责

\`\`\`text
.devops/
├── Dockerfile.dev          # IDE 远程开发；可含 SSH 和完整 Rust 工具链
├── compose.dev.yaml        # 源码挂载、Cargo 缓存、仅本机端口
├── Dockerfile.sandbox      # 模型命令真正运行的最小工具镜像
Dockerfile                  # Axum API 的生产多阶段镜像
compose.yaml                # API 服务部署，不包含开发 SSH
\`\`\`

不要用一个“万能镜像”同时承担 SSH 开发、API 服务和不可信命令执行。

### 2. Axum API 生产 Dockerfile

\`\`\`dockerfile
# syntax=docker/dockerfile:1.7
FROM rust:1.97-bookworm AS builder
WORKDIR /app

COPY Cargo.toml Cargo.lock ./
COPY src ./src
RUN --mount=type=cache,target=/usr/local/cargo/registry \
    --mount=type=cache,target=/app/target \
    cargo build --locked --release \
    && cp target/release/sandbox /tmp/sandbox

FROM debian:bookworm-slim
RUN apt-get update \
    && apt-get install -y --no-install-recommends ca-certificates \
    && rm -rf /var/lib/apt/lists/* \
    && groupadd --system --gid 10001 sandbox \
    && useradd --system --uid 10001 --gid 10001 --home /app sandbox

COPY --from=builder /tmp/sandbox /usr/local/bin/sandbox
WORKDIR /app
USER 10001:10001
ENV SANDBOX_HOST=0.0.0.0
ENV SANDBOX_PORT=3000
EXPOSE 3000
ENTRYPOINT ["/usr/local/bin/sandbox"]
\`\`\`

Rust 侧必须把监听地址改为配置项；容器内监听 \`127.0.0.1\` 时，Docker 端口映射无法访问服务。

多阶段构建把编译器留在 builder，只复制二进制到 runtime，是 Docker 官方推荐的编译型语言镜像模式：[Multi-stage builds](https://docs.docker.com/build/building/multi-stage/)。

### 3. 工具执行镜像

\`\`\`dockerfile
FROM ubuntu:24.04

ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update \
    && apt-get install -y --no-install-recommends \
       bash ca-certificates curl git python3 ripgrep \
    && rm -rf /var/lib/apt/lists/* \
    && groupadd --gid 10001 runner \
    && useradd --uid 10001 --gid 10001 --create-home runner

USER 10001:10001
WORKDIR /workspace
CMD ["/bin/bash"]
\`\`\`

生产发布时应把基础镜像 tag 固定到经过验证的 digest。只安装 Agent 实际需要的工具；浏览器、Office、Node 等大依赖应按能力拆成不同镜像。

### 4. Rust 运行时必须生成的安全参数

\`\`\`text
docker run --rm
  --name manus-sandbox-<uuid>
  --label manus.sandbox=true
  --network none
  --read-only
  --cap-drop ALL
  --security-opt no-new-privileges
  --pids-limit 64
  --memory 512m
  --cpus 1
  --tmpfs /tmp:rw,noexec,nosuid,size=64m
  --mount type=bind,src=<canonical-workspace>,dst=/workspace
  --workdir /workspace
  manus-sandbox:<pinned-version>
  sh -lc <command>
\`\`\`

设计要点：

- 使用 \`tokio::process::Command\` 逐个传参数，禁止拼成 \`docker run ...\` shell 字符串。
- 工作区先 \`canonicalize\`，拒绝根目录、系统目录、设备目录和 Docker socket。
- 网络默认关闭，按任务能力显式开启；“需要调用 LLM”不代表工具容器本身需要访问网络。
- 配置固定 CPU、内存和 PID 默认值，不要只提供可选字段。
- 默认非 root、只读根文件系统；仅工作区和小型 tmpfs 可写。
- 生成容器名或读取容器 ID；超时时显式执行 \`docker rm -f\`，不能只丢弃 Tokio future。
- 环境变量使用 allowlist，API key 默认不进入工具容器。
- 如果最大长度按字符定义，保持项目现有 \`chars().take(max_length)\` 即可；如果按字节限制子进程输出，先退到 UTF-8 字符边界，不能直接 \`String::truncate(max_bytes)\`。

### 5. 开发容器保留 SSH 时

- 只放在 \`Dockerfile.dev\`。
- 使用普通开发用户，不允许 root SSH。
- 端口绑定 \`127.0.0.1:2222:22\`，避免暴露到局域网。
- Compose 将仓库挂载到 \`/workspace\`，并用 named volume 缓存 Cargo registry 和 \`target\`。
- 应用端口统一为 3000，Rust 服务监听 \`0.0.0.0:3000\`。
- 生产镜像删除 SSH server、密码、编译器和源码挂载。

### 6. Docker API 的位置

不建议把 \`/var/run/docker.sock\` 直接挂进面向公网的 Axum API 容器；Docker daemon API 通常拥有宿主机级高权限。更稳妥的顺序是：

1. 开发期：Rust API 直接在宿主机运行并调用本机 Docker。
2. 单机生产：使用 [Docker Rootless mode](https://docs.docker.com/engine/security/rootless/) 的专用用户和 socket。
3. 多租户生产：把容器生命周期管理拆成内部 runner 服务，API 只提交经过校验的任务，不直接控制宿主 daemon。

## 建议测试

单元测试：

- 默认参数包含断网、只读根、cap drop、禁止提权、CPU/内存/PID。
- 危险工作区和 Docker socket 被拒绝。
- read-only/read-write 挂载模式正确。
- 环境变量只传 allowlist。
- 中文输出在限制边界处不会 panic 或产生无效 UTF-8。
- 指定后端不可用时生产模式 fail closed。

可选 Docker 集成测试：

- 容器能读写工作区，但不能写 \`/etc\`。
- 默认无法访问外网和宿主服务。
- 超时后不存在同 label 的遗留容器。
- PID、内存限制真实生效。
- 并发任务使用不同容器和工作区。

## 调研可信度

- **高**：仓库是否存在 Dockerfile、Compose、Docker runtime，以及源码中明确出现的参数和用户设置。
- **中**：生产适用性评价；本次是源码审查，没有逐个构建并运行所有第三方镜像。
- **时效边界**：结论对应文首提交快照，活跃项目后续可能调整默认值或目录。
