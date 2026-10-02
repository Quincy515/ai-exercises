# syntax=docker/dockerfile:1.7

# 1.确认基础镜像源：Rust 版本与现有沙箱构建保持一致，满足当前 Cargo.lock 的要求。
ARG RUST_VERSION=1.97
FROM rust:${RUST_VERSION}-bullseye AS builder
ARG RUST_VERSION
ARG TARGETARCH

# 2.配置构建工作空间，按工具链及架构隔离 Linux 编译缓存。
WORKDIR /build
ENV CARGO_TARGET_DIR=/build/target/${RUST_VERSION}-bullseye-${TARGETARCH}
COPY Cargo.toml Cargo.lock ./
COPY migration ./migration
COPY src ./src

# 3.使用 Cargo.lock 安装依赖并生成 API 可执行文件；缓存保留在构建阶段。
RUN --mount=type=cache,id=mooc-manus-server-cargo-registry,target=/usr/local/cargo/registry,sharing=locked \
    --mount=type=cache,id=mooc-manus-server-cargo-target,target=/build/target,sharing=locked \
    cargo build --release --locked --bin server-cli \
    && mkdir -p /out \
    && cp "${CARGO_TARGET_DIR}/release/server-cli" /out/server-cli

# 4.提供 Node.js 22，用于执行配置为 npx 的 MCP stdio 服务。
FROM node:22-bookworm-slim AS node

# 5.使用 Ubuntu 运行 Rust API；浏览器通过 CDP 连接到沙箱中的 Chromium。
FROM ubuntu:22.04 AS runtime
WORKDIR /app
RUN apt-get update \
    && apt-get install -y --no-install-recommends ca-certificates curl libgcc-s1 libstdc++6 \
    && rm -rf /var/lib/apt/lists/*
COPY --from=node /usr/local/bin/node /usr/local/bin/node
COPY --from=node /usr/local/lib/node_modules /usr/local/lib/node_modules
RUN ln -s /usr/local/lib/node_modules/npm/bin/npm-cli.js /usr/local/bin/npm \
    && ln -s /usr/local/lib/node_modules/npm/bin/npx-cli.js /usr/local/bin/npx

# 6.拷贝运行文件；配置由容器环境变量注入，邮件模板和迁移已编译进二进制。
COPY --from=builder /out/server-cli /app/server-cli
COPY config/docker.yaml /app/config/docker.yaml
COPY --chmod=755 run.sh /app/run.sh
RUN mkdir -p /app/storage

# 7.暴露 Loco API 端口，Docker 停止信号交给 Rust 主进程处理。
EXPOSE 5150
STOPSIGNAL SIGTERM

# 8.启动命令脚本。
CMD ["./run.sh"]
