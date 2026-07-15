# syntax=docker/dockerfile:1.7

# Rust 构建阶段只负责生成 Axum API 二进制，不进入最终沙箱镜像。
ARG RUST_VERSION=1.97
FROM rust:${RUST_VERSION}-bullseye AS builder

ARG APP_NAME=sandbox

WORKDIR /build
COPY . .

# 使用 BuildKit 缓存 Cargo 依赖和编译产物，最终只输出一个可执行文件。
RUN --mount=type=cache,target=/usr/local/cargo/registry \
    --mount=type=cache,target=/build/target \
    cargo build --release --locked --bin "${APP_NAME}" \
    && mkdir -p /out \
    && cp "/build/target/release/${APP_NAME}" /out/sandbox

# 1.使用 Ubuntu 基础镜像作为完整沙箱运行环境
FROM ubuntu:22.04 AS runtime

# 2.安装过程避免交互式提示并设置主机名
ENV DEBIAN_FRONTEND=noninteractive \
    HOSTNAME=sandbox

# 3.将 Ubuntu 默认的 apt 软件源地址统一换成阿里云镜像源
RUN sed -i 's|http://archive.ubuntu.com/ubuntu/|http://mirrors.aliyun.com/ubuntu/|g' /etc/apt/sources.list \
    && sed -i 's|http://security.ubuntu.com/ubuntu/|http://mirrors.aliyun.com/ubuntu/|g' /etc/apt/sources.list \
    && sed -i 's|http://ports.ubuntu.com/ubuntu-ports/|http://mirrors.aliyun.com/ubuntu-ports/|g' /etc/apt/sources.list

# 4.更新并安装沙箱基础软件，完成后删除 apt 缓存
RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        bc \
        ca-certificates \
        curl \
        git \
        gnupg \
        socat \
        software-properties-common \
        sudo \
        supervisor \
        websockify \
        wget \
        x11vnc \
        xterm \
        xvfb \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# 5.创建 ubuntu 用户并赋予免密 sudo 权限
RUN useradd -m -d /home/ubuntu -s /bin/bash ubuntu \
    && printf '%s\n' 'ubuntu ALL=(ALL) NOPASSWD:ALL' > /etc/sudoers.d/ubuntu \
    && chmod 0440 /etc/sudoers.d/ubuntu \
    && mkdir -p /sandbox /tmp/chromium-profile /var/log/supervisor \
    && chown -R ubuntu:ubuntu /sandbox /tmp/chromium-profile /var/log/supervisor

# 6.Ubuntu 22.04 默认提供 Python 3.10，无需额外 PPA
RUN add-apt-repository ppa:deadsnakes/ppa && \
    apt-get update \
    && apt-get install -y --no-install-recommends \
        python3.12 \
        python3.12-dev \
        python3.12-venv \
        python3-pip \
    && update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.12 1 \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# 7.为pip3配置阿里云镜像源
RUN pip3 config set global.index-url https://mirrors.aliyun.com/pypi/simple/

# 8.安装node.js 24.12.0版本(LTS)
RUN mkdir -p /etc/apt/keyrings && \
    curl -fsSL https://deb.nodesource.com/gpgkey/nodesource-repo.gpg.key | gpg --dearmor -o /etc/apt/keyrings/nodesource.gpg && \
    echo "deb [signed-by=/etc/apt/keyrings/nodesource.gpg] https://deb.nodesource.com/node_24.x nodistro main" | tee /etc/apt/sources.list.d/nodesource.list && \
    apt-get update && \
    apt-get install -y nodejs && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

# 9.为所有用户配置 npm 阿里云镜像源
RUN npm config set registry https://registry.npmmirror.com --location=global

# 10.安装 Chromium 浏览器
RUN add-apt-repository ppa:xtradeb/apps -y \
    && apt-get update \
    && apt-get install -y --no-install-recommends chromium \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# 11.安装中文字体和语言支持
RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        fonts-noto-cjk \
        fonts-noto-color-emoji \
        language-pack-zh-hans \
        locales \
    && locale-gen zh_CN.UTF-8 \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# 12.设置默认编码和 Rust API 运行参数
ENV HOME=/home/ubuntu \
    LANG=zh_CN.UTF-8 \
    LANGUAGE=zh_CN:zh \
    LC_ALL=zh_CN.UTF-8 \
    DISPLAY=:1 \
    SERVER_HOST=0.0.0.0 \
    SERVER_PORT=3000 \
    LOG_LEVEL=INFO \
    SERVER_TIMEOUT_MINUTES=60 \
    CHROME_ARGS=""

# 13.设置沙箱工作空间
WORKDIR /sandbox

# 14.只从构建阶段复制 Rust Release 二进制
COPY --from=builder /out/sandbox /usr/local/bin/sandbox

# 15.复制 Supervisor 配置，不复制 Rust 源码和 Cargo 编译缓存
COPY supervisord.conf /sandbox/supervisord.conf

# 16.暴露端口：3000(Axum API)、9222(CDP)、5900(VNC)、5901(WebSocket)
EXPOSE 3000 9222 5900 5901

# 17.使用 Axum OpenAPI 文档检查容器健康状态
HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
    CMD curl --fail --silent --show-error http://127.0.0.1:3000/api-docs/openapi.json > /dev/null || exit 1

STOPSIGNAL SIGTERM

# 18.使用 Supervisor 管理 Axum、Chromium 和远程桌面进程
CMD ["supervisord", "-n", "-c", "/sandbox/supervisord.conf"]
