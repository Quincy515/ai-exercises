# syntax=docker/dockerfile:1.7

# Rust 版本和开发用户 ID 可以在 Compose 中覆盖，方便其他项目复用。
ARG RUST_VERSION=1.97
FROM rust:${RUST_VERSION}-bookworm

ARG DEV_UID=1000
ARG DEV_GID=1000

# 源码由 Compose 挂载，编译产物和 Cargo 下载缓存存入独立卷。
ENV DEBIAN_FRONTEND=noninteractive \
    HOME=/home/sandbox \
    SSH_PASSWORD=sandbox \
    CARGO_TARGET_DIR=/cargo-target

# 安装 Rust 项目常用工具和 OpenSSH Server。
RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        ca-certificates \
        curl \
        git \
        gosu \
        libssl-dev \
        openssh-server \
        pkg-config \
    && rm -rf /var/lib/apt/lists/* \
    && mkdir -p \
        /run/sshd \
        /workspace \
        /cargo-target \
        /usr/local/cargo/git \
        /usr/local/cargo/registry

# 让 Compose 命令和 SSH 登录 Shell 都能找到 Cargo。
RUN printf '%s\n' \
        'export CARGO_HOME=/usr/local/cargo' \
        'export RUSTUP_HOME=/usr/local/rustup' \
        'export CARGO_TARGET_DIR=/cargo-target' \
        'export PATH="$CARGO_HOME/bin:$PATH"' \
        > /etc/profile.d/rust.sh

# 开发环境使用普通用户；密码由 Compose 在启动时注入。
RUN groupadd --gid "${DEV_GID}" sandbox \
    && useradd \
        --uid "${DEV_UID}" \
        --gid "${DEV_GID}" \
        --create-home \
        --shell /bin/bash \
        sandbox \
    && printf '%s\n' \
        'PasswordAuthentication yes' \
        'PermitRootLogin no' \
        'AllowUsers sandbox' \
        > /etc/ssh/sshd_config.d/sandbox.conf \
    && rm -f /etc/ssh/ssh_host_* \
    && chown -R sandbox:sandbox \
        /workspace \
        /cargo-target \
        /usr/local/cargo

WORKDIR /workspace

EXPOSE 22 3000

HEALTHCHECK --interval=30s --timeout=3s --retries=3 \
    CMD bash -c '</dev/tcp/127.0.0.1/22'

# 单独运行开发镜像时启动 SSH；Compose 会同时启动 SSH 和 Axum API。
CMD ["bash", "-lc", "echo \"sandbox:${SSH_PASSWORD}\" | chpasswd && ssh-keygen -A && exec /usr/sbin/sshd -D -e"]
