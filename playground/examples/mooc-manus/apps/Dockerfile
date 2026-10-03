# syntax=docker/dockerfile:1.7

# 1.构建阶段同时需要 Node.js 与 Rust，用于生成 Crux WASM 和 TypeScript 类型。
ARG RUST_VERSION=1.97
FROM node:22-bookworm-slim AS node
FROM rust:${RUST_VERSION}-bookworm AS builder
ARG TARGETARCH
WORKDIR /build
COPY --from=node /usr/local/bin/node /usr/local/bin/node
COPY --from=node /usr/local/lib/node_modules /usr/local/lib/node_modules
RUN ln -s /usr/local/lib/node_modules/npm/bin/npm-cli.js /usr/local/bin/npm \
    && ln -s /usr/local/lib/node_modules/npm/bin/npx-cli.js /usr/local/bin/npx

# 2.只安装 Web 所需的 WASM target；容器工具链由基础镜像固定。
RUN rustup target add wasm32-unknown-unknown \
    && cargo install just --version 1.58.0 --locked

# 3.Binaryen 版本与开发文档一致，按 Linux 架构下载并校验官方发布包。
RUN case "$TARGETARCH" in \
        amd64) binaryen_arch=x86_64; binaryen_sha=195ddc94f9bc89f45abdabb0b9eea86023d727ba90eac8b35b80f2544fc30572 ;; \
        arm64) binaryen_arch=aarch64; binaryen_sha=c58562417836c5d0493d89bdefc434933bdc097db641b483df86bcfa557a107f ;; \
        *) echo "Unsupported architecture: $TARGETARCH" >&2; exit 1 ;; \
    esac \
    && curl -fsSL "https://github.com/WebAssembly/binaryen/releases/download/version_132/binaryen-version_132-${binaryen_arch}-linux.tar.gz" -o /tmp/binaryen.tar.gz \
    && echo "$binaryen_sha  /tmp/binaryen.tar.gz" | sha256sum -c - \
    && tar -xzf /tmp/binaryen.tar.gz -C /opt \
    && ln -s /opt/binaryen-version_132/bin/wasm-opt /usr/local/bin/wasm-opt \
    && rm /tmp/binaryen.tar.gz

# 4.读取项目声明的 pnpm 版本；只跳过 Electron 二进制下载，保留工作区依赖图。
COPY package.json ./
RUN npm install --global "$(node -p 'require("./package.json").packageManager')"
ENV CI=true ELECTRON_SKIP_BINARY_DOWNLOAD=1
COPY . .

# 5.从干净源码按既有顺序生成 WASM、类型、安装依赖，再构建 SPA 页面壳。
# BoltFFI CLI 版本继续由 Justfile 从 Cargo.toml 读取，避免 CLI/runtime 不一致。
RUN --mount=type=cache,id=mooc-manus-web-cargo-registry,target=/usr/local/cargo/registry,sharing=locked \
    --mount=type=cache,id=mooc-manus-web-cargo-target-${TARGETARCH},target=/build/target,sharing=locked \
    just build-web \
    && test -f tanstack-app/dist/client/_shell.html

# 6.运行镜像只保留静态产物，由 nginx 同源转发 API、SSE 与 VNC WebSocket。
FROM nginx:1.30-alpine AS runtime
RUN rm -f /usr/share/nginx/html/*.html
COPY deploy/nginx.conf /etc/nginx/conf.d/default.conf
COPY --from=builder /build/tanstack-app/dist/client /usr/share/nginx/html
EXPOSE 80
