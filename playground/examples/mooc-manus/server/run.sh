#!/bin/sh
set -eu

# 启动 Loco 服务；exec 让 Rust API 成为容器主进程并接收 SIGTERM。
exec /app/server-cli start --environment docker
