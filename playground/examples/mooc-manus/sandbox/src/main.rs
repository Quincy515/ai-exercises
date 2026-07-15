use std::io;

use sandbox::{
    core::Settings,
    create_app,
    services::{SUPERVISOR_RESTART_HELPER_ARG, SupervisorService},
};
use tokio::net::TcpListener;
use tracing_subscriber::prelude::*;

#[tokio::main]
async fn main() -> std::io::Result<()> {
    let settings = Settings::load();

    tracing_subscriber::registry()
        .with(tracing_subscriber::EnvFilter::builder().parse_lossy(&settings.log_level))
        .with(tracing_subscriber::fmt::layer())
        .init();

    let mut args = std::env::args();
    let _program = args.next();
    if args.next().as_deref() == Some(SUPERVISOR_RESTART_HELPER_ARG) {
        let rpc_url = args.next().ok_or_else(|| {
            io::Error::new(
                io::ErrorKind::InvalidInput,
                "重启辅助进程缺少 Supervisor Unix Socket 路径",
            )
        })?;
        return SupervisorService::run_restart_helper(rpc_url)
            .await
            .map_err(|err| io::Error::other(err.to_string()));
    }

    tracing::info!("MoocManus沙箱正在初始化");

    let listener = TcpListener::bind((settings.server_host.as_str(), settings.server_port)).await?;
    tracing::info!("MoocManus沙箱监听地址: {}", listener.local_addr()?);
    axum::serve(listener, create_app())
        .with_graceful_shutdown(shutdown_signal())
        .await?;

    tracing::info!("MoocManus沙箱关闭成功");

    Ok(())
}

/// 接管终止信号，让正在处理的 HTTP 请求完成响应后再退出 Axum。
async fn shutdown_signal() {
    #[cfg(unix)]
    {
        let terminate = async {
            match tokio::signal::unix::signal(tokio::signal::unix::SignalKind::terminate()) {
                Ok(mut signal) => {
                    signal.recv().await;
                }
                Err(err) => {
                    tracing::error!(error = %err, "监听 SIGTERM 失败");
                    std::future::pending::<()>().await;
                }
            }
        };

        tokio::select! {
            result = tokio::signal::ctrl_c() => {
                if let Err(err) = result {
                    tracing::error!(error = %err, "监听 Ctrl+C 失败");
                }
            }
            _ = terminate => {}
        }
    }

    #[cfg(not(unix))]
    if let Err(err) = tokio::signal::ctrl_c().await {
        tracing::error!(error = %err, "监听 Ctrl+C 失败");
    }

    tracing::info!("收到退出信号，等待现有 HTTP 请求完成");
}
