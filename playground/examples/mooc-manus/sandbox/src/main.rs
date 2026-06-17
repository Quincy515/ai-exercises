use sandbox::{core::Settings, create_app};
use tokio::net::TcpListener;
use tracing_subscriber::prelude::*;

#[tokio::main]
async fn main() -> std::io::Result<()> {
    let settings = Settings::load();

    tracing_subscriber::registry()
        .with(tracing_subscriber::EnvFilter::builder().parse_lossy(&settings.log_level))
        .with(tracing_subscriber::fmt::layer())
        .init();

    tracing::info!("MoocManus沙箱正在初始化");

    let listener = TcpListener::bind("127.0.0.1:3000").await?;
    tracing::info!("MoocManus沙箱监听地址: {}", listener.local_addr()?);
    axum::serve(listener, create_app()).await?;

    tracing::info!("MoocManus沙箱关闭成功");

    Ok(())
}
