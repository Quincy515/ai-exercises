use async_trait::async_trait;
use loco_rs::{
    app::{AppContext, Hooks, Initializer},
    bgworker::{BackgroundWorker, Queue},
    boot::{create_app, BootResult, StartMode},
    config::Config,
    controller::AppRoutes,
    db::{self, truncate_table},
    environment::Environment,
    task::Tasks,
    Result,
};
use migration::Migrator;
use std::{path::Path, time::Duration};

#[allow(unused_imports)]
use crate::{
    application::{services::AgentService, shutdown::ShutdownSignal},
    controllers,
    infrastructure::{external::RedisStreamTask, storage::configure_storage},
    initializers::openapi::OpenApiInitializer,
    models::_entities::users,
    openapi::clear_routes,
    tasks,
    workers::downloader::DownloadWorker,
};

pub struct App;
#[async_trait]
impl Hooks for App {
    fn app_name() -> &'static str {
        env!("CARGO_CRATE_NAME")
    }

    fn app_version() -> String {
        format!(
            "{} ({})",
            env!("CARGO_PKG_VERSION"),
            option_env!("BUILD_SHA")
                .or(option_env!("GITHUB_SHA"))
                .unwrap_or("dev")
        )
    }

    async fn boot(
        mode: StartMode,
        environment: &Environment,
        config: Config,
    ) -> Result<BootResult> {
        create_app::<Self, Migrator>(mode, environment, config).await
    }

    async fn initializers(_ctx: &AppContext) -> Result<Vec<Box<dyn Initializer>>> {
        Ok(vec![Box::new(OpenApiInitializer)])
    }

    async fn after_context(ctx: AppContext) -> Result<AppContext> {
        configure_storage(ctx).await
    }

    fn routes(ctx: &AppContext) -> AppRoutes {
        clear_routes();
        // 路由开始服务前创建共享关闭标记；保留已有标记，避免重置关闭状态。
        if !ctx.shared_store.contains::<ShutdownSignal>() {
            ctx.shared_store.insert(ShutdownSignal::default());
        }

        AppRoutes::with_default_routes() // controller routes below
            .add_route(controllers::app_config::routes())
            .add_route(controllers::status::routes())
            .add_route(controllers::files::routes())
            .add_route(controllers::sessions::routes())
            .add_route(controllers::auth::routes())
    }
    async fn connect_workers(ctx: &AppContext, queue: &Queue) -> Result<()> {
        queue.register(DownloadWorker::build(ctx)).await?;
        Ok(())
    }

    async fn on_shutdown(ctx: &AppContext) {
        tracing::info!("MoocManus正在关闭");
        // 4.结束长连接并等待 Agent 清理，避免持续 SSE 阻塞框架的优雅退出。
        let shutdown = ctx.shared_store.get::<ShutdownSignal>().unwrap_or_default();
        shutdown_agents(AgentService::shutdown::<RedisStreamTask>(&shutdown)).await;
        // 5.数据库、Redis 和存储句柄由 Loco 上下文及 Rust 所有权继续管理。
        // 收尾中仍需保存 Done/状态，因此此处保留连接池直到框架结束请求与任务。
        tracing::info!("MoocManus应用关闭钩子执行完成");
    }

    #[allow(unused_variables)]
    fn register_tasks(tasks: &mut Tasks) {
        // tasks-inject (do not remove)
    }
    async fn truncate(ctx: &AppContext) -> Result<()> {
        truncate_table(&ctx.db, users::Entity).await?;
        Ok(())
    }
    async fn seed(ctx: &AppContext, base: &Path) -> Result<()> {
        db::seed::<users::ActiveModel>(&ctx.db, &base.join("users.yaml").display().to_string())
            .await?;
        Ok(())
    }
}

async fn shutdown_agents(shutdown: impl std::future::Future<Output = anyhow::Result<()>>) {
    match tokio::time::timeout(Duration::from_secs(30), shutdown).await {
        Ok(Ok(())) => tracing::info!("Agent服务成功关闭"),
        Ok(Err(error)) => tracing::error!(error = %error, "Agent服务关闭期间出现错误"),
        Err(_) => tracing::warn!("Agent服务清理等待超时，部分资源可能未完成释放"),
    }
}

#[cfg(test)]
#[path = "app_shutdown_tests.rs"]
mod shutdown_tests;
