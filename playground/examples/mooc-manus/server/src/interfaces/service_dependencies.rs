use std::sync::Arc;

use loco_rs::app::AppContext;
use tracing::info;

use crate::{
    application::services::{AppConfigService, StatusService},
    domain::external::HealthChecker,
    infrastructure::{
        external::{PostgresHealthChecker, RedisHealthChecker},
        repositories::SeaOrmAppConfigRepository,
    },
};

/// 获取应用配置服务。
/// Build the application config service.
pub fn get_app_config_service(ctx: &AppContext) -> AppConfigService<SeaOrmAppConfigRepository> {
    // 1. 获取数据仓库 AppConfigRepository 并打印日志。
    // 1. Build the AppConfigRepository and record a setup log.
    info!("加载获取应用配置服务");
    let app_config_repository = SeaOrmAppConfigRepository::new(ctx.db.clone());

    // 2. 创建 AppConfigService 实例并返回。
    // 2. Create the AppConfigService instance and return it.
    AppConfigService::new(app_config_repository)
}

/// 获取状态服务。
/// Build the status service.
pub fn get_status_service(ctx: &AppContext) -> StatusService {
    // 1. 初始化 Postgres 和 Redis 健康检查器。
    // 1. Initialize Postgres and Redis health checkers.
    let postgres_checker: Arc<dyn HealthChecker> =
        Arc::new(PostgresHealthChecker::new(ctx.db.clone()));
    let redis_checker: Arc<dyn HealthChecker> =
        Arc::new(RedisHealthChecker::new(ctx.cache.clone()));

    // 2. 创建 StatusService 实例并返回。
    // 2. Create the StatusService instance and return it.
    info!("加载获取状态服务");
    StatusService::new(vec![postgres_checker, redis_checker])
}
