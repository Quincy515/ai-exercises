use loco_rs::app::AppContext;
use tracing::info;

use crate::{
    application::services::AppConfigService,
    infrastructure::repositories::SeaOrmAppConfigRepository,
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
