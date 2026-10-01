use std::sync::Arc;

use anyhow::{bail, Result};
use loco_rs::{app::AppContext, config::CacheConfig};
use tracing::info;

use crate::{
    application::services::{
        AgentService, AppConfigService, FileService, SessionService, StatusService,
    },
    domain::{
        external::HealthChecker,
        repositories::{AppConfigRepository, FileRepository},
    },
    infrastructure::{
        external::{
            BingSearchEngine, DockerSandboxFactory, LocoFileStorage, OpenAILLM,
            PostgresHealthChecker, RedisHealthChecker, RedisStreamTaskFactory, RepairJsonParser,
        },
        repositories::SeaOrmAppConfigRepository,
        settings::AppSettings,
    },
};

use super::repository_dependencies::{get_db_file_repository, get_db_session_repository};

/// 获取 Agent 服务，按请求读取配置快照并组装既有领域能力。
pub async fn get_agent_service(ctx: &AppContext) -> Result<AgentService> {
    let CacheConfig::Redis(redis_config) = &ctx.config.cache else {
        bail!("Agent 任务需要配置 Redis 缓存服务");
    };
    // 构造客户端时不建立连接，创建任务时才获取 Redis Stream 连接。
    let task_factory = Arc::new(RedisStreamTaskFactory::new(redis::Client::open(
        redis_config.uri.as_str(),
    )?));
    let config = SeaOrmAppConfigRepository::new(ctx.db.clone())
        .load()
        .await?
        .unwrap_or_default();
    let settings = AppSettings::from_config(&ctx.config)?;
    let file_repository: Arc<dyn FileRepository> = Arc::new(get_db_file_repository(ctx));
    let file_storage = Arc::new(LocoFileStorage::new(
        Arc::clone(&ctx.storage),
        Arc::clone(&file_repository),
    ));

    Ok(AgentService::new(
        Arc::new(get_db_session_repository(ctx)),
        Arc::new(OpenAILLM::new(config.llm_config)),
        config.agent_config,
        config.mcp_config,
        config.a2a_config,
        Arc::new(DockerSandboxFactory::new(settings.sandbox)),
        task_factory,
        Arc::new(RepairJsonParser),
        Arc::new(BingSearchEngine::new()),
        file_storage,
        file_repository,
    ))
}

/// 获取会话服务，复用应用上下文的数据库连接池。
pub fn get_session_service(ctx: &AppContext) -> SessionService {
    SessionService::new(Arc::new(get_db_session_repository(ctx)))
}

/// 获取文件存储桶，复用 Loco 的存储驱动和数据库连接池。
pub fn get_file_storage(ctx: &AppContext) -> LocoFileStorage {
    let file_repository = Arc::new(get_db_file_repository(ctx));
    LocoFileStorage::new(Arc::clone(&ctx.storage), file_repository)
}

/// 获取文件服务，复用 Loco 上下文的存储驱动和数据库连接池。
pub fn get_file_service(ctx: &AppContext) -> FileService {
    // 1.初始化文件仓库和文件存储桶；两者共享同一个仓库实例。
    let file_repository: Arc<dyn FileRepository> = Arc::new(get_db_file_repository(ctx));
    let file_storage = Arc::new(LocoFileStorage::new(
        Arc::clone(&ctx.storage),
        Arc::clone(&file_repository),
    ));

    // 2.构建服务并返回。
    FileService::new(file_storage, file_repository)
}

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
