use loco_rs::app::AppContext;
use tracing::info;

use crate::infrastructure::repositories::SeaOrmSessionRepository;

/// 获取基于数据库的会话仓库。
pub fn get_db_session_repository(ctx: &AppContext) -> SeaOrmSessionRepository {
    // 1.从应用上下文复用连接池，不缓存请求事务或重新建立数据库连接。
    info!("加载获取 SeaOrmSessionRepository");

    // 2.克隆的是共享连接池句柄，具体数据库操作由仓库负责。
    SeaOrmSessionRepository::new(ctx.db.clone())
}
