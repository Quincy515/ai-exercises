use loco_rs::schema::*;
use sea_orm_migration::prelude::*;

#[derive(DeriveMigrationName)]
pub struct Migration;

#[async_trait::async_trait]
impl MigrationTrait for Migration {
    async fn up(&self, m: &SchemaManager) -> Result<(), DbErr> {
        create_table(
            m,
            "agent_configs",
            &[
                ("id", ColType::PkAuto),
                ("max_iterations", ColType::SmallUnsignedNull),
                ("max_retries", ColType::SmallUnsignedNull),
                ("max_search_results", ColType::SmallUnsignedNull),
                ("uuid", ColType::Uuid),
                ("user_id", ColType::UuidNull),
                ("status", ColType::StringNull),
                ("is_deleted", ColType::BooleanNull),
                ("remark", ColType::TextNull),
            ],
            &[],
        )
        .await
    }

    async fn down(&self, m: &SchemaManager) -> Result<(), DbErr> {
        drop_table(m, "agent_configs").await
    }
}
