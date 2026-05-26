use loco_rs::schema::*;
use sea_orm_migration::prelude::*;

#[derive(DeriveMigrationName)]
pub struct Migration;

#[async_trait::async_trait]
impl MigrationTrait for Migration {
    async fn up(&self, m: &SchemaManager) -> Result<(), DbErr> {
        create_table(m, "llm_configs",
            &[
            
            ("id", ColType::PkAuto),
            
            ("base_url", ColType::StringNull),
            ("api_key", ColType::StringNull),
            ("model_name", ColType::StringNull),
            ("temperature", ColType::FloatNull),
            ("max_tokens", ColType::BigIntegerNull),
            ("uuid", ColType::Uuid),
            ("user_id", ColType::UuidNull),
            ("status", ColType::StringNull),
            ("is_deleted", ColType::BooleanNull),
            ("remark", ColType::TextNull),
            ],
            &[
            ]
        ).await
    }

    async fn down(&self, m: &SchemaManager) -> Result<(), DbErr> {
        drop_table(m, "llm_configs").await
    }
}
