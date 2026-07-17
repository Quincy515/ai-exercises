use loco_rs::schema::*;
use sea_orm_migration::prelude::*;

#[derive(DeriveMigrationName)]
pub struct Migration;

#[async_trait::async_trait]
impl MigrationTrait for Migration {
    async fn up(&self, m: &SchemaManager) -> Result<(), DbErr> {
        create_table(m, "a2a_servers",
            &[
            
            ("id", ColType::PkAuto),
            
            ("uuid", ColType::Uuid),
            ("base_url", ColType::StringNull),
            ("enabled", ColType::BooleanNull),
            ("agent_card", ColType::JsonNull),
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
        drop_table(m, "a2a_servers").await
    }
}
