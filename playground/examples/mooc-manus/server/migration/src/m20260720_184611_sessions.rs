use loco_rs::schema::*;
use sea_orm_migration::prelude::*;

#[derive(DeriveMigrationName)]
pub struct Migration;

#[async_trait::async_trait]
impl MigrationTrait for Migration {
    async fn up(&self, m: &SchemaManager) -> Result<(), DbErr> {
        create_table(
            m,
            "sessions",
            &[
                ("id", ColType::PkAuto),
                ("uuid", ColType::UuidNull),
                ("sandbox_id", ColType::StringNull),
                ("task_id", ColType::StringNull),
                ("title", ColType::StringNull),
                ("unread_message_count", ColType::IntegerNull),
                ("latest_message", ColType::TextNull),
                ("latest_message_at", ColType::TimestampWithTimeZoneNull),
                ("events", ColType::JsonBinaryNull),
                ("files", ColType::JsonBinaryNull),
                ("memories", ColType::JsonBinaryNull),
                ("status", ColType::StringNull),
                ("user_id", ColType::UuidNull),
                ("is_deleted", ColType::BooleanNull),
                ("remark", ColType::TextNull),
            ],
            &[],
        )
        .await
    }

    async fn down(&self, m: &SchemaManager) -> Result<(), DbErr> {
        drop_table(m, "sessions").await
    }
}
