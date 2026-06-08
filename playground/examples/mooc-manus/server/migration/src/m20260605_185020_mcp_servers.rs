use loco_rs::schema::*;
use sea_orm_migration::prelude::*;

#[derive(DeriveMigrationName)]
pub struct Migration;

#[async_trait::async_trait]
impl MigrationTrait for Migration {
    async fn up(&self, m: &SchemaManager) -> Result<(), DbErr> {
        create_table(
            m,
            "mcp_servers",
            &[
                ("id", ColType::PkAuto),
                ("uuid", ColType::Uuid),
                ("user_id", ColType::UuidNull),
                ("status", ColType::StringNull),
                ("is_deleted", ColType::BooleanNull),
                ("remark", ColType::TextNull),
                ("transport", ColType::String),
                ("enabled", ColType::Boolean),
                ("description", ColType::TextNull),
                ("env", ColType::JsonNull),
                ("command", ColType::StringNull),
                ("args", ColType::array_null(ArrayColType::String)),
                ("url", ColType::StringNull),
                ("headers", ColType::JsonNull),
                ("name", ColType::String),
            ],
            &[],
        )
        .await
    }

    async fn down(&self, m: &SchemaManager) -> Result<(), DbErr> {
        drop_table(m, "mcp_servers").await
    }
}
