use loco_rs::schema::*;
use sea_orm_migration::prelude::*;

#[derive(DeriveMigrationName)]
pub struct Migration;

#[async_trait::async_trait]
impl MigrationTrait for Migration {
    async fn up(&self, m: &SchemaManager) -> Result<(), DbErr> {
        create_table(m, "files",
            &[
            
            ("id", ColType::PkAuto),
            
            ("uuid", ColType::UuidNull),
            ("filename", ColType::StringNull),
            ("filepath", ColType::StringNull),
            ("key", ColType::StringNull),
            ("extension", ColType::StringNull),
            ("mime_type", ColType::StringNull),
            ("size", ColType::IntegerNull),
            ("user_id", ColType::UuidNull),
            ("is_deleted", ColType::BooleanNull),
            ("remark", ColType::TextNull),
            ],
            &[
            ]
        ).await
    }

    async fn down(&self, m: &SchemaManager) -> Result<(), DbErr> {
        drop_table(m, "files").await
    }
}
