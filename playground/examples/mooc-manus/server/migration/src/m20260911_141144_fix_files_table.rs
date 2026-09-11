use sea_orm_migration::prelude::*;

#[derive(DeriveMigrationName)]
pub struct Migration;

#[async_trait::async_trait]
impl MigrationTrait for Migration {
    async fn up(&self, m: &SchemaManager) -> Result<(), DbErr> {
        m.get_connection()
            .execute_unprepared(r#"CREATE UNIQUE INDEX idx_files_uuid ON files(uuid)"#)
            .await?;

        m.get_connection()
            .execute_unprepared(r#"CREATE INDEX idx_files_user_id ON files(user_id)"#)
            .await?;

        m.get_connection()
            .execute_unprepared(r#"COMMENT ON TABLE files IS '文件数据库表';"#)
            .await?;
        m.get_connection()
            .execute_unprepared(r#"COMMENT ON COLUMN files.id IS '文件表 ID';"#)
            .await?;
        m.get_connection()
            .execute_unprepared(r#"COMMENT ON COLUMN files.uuid IS '文件表 UUID';"#)
            .await?;
        m.get_connection()
            .execute_unprepared(r#"COMMENT ON COLUMN files.filename IS '文件名';"#)
            .await?;
        m.get_connection()
            .execute_unprepared(r#"COMMENT ON COLUMN files.filepath IS '文件路径';"#)
            .await?;
        m.get_connection()
            .execute_unprepared(r#"COMMENT ON COLUMN files.key IS '文件云存储中的路径';"#)
            .await?;
        m.get_connection()
            .execute_unprepared(r#"COMMENT ON COLUMN files.extension IS '文件扩展名';"#)
            .await?;
        m.get_connection()
            .execute_unprepared(r#"COMMENT ON COLUMN files.mime_type IS '文件 MIME 类型';"#)
            .await?;
        m.get_connection()
            .execute_unprepared(r#"COMMENT ON COLUMN files.size IS '文件大小，单位为字节';"#)
            .await?;
        m.get_connection()
            .execute_unprepared(r#"COMMENT ON COLUMN files.user_id IS '用户 UUID';"#)
            .await?;
        m.get_connection()
            .execute_unprepared(r#"COMMENT ON COLUMN files.is_deleted IS '是否已删除';"#)
            .await?;
        m.get_connection()
            .execute_unprepared(r#"COMMENT ON COLUMN files.remark IS '备注';"#)
            .await?;

        Ok(())
    }

    async fn down(&self, m: &SchemaManager) -> Result<(), DbErr> {
        m.get_connection()
            .execute_unprepared(r#"DROP INDEX idx_files_uuid;"#)
            .await?;
        m.get_connection()
            .execute_unprepared(r#"DROP INDEX idx_files_user_id;"#)
            .await?;
        Ok(())
    }
}
