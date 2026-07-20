use sea_orm_migration::prelude::*;

#[derive(DeriveMigrationName)]
pub struct Migration;

#[async_trait::async_trait]
impl MigrationTrait for Migration {
    async fn up(&self, m: &SchemaManager) -> Result<(), DbErr> {
        m.get_connection()
            .execute_unprepared(r#"CREATE UNIQUE INDEX idx_sessions_uuid ON sessions(uuid)"#)
            .await?;

        m.get_connection()
            .execute_unprepared(r#"CREATE INDEX idx_sessions_user_id ON sessions(user_id)"#)
            .await?;

        m.get_connection()
            .execute_unprepared(r#"COMMENT ON TABLE sessions IS '会话数据库表';"#)
            .await?;
        m.get_connection()
            .execute_unprepared(r#"COMMENT ON COLUMN sessions.id IS '会话表 ID';"#)
            .await?;
        m.get_connection()
            .execute_unprepared(r#"COMMENT ON COLUMN sessions.uuid IS '会话表 UUID';"#)
            .await?;
        m.get_connection()
            .execute_unprepared(r#"COMMENT ON COLUMN sessions.sandbox_id IS '沙箱 ID';"#)
            .await?;
        m.get_connection()
            .execute_unprepared(r#"COMMENT ON COLUMN sessions.task_id IS '任务 ID';"#)
            .await?;
        m.get_connection()
            .execute_unprepared(r#"COMMENT ON COLUMN sessions.title IS '会话标题';"#)
            .await?;
        m.get_connection()
            .execute_unprepared(
                r#"COMMENT ON COLUMN sessions.unread_message_count IS '未读消息数';"#,
            )
            .await?;
        m.get_connection()
            .execute_unprepared(r#"COMMENT ON COLUMN sessions.latest_message IS '最后一条消息';"#)
            .await?;
        m.get_connection()
            .execute_unprepared(
                r#"COMMENT ON COLUMN sessions.latest_message_at IS '最后一条消息时间';"#,
            )
            .await?;
        m.get_connection()
            .execute_unprepared(r#"COMMENT ON COLUMN sessions.events IS '事件列表';"#)
            .await?;
        m.get_connection()
            .execute_unprepared(r#"COMMENT ON COLUMN sessions.files IS '文件列表';"#)
            .await?;
        m.get_connection()
            .execute_unprepared(
                r#"COMMENT ON COLUMN sessions.memories IS '会话两个 Agent 的记忆';"#,
            )
            .await?;
        m.get_connection()
            .execute_unprepared(r#"COMMENT ON COLUMN sessions.user_id IS '用户 UUID';"#)
            .await?;
        m.get_connection()
            .execute_unprepared(r#"COMMENT ON COLUMN sessions.status IS '会话状态';"#)
            .await?;
        m.get_connection()
            .execute_unprepared(r#"COMMENT ON COLUMN sessions.is_deleted IS '是否已删除';"#)
            .await?;
        m.get_connection()
            .execute_unprepared(r#"COMMENT ON COLUMN sessions.remark IS '备注';"#)
            .await?;

        Ok(())
    }

    async fn down(&self, m: &SchemaManager) -> Result<(), DbErr> {
        m.get_connection()
            .execute_unprepared(r#"DROP INDEX idx_sessions_uuid;"#)
            .await?;
        m.get_connection()
            .execute_unprepared(r#"DROP INDEX idx_sessions_user_id;"#)
            .await?;
        Ok(())
    }
}
