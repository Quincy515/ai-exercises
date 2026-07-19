use sea_orm_migration::prelude::*;

#[derive(DeriveMigrationName)]
pub struct Migration;

#[async_trait::async_trait]
impl MigrationTrait for Migration {
    async fn up(&self, m: &SchemaManager) -> Result<(), DbErr> {
        m.get_connection()
            .execute_unprepared(r#"CREATE UNIQUE INDEX idx_a2a_servers_uuid ON a2a_servers(uuid)"#)
            .await?;

        m.get_connection()
            .execute_unprepared(r#"CREATE INDEX idx_a2a_servers_user_id ON a2a_servers(user_id)"#)
            .await?;

        m.get_connection()
            .execute_unprepared(r#"COMMENT ON TABLE a2a_servers IS 'A2A 服务器配置';"#)
            .await?;
        m.get_connection()
            .execute_unprepared(r#"COMMENT ON COLUMN a2a_servers.id IS 'A2A 服务器配置 ID';"#)
            .await?;
        m.get_connection()
            .execute_unprepared(r#"COMMENT ON COLUMN a2a_servers.uuid IS 'A2A 服务器配置 UUID';"#)
            .await?;
        m.get_connection()
            .execute_unprepared(
                r#"COMMENT ON COLUMN a2a_servers.base_url IS 'A2A 服务器配置基础 URL';"#,
            )
            .await?;
        m.get_connection()
            .execute_unprepared(
                r#"COMMENT ON COLUMN a2a_servers.enabled IS 'A2A 服务器配置是否启用';"#,
            )
            .await?;
        m.get_connection()
            .execute_unprepared(
                r#"COMMENT ON COLUMN a2a_servers.agent_card IS 'A2A 服务器配置详情';"#,
            )
            .await?;
        m.get_connection()
            .execute_unprepared(r#"COMMENT ON COLUMN a2a_servers.user_id IS '用户 UUID';"#)
            .await?;
        m.get_connection()
            .execute_unprepared(r#"COMMENT ON COLUMN a2a_servers.status IS '状态';"#)
            .await?;
        m.get_connection()
            .execute_unprepared(r#"COMMENT ON COLUMN a2a_servers.is_deleted IS '是否已删除';"#)
            .await?;
        m.get_connection()
            .execute_unprepared(r#"COMMENT ON COLUMN a2a_servers.remark IS '备注';"#)
            .await?;

        Ok(())
    }

    async fn down(&self, m: &SchemaManager) -> Result<(), DbErr> {
        m.get_connection()
            .execute_unprepared(r#"DROP INDEX idx_a2a_servers_uuid;"#)
            .await?;
        m.get_connection()
            .execute_unprepared(r#"DROP INDEX idx_a2a_servers_user_id;"#)
            .await?;
        Ok(())
    }
}
