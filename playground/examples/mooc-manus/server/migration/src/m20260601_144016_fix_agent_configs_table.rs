use sea_orm_migration::prelude::*;

#[derive(DeriveMigrationName)]
pub struct Migration;

#[async_trait::async_trait]
impl MigrationTrait for Migration {
    async fn up(&self, m: &SchemaManager) -> Result<(), DbErr> {
        m.get_connection()
            .execute_unprepared(r#"CREATE INDEX idx_agent_configs_uuid ON agent_configs(uuid)"#)
            .await?;

        m.get_connection()
            .execute_unprepared(
                r#"CREATE INDEX idx_agent_configs_user_id ON agent_configs(user_id)"#,
            )
            .await?;

        m.get_connection()
            .execute_unprepared(r#"COMMENT ON TABLE agent_configs IS 'Agent 配置';"#)
            .await?;
        m.get_connection()
            .execute_unprepared(r#"COMMENT ON COLUMN agent_configs.id IS 'Agent 配置 ID';"#)
            .await?;
        m.get_connection()
            .execute_unprepared(
                r#"COMMENT ON COLUMN agent_configs.max_iterations IS 'Agent 最大迭代次数必须在 1 到 999 之间';"#,
            )
            .await?;
        m.get_connection()
            .execute_unprepared(
                r#"COMMENT ON COLUMN agent_configs.max_retries IS 'Agent 最大重试次数必须在 2 到 9 之间';"#,
            )
            .await?;
        m.get_connection()
            .execute_unprepared(
                r#"COMMENT ON COLUMN agent_configs.max_search_results IS 'Agent 最大搜索结果必须在 2 到 29 之间';"#,
            )
            .await?;
        m.get_connection()
            .execute_unprepared(r#"COMMENT ON COLUMN agent_configs.uuid IS 'Agent 配置 UUID';"#)
            .await?;
        m.get_connection()
            .execute_unprepared(r#"COMMENT ON COLUMN agent_configs.user_id IS '用户 UUID';"#)
            .await?;
        m.get_connection()
            .execute_unprepared(r#"COMMENT ON COLUMN agent_configs.status IS '状态';"#)
            .await?;
        m.get_connection()
            .execute_unprepared(r#"COMMENT ON COLUMN agent_configs.is_deleted IS '是否已删除';"#)
            .await?;
        m.get_connection()
            .execute_unprepared(r#"COMMENT ON COLUMN agent_configs.remark IS '备注';"#)
            .await?;

        Ok(())
    }

    async fn down(&self, m: &SchemaManager) -> Result<(), DbErr> {
        m.get_connection()
            .execute_unprepared(r#"DROP INDEX idx_agent_configs_uuid;"#)
            .await?;
        m.get_connection()
            .execute_unprepared(r#"DROP INDEX idx_agent_configs_user_id;"#)
            .await?;
        Ok(())
    }
}
