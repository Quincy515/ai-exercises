use sea_orm_migration::prelude::*;

#[derive(DeriveMigrationName)]
pub struct Migration;

#[async_trait::async_trait]
impl MigrationTrait for Migration {
    async fn up(&self, m: &SchemaManager) -> Result<(), DbErr> {
        m.get_connection()
            .execute_unprepared(r#"CREATE INDEX idx_llm_configs_uuid ON llm_configs(uuid)"#)
            .await?;

        m.get_connection()
            .execute_unprepared(r#"CREATE INDEX idx_llm_configs_user_id ON llm_configs(user_id)"#)
            .await?;

        m.get_connection()
            .execute_unprepared(r#"COMMENT ON TABLE llm_configs IS 'LLM 配置';"#)
            .await?;
        m.get_connection()
            .execute_unprepared(r#"COMMENT ON COLUMN llm_configs.id IS 'LLM 配置 ID';"#)
            .await?;
        m.get_connection()
            .execute_unprepared(r#"COMMENT ON COLUMN llm_configs.base_url IS 'LLM 服务基础地址';"#)
            .await?;
        m.get_connection()
            .execute_unprepared(r#"COMMENT ON COLUMN llm_configs.api_key IS 'LLM API 密钥';"#)
            .await?;
        m.get_connection()
            .execute_unprepared(r#"COMMENT ON COLUMN llm_configs.model_name IS 'LLM 模型名称';"#)
            .await?;
        m.get_connection()
            .execute_unprepared(r#"COMMENT ON COLUMN llm_configs.temperature IS '采样温度';"#)
            .await?;
        m.get_connection()
            .execute_unprepared(
                r#"COMMENT ON COLUMN llm_configs.max_tokens IS '最大输出 token 数';"#,
            )
            .await?;
        m.get_connection()
            .execute_unprepared(r#"COMMENT ON COLUMN llm_configs.uuid IS 'LLM 配置 UUID';"#)
            .await?;
        m.get_connection()
            .execute_unprepared(r#"COMMENT ON COLUMN llm_configs.user_id IS '用户 UUID';"#)
            .await?;
        m.get_connection()
            .execute_unprepared(r#"COMMENT ON COLUMN llm_configs.status IS '状态';"#)
            .await?;
        m.get_connection()
            .execute_unprepared(r#"COMMENT ON COLUMN llm_configs.is_deleted IS '是否已删除';"#)
            .await?;
        m.get_connection()
            .execute_unprepared(r#"COMMENT ON COLUMN llm_configs.remark IS '备注';"#)
            .await?;

        Ok(())
    }

    async fn down(&self, m: &SchemaManager) -> Result<(), DbErr> {
        m.get_connection()
            .execute_unprepared(r#"DROP INDEX idx_llm_configs_uuid;"#)
            .await?;
        m.get_connection()
            .execute_unprepared(r#"DROP INDEX idx_llm_configs_user_id;"#)
            .await?;
        Ok(())
    }
}
