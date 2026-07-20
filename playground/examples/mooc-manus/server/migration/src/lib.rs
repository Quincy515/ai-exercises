#![allow(elided_lifetimes_in_paths)]
#![allow(clippy::wildcard_imports)]
pub use sea_orm_migration::prelude::*;
mod m20220101_000001_users;

mod m20260526_131658_llm_configs;
mod m20260526_134746_fix_llm_configs_table;
mod m20260601_143631_agent_configs;
mod m20260601_144016_fix_agent_configs_table;
mod m20260605_185020_mcp_servers;
mod m20260717_191151_a2a_servers;
mod m20260718_113716_fix_a2a_servers_table;
mod m20260720_184611_sessions;
mod m20260720_191303_fix_sessions_table;
pub struct Migrator;

#[async_trait::async_trait]
impl MigratorTrait for Migrator {
    fn migrations() -> Vec<Box<dyn MigrationTrait>> {
        vec![
            Box::new(m20220101_000001_users::Migration),
            Box::new(m20260526_131658_llm_configs::Migration),
            Box::new(m20260526_134746_fix_llm_configs_table::Migration),
            Box::new(m20260601_143631_agent_configs::Migration),
            Box::new(m20260601_144016_fix_agent_configs_table::Migration),
            Box::new(m20260605_185020_mcp_servers::Migration),
            Box::new(m20260717_191151_a2a_servers::Migration),
            Box::new(m20260718_113716_fix_a2a_servers_table::Migration),
            Box::new(m20260720_184611_sessions::Migration),
            Box::new(m20260720_191303_fix_sessions_table::Migration),
            // inject-above (do not remove this comment)
        ]
    }
}
