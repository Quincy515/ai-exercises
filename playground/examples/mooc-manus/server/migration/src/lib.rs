#![allow(elided_lifetimes_in_paths)]
#![allow(clippy::wildcard_imports)]
pub use sea_orm_migration::prelude::*;
mod m20220101_000001_users;

mod m20260526_131658_llm_configs;
mod m20260526_134746_fix_llm_configs_table;
pub struct Migrator;

#[async_trait::async_trait]
impl MigratorTrait for Migrator {
    fn migrations() -> Vec<Box<dyn MigrationTrait>> {
        vec![
            Box::new(m20220101_000001_users::Migration),
            Box::new(m20260526_131658_llm_configs::Migration),
            Box::new(m20260526_134746_fix_llm_configs_table::Migration),
            // inject-above (do not remove this comment)
        ]
    }
}