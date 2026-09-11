//! 文件相关集成测试共享的临时 PostgreSQL 夹具。
//!
//! 仅使用测试自行创建的实例和文件表迁移，不加载应用配置或连接现有数据库。
//! PostgreSQL 程序需位于 PATH，或用 FILE_REPOSITORY_PG_BIN 指定目录。

use std::{
    env,
    path::{Path, PathBuf},
    process::{Command, Output},
    time::Duration,
};

use anyhow::{bail, ensure, Context, Result};
use migration::{Migrator, MigratorTrait, SchemaManager};
use sea_orm::{ConnectOptions, Database, DatabaseConnection};
use tempfile::TempDir;

/// 每个测试使用独立实例；即使断言 panic，Drop 也会停止该实例并移除临时目录。
pub struct TestDatabase {
    pub db: DatabaseConnection,
    _postgres: TemporaryPostgres,
}

impl TestDatabase {
    pub async fn new() -> Result<Self> {
        let postgres = TemporaryPostgres::start()?;
        let mut options = ConnectOptions::new(format!(
            "postgres://postgres@localhost/postgres?host={}",
            postgres.directory.path().display()
        ));
        options
            .min_connections(1)
            .max_connections(8)
            .connect_timeout(Duration::from_secs(10))
            .sqlx_logging(false);
        let db = Database::connect(options)
            .await
            .context("无法连接测试自行创建的 PostgreSQL 实例")?;
        let fixture = Self {
            db,
            _postgres: postgres,
        };

        // 直接复用课程的建表和索引迁移，避免测试手写另一套表结构。
        let manager = SchemaManager::new(&fixture.db);
        let migrations = Migrator::migrations()
            .into_iter()
            .filter(|migration| {
                matches!(
                    migration.name(),
                    "m20260911_140128_files" | "m20260911_141144_fix_files_table"
                )
            })
            .collect::<Vec<_>>();
        ensure!(migrations.len() == 2, "没有找到文件表的两条迁移");
        for migration in migrations {
            migration.up(&manager).await?;
        }

        Ok(fixture)
    }
}

struct TemporaryPostgres {
    directory: TempDir,
    bin_dir: PathBuf,
}

impl TemporaryPostgres {
    fn start() -> Result<Self> {
        let bin_dir = postgres_bin_dir()?;
        // 使用短路径以避免 macOS 的 Unix socket 路径长度限制。
        let directory = tempfile::Builder::new()
            .prefix("file-repository-")
            .tempdir_in("/tmp")?;
        let data_dir = directory.path().join("data");
        command_output(
            Command::new(bin_dir.join("initdb"))
                .arg("-D")
                .arg(&data_dir)
                .args([
                    "--username=postgres",
                    "--auth=trust",
                    "--no-sync",
                    "--encoding=UTF8",
                    "--locale=C",
                ]),
        )?;

        let instance = Self { directory, bin_dir };
        command_output(
            Command::new(instance.bin_dir.join("pg_ctl"))
                .arg("-D")
                .arg(&data_dir)
                .arg("-l")
                .arg(instance.directory.path().join("postgres.log"))
                .args(["-w", "-t", "15", "-o"])
                .arg(format!(
                    "-c listen_addresses='' -c unix_socket_directories='{}' \
                     -c shared_buffers=8MB -c max_connections=16 -c fsync=off",
                    instance.directory.path().display()
                ))
                .arg("start"),
        )?;
        Ok(instance)
    }
}

impl Drop for TemporaryPostgres {
    fn drop(&mut self) {
        // 只停止本夹具 initdb 出来的数据目录，绝不操作系统已有 PostgreSQL 服务。
        let stopped = Command::new(self.bin_dir.join("pg_ctl"))
            .arg("-D")
            .arg(self.directory.path().join("data"))
            .args(["-m", "immediate", "-w", "-t", "10", "stop"])
            .output();
        if !matches!(stopped, Ok(ref output) if output.status.success()) {
            eprintln!("临时 PostgreSQL 停止失败：{stopped:?}");
        }
    }
}

fn postgres_bin_dir() -> Result<PathBuf> {
    if let Some(directory) = env::var_os("FILE_REPOSITORY_PG_BIN") {
        let directory = PathBuf::from(directory);
        ensure!(
            directory.join("initdb").is_file() && directory.join("pg_ctl").is_file(),
            "FILE_REPOSITORY_PG_BIN 必须指向包含 initdb 和 pg_ctl 的目录"
        );
        return Ok(directory);
    }

    let path = env::var_os("PATH").unwrap_or_default();
    env::split_paths(&path)
        .chain([
            Path::new("/usr/local/opt/postgresql@17/bin").to_path_buf(),
            Path::new("/opt/homebrew/opt/postgresql@17/bin").to_path_buf(),
        ])
        .find(|directory| directory.join("initdb").is_file() && directory.join("pg_ctl").is_file())
        .context("集成测试需要 PostgreSQL；请安装并设置 FILE_REPOSITORY_PG_BIN，测试不会跳过")
}

fn command_output(command: &mut Command) -> Result<Output> {
    let output = command
        .output()
        .with_context(|| format!("无法执行测试数据库程序：{command:?}"))?;
    if !output.status.success() {
        bail!(
            "测试数据库程序执行失败：{command:?}\n{}\n{}",
            String::from_utf8_lossy(&output.stdout),
            String::from_utf8_lossy(&output.stderr)
        );
    }
    Ok(output)
}
