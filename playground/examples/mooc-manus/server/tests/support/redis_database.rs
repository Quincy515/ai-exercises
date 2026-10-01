//! SSE 集成测试专用 Redis：独立 Unix socket，关闭 TCP 与持久化。

use std::{
    env,
    path::PathBuf,
    process::{Child, Command, Stdio},
    time::Duration,
};

use anyhow::{bail, Context, Result};
use tempfile::TempDir;

pub struct TestRedis {
    pub client: redis::Client,
    pub uri: String,
    child: Child,
    _directory: TempDir,
}

impl TestRedis {
    pub async fn new() -> Result<Self> {
        let directory = tempfile::Builder::new()
            .prefix("chat-redis-")
            .tempdir_in("/tmp")?;
        let socket = directory.path().join("redis.sock");
        let uri = format!("redis+unix://{}", socket.display());
        let client = redis::Client::open(uri.as_str())?;
        let program = env::var_os("CHAT_TEST_REDIS_BIN")
            .map(PathBuf::from)
            .unwrap_or_else(|| PathBuf::from("redis-server"));
        let child = Command::new(program)
            .args(["--port", "0", "--unixsocket"])
            .arg(&socket)
            .args([
                "--unixsocketperm",
                "700",
                "--save",
                "",
                "--appendonly",
                "no",
                "--daemonize",
                "no",
                "--dir",
            ])
            .arg(directory.path())
            .arg("--logfile")
            .arg(directory.path().join("redis.log"))
            .stdout(Stdio::null())
            .stderr(Stdio::null())
            .spawn()
            .context("SSE 集成测试需要 redis-server；可用 CHAT_TEST_REDIS_BIN 指定程序")?;
        let mut instance = Self {
            client,
            uri,
            child,
            _directory: directory,
        };
        for _ in 0..100 {
            if let Some(status) = instance.child.try_wait()? {
                bail!("测试自行创建的 Redis 启动失败：{status}");
            }
            if let Ok(mut connection) = instance.client.get_multiplexed_async_connection().await {
                if redis::cmd("PING")
                    .query_async::<String>(&mut connection)
                    .await
                    .is_ok()
                {
                    return Ok(instance);
                }
            }
            tokio::time::sleep(Duration::from_millis(20)).await;
        }
        bail!("测试自行创建的 Redis 未在两秒内启动");
    }
}

impl Drop for TestRedis {
    fn drop(&mut self) {
        // 只关闭本夹具 spawn 的子进程，等待退出后再释放临时目录。
        let _ = self.child.kill();
        let _ = self.child.wait();
    }
}
