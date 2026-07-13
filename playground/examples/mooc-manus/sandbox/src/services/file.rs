use tokio::{fs, process::Command};
use tracing::error;

use crate::{exceptions::AppException, models::FileReadResult};

const TRUNCATED_MARKER: &str = "(truncated)";

/// 文件沙箱服务
#[derive(Debug, Default)]
pub struct FileService;

impl FileService {
    pub const fn new() -> Self {
        Self
    }

    /// 根据传递的文件路径+起始行号+权限+最大长度读取文件内容
    pub async fn read_file(
        &self,
        filepath: &str,
        start_line: Option<usize>,
        end_line: Option<usize>,
        sudo: Option<bool>,
        max_length: Option<usize>,
    ) -> Result<FileReadResult, AppException> {
        let sudo = sudo.unwrap_or(false);

        // 1.检测在当前权限下能否获取该文件
        if !sudo && !fs::try_exists(filepath).await.unwrap_or(false) {
            let message = format!("要读取的文件不存在或无权限: {filepath}");
            error!("{message}");
            return Err(AppException::not_found(message));
        }

        // 2.判断是否为sudo+非windows系统，如果是则使用命令行的方式读取文件
        let mut content = if sudo && !cfg!(windows) {
            // 3.使用sudo cat命令读取文件内容
            let output = Command::new("sudo")
                .arg("cat")
                .arg("--")
                .arg(filepath)
                .output()
                .await
                .map_err(|err| AppException::bad_request(format!("读取文件失败: {err}")))?;

            // 4.判断子进程的状态是否正常结束并读取输出内容
            if !output.status.success() {
                let stderr = String::from_utf8_lossy(&output.stderr);
                return Err(AppException::bad_request(format!("读取文件失败: {stderr}")));
            }

            String::from_utf8_lossy(&output.stdout).into_owned()
        } else {
            // 5.使用Tokio的阻塞线程池读取文件，避免阻塞异步运行时
            fs::read_to_string(filepath)
                .await
                .map_err(|err| AppException::internal(format!("读取文件失败: {err}")))?
        };

        // 6.判断是否传递了读取范围，将内容切割成行并提取指定范围
        if start_line.is_some() || end_line.is_some() {
            let start = start_line.unwrap_or(0);
            let end = end_line.unwrap_or(usize::MAX);
            content = content
                .lines()
                .skip(start)
                .take(end.saturating_sub(start))
                .collect::<Vec<_>>()
                .join("\n");
        }

        // 7.按字符数裁切内容，避免截断UTF-8字符
        if let Some(max_length) = max_length
            && max_length > 0
            && content.chars().count() > max_length
        {
            content = content.chars().take(max_length).collect::<String>() + TRUNCATED_MARKER;
        }

        Ok(FileReadResult {
            filepath: filepath.to_string(),
            content,
        })
    }
}

#[cfg(test)]
mod tests {
    use std::path::PathBuf;

    use axum::http::StatusCode;
    use uuid::Uuid;

    use super::*;

    struct TestFile(PathBuf);

    impl TestFile {
        async fn create(content: &str) -> Self {
            let path = std::env::temp_dir().join(format!("sandbox-read-{}", Uuid::new_v4()));
            fs::write(&path, content)
                .await
                .expect("test file should be writable");
            Self(path)
        }

        fn as_str(&self) -> &str {
            self.0.to_str().expect("test path should be utf-8")
        }
    }

    impl Drop for TestFile {
        fn drop(&mut self) {
            let _ = std::fs::remove_file(&self.0);
        }
    }

    #[tokio::test]
    async fn reads_the_requested_line_range() {
        let file = TestFile::create("第一行\n第二行\n第三行\n第四行\n").await;

        let result = FileService::new()
            .read_file(file.as_str(), Some(1), Some(3), Some(false), None)
            .await
            .expect("file range should be readable");

        assert_eq!(result.filepath, file.as_str());
        assert_eq!(result.content, "第二行\n第三行");
    }

    #[tokio::test]
    async fn truncates_by_unicode_character_count() {
        let file = TestFile::create("你好Rust").await;

        let result = FileService::new()
            .read_file(file.as_str(), None, None, Some(false), Some(3))
            .await
            .expect("file should be readable");

        assert_eq!(result.content, "你好R(truncated)");
    }

    #[tokio::test]
    async fn reports_a_missing_file_as_not_found() {
        let path = std::env::temp_dir().join(format!("sandbox-missing-{}", Uuid::new_v4()));

        let error = FileService::new()
            .read_file(
                path.to_str().expect("test path should be utf-8"),
                None,
                None,
                Some(false),
                Some(10_000),
            )
            .await
            .expect_err("missing file should fail");

        assert_eq!(error.status_code, StatusCode::NOT_FOUND);
        assert!(error.msg.contains("要读取的文件不存在或无权限"));
    }
}
