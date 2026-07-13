use std::{path::Path, process::Stdio};

use tokio::{fs, io::AsyncWriteExt, process::Command};
use tracing::error;

use crate::{
    exceptions::AppException,
    models::{FileReadResult, FileWriteResult},
};

const TRUNCATED_MARKER: &str = "(truncated)";
// 沙箱文本统一使用 UTF-8；若要支持 Windows 本地代码页，应在文件 I/O 边界单独引入编码转换。

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
        file_path: &str,
        start_line: Option<usize>,
        end_line: Option<usize>,
        sudo: Option<bool>,
        max_length: Option<usize>,
    ) -> Result<FileReadResult, AppException> {
        let sudo = sudo.unwrap_or(false);

        // 1.检测在当前权限下能否获取该文件
        if !sudo && !fs::try_exists(file_path).await.unwrap_or(false) {
            let message = format!("要读取的文件不存在或无权限: {file_path}");
            error!("{message}");
            return Err(AppException::not_found(message));
        }

        // 2.判断是否为sudo+非windows系统，如果是则使用命令行的方式读取文件
        let mut content = if sudo && !cfg!(windows) {
            // 3.使用sudo cat命令读取文件内容
            let output = Command::new("sudo")
                .arg("cat")
                .arg("--")
                .arg(file_path)
                .output()
                .await
                .map_err(|err| AppException::bad_request(format!("读取文件失败: {err}")))?;

            // 4.判断子进程的状态是否正常结束，并按UTF-8读取输出内容
            if !output.status.success() {
                let stderr = String::from_utf8_lossy(&output.stderr);
                return Err(AppException::bad_request(format!("读取文件失败: {stderr}")));
            }

            String::from_utf8_lossy(&output.stdout).into_owned()
        } else {
            // 5.使用Tokio的阻塞线程池按UTF-8读取文件，避免阻塞异步运行时
            fs::read_to_string(file_path)
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
            file_path: file_path.to_string(),
            content,
        })
    }

    /// 根据传递的文件路径+内容向指定文件写入内容
    pub async fn write_file(
        &self,
        file_path: &str,
        mut content: String,
        append: Option<bool>,
        leading_newline: Option<bool>,
        trailing_newline: Option<bool>,
        sudo: Option<bool>,
    ) -> Result<FileWriteResult, AppException> {
        let append = append.unwrap_or(false);

        // 1.组装实际写入的内容
        if leading_newline.unwrap_or(false) {
            content.insert(0, '\n');
        }
        if trailing_newline.unwrap_or(false) {
            content.push('\n');
        }
        let bytes_written = content.len();

        // 2.判断是否是sudo权限，并且不是windows系统
        if sudo.unwrap_or(false) && !cfg!(windows) {
            // 3.通过sudo tee的标准输入写入内容，避免临时文件和Shell字符串拼接
            let mut command = Command::new("sudo");
            command.arg("tee");
            if append {
                command.arg("-a");
            }

            // 4.创建子进程并打开标准输入、错误输出管道
            let mut child = command
                .arg("--")
                .arg(file_path)
                .stdin(Stdio::piped())
                .stdout(Stdio::null())
                .stderr(Stdio::piped())
                .spawn()
                .map_err(|err| AppException::bad_request(format!("文件内容写入失败: {err}")))?;

            // 5.将完整内容写入子进程并关闭标准输入
            let mut stdin = child
                .stdin
                .take()
                .ok_or_else(|| AppException::bad_request("文件内容写入失败: 无法打开标准输入"))?;
            if let Err(err) = stdin.write_all(content.as_bytes()).await {
                let _ = child.kill().await;
                return Err(AppException::bad_request(format!(
                    "文件内容写入失败: {err}"
                )));
            }
            drop(stdin);

            // 6.等待子进程结束并检查执行状态
            let output = child
                .wait_with_output()
                .await
                .map_err(|err| AppException::bad_request(format!("文件内容写入失败: {err}")))?;
            if !output.status.success() {
                let stderr = String::from_utf8_lossy(&output.stderr);
                return Err(AppException::bad_request(format!(
                    "文件内容写入失败: {stderr}"
                )));
            }
        } else {
            // 7.非sudo或者windows下直接写入，先确保父目录存在
            if let Some(parent) = Path::new(file_path)
                .parent()
                .filter(|parent| !parent.as_os_str().is_empty())
            {
                fs::create_dir_all(parent)
                    .await
                    .map_err(|err| AppException::internal(format!("文件内容写入失败: {err}")))?;
            }

            // 8.根据追加模式打开文件
            let mut options = fs::OpenOptions::new();
            options.create(true).write(true);
            if append {
                options.append(true);
            } else {
                options.truncate(true);
            }

            // 9.异步写入完整内容
            let mut file = options
                .open(file_path)
                .await
                .map_err(|err| AppException::internal(format!("文件内容写入失败: {err}")))?;
            file.write_all(content.as_bytes())
                .await
                .map_err(|err| AppException::internal(format!("文件内容写入失败: {err}")))?;
            file.flush()
                .await
                .map_err(|err| AppException::internal(format!("文件内容写入失败: {err}")))?;
        }

        // 10.返回文件路径和UTF-8字节数
        Ok(FileWriteResult {
            file_path: file_path.to_string(),
            bytes_written: Some(bytes_written),
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

    struct TestDirectory(PathBuf);

    impl TestDirectory {
        fn new() -> Self {
            Self(std::env::temp_dir().join(format!("sandbox-write-{}", Uuid::new_v4())))
        }

        fn file_path(&self, relative_path: &str) -> PathBuf {
            self.0.join(relative_path)
        }
    }

    impl Drop for TestDirectory {
        fn drop(&mut self) {
            let _ = std::fs::remove_dir_all(&self.0);
        }
    }

    #[tokio::test]
    async fn reads_the_requested_line_range() {
        let file = TestFile::create("第一行\n第二行\n第三行\n第四行\n").await;

        let result = FileService::new()
            .read_file(file.as_str(), Some(1), Some(3), Some(false), None)
            .await
            .expect("file range should be readable");

        assert_eq!(result.file_path, file.as_str());
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

    #[tokio::test]
    async fn writes_utf8_content_and_creates_parent_directories() {
        let directory = TestDirectory::new();
        let path = directory.file_path("nested/agent.txt");
        let content = "你好Agent";

        let result = FileService::new()
            .write_file(
                path.to_str().expect("test path should be utf-8"),
                content.to_string(),
                Some(false),
                Some(true),
                Some(true),
                Some(false),
            )
            .await
            .expect("file should be writable");

        let expected = format!("\n{content}\n");
        assert_eq!(fs::read_to_string(&path).await.unwrap(), expected);
        assert_eq!(result.file_path, path.to_str().unwrap());
        assert_eq!(result.bytes_written, Some(expected.len()));
    }

    #[tokio::test]
    async fn appends_content_without_overwriting_existing_data() {
        let file = TestFile::create("第一行").await;

        let result = FileService::new()
            .write_file(
                file.as_str(),
                "第二行".to_string(),
                Some(true),
                Some(true),
                Some(false),
                Some(false),
            )
            .await
            .expect("file should be appendable");

        assert_eq!(fs::read_to_string(&file.0).await.unwrap(), "第一行\n第二行");
        assert_eq!(result.bytes_written, Some("\n第二行".len()));
    }
}
