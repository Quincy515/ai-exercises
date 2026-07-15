use std::{path::Path, process::Stdio};

use glob::glob;
use regex::Regex;
use tokio::{fs, io::AsyncWriteExt, process::Command};
use tracing::error;

use crate::{
    exceptions::AppException,
    models::{
        FileCheckResult, FileDeleteResult, FileFindResult, FileReadResult, FileReplaceResult,
        FileSearchResult, FileUploadResult, FileWriteResult,
    },
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

    /// 根据传递的数据替换文件内指定的内容
    pub async fn replace_in_file(
        &self,
        file_path: &str,
        old_str: &str,
        new_str: &str,
        sudo: Option<bool>,
    ) -> Result<FileReplaceResult, AppException> {
        // 1.调用服务获取对应的文件内容
        let content = self
            .read_file(file_path, None, None, sudo, None)
            .await?
            .content;

        // 2.计算old_str出现的次数，只有出现次数>0才需要替换
        let replaced_count = content.matches(old_str).count();

        // 4.将替换后的新内容写入到文件中
        if replaced_count > 0 {
            self.write_file(
                file_path,
                content.replace(old_str, new_str), // 3.替换旧内容
                Some(false),
                Some(false),
                Some(false),
                sudo,
            )
            .await?;
        }

        Ok(FileReplaceResult {
            file_path: file_path.to_string(),
            replaced_count,
        })
    }

    /// 根据传递的文件路径和正则表达式查询文件内符合的内容
    pub async fn search_in_file(
        &self,
        file_path: &str,
        regex: &str,
        sudo: Option<bool>,
    ) -> Result<FileSearchResult, AppException> {
        // 1.调用服务获取对应的文件内容
        let content = self
            .read_file(file_path, None, None, sudo, None)
            .await?
            .content;
        // 2.将外部传递的regex转换为正则
        let pattern = Regex::new(regex).map_err(|err| {
            AppException::bad_request(format!("传递正则表达式[{regex}]出错: {err}"))
        })?;

        // 4.创建一个异步函数，使用子线程方式执行避免长时间io阻塞
        let (matches, line_numbers) = tokio::task::spawn_blocking(move || {
            // 3.将读取的内容拆分成每一行
            content
                .lines()
                .enumerate()
                .filter_map(|(line_number, line)| {
                    pattern
                        .find(line)
                        .filter(|matched| matched.start() == 0)
                        .map(|_| (line.to_string(), line_number))
                })
                .unzip()
        })
        .await
        .map_err(|err| AppException::internal(format!("搜索文件内容失败: {err}")))?;

        Ok(FileSearchResult {
            file_path: file_path.to_string(),
            matches,
            line_numbers,
        })
    }

    /// 根据传递的文件夹路径和glob规则查询文件列表
    pub async fn find_files(
        &self,
        dir_path: &str,
        glob_pattern: &str,
    ) -> Result<FileFindResult, AppException> {
        // 1.检测下传递进来的目录是否存在
        if !fs::try_exists(dir_path).await.unwrap_or(false) {
            return Err(AppException::not_found(format!(
                "当前文件夹不存在: {dir_path}"
            )));
        }

        // 2.将外部传递的glob_pattern转换为搜索模式
        let search_pattern = Path::new(dir_path)
            .join(glob_pattern)
            .to_string_lossy()
            .into_owned();

        // 3.创建一个异步函数，使用子线程方式执行避免长时间io阻塞
        let mut files = tokio::task::spawn_blocking(move || {
            let entries = glob(&search_pattern)
                .map_err(|err| AppException::bad_request(format!("glob文件规则无效: {err}")))?;

            entries
                .map(|entry| {
                    entry
                        .map(|path| path.to_string_lossy().into_owned())
                        .map_err(|err| AppException::internal(format!("查找文件失败: {err}")))
                })
                .collect::<Result<Vec<_>, _>>()
        })
        .await
        .map_err(|err| AppException::internal(format!("查找文件失败: {err}")))??;
        files.sort();

        Ok(FileFindResult {
            dir_path: dir_path.to_string(),
            files,
        })
    }

    /// 将分块接收完成的临时文件上传至目标路径
    pub async fn upload_file(
        &self,
        source_path: &Path,
        file_path: &str,
    ) -> Result<FileUploadResult, AppException> {
        if let Some(parent) = Path::new(file_path)
            .parent()
            .filter(|parent| !parent.as_os_str().is_empty())
        {
            fs::create_dir_all(parent)
                .await
                .map_err(|err| AppException::internal(format!("上传文件到沙箱出错: {err}")))?;
        }

        let file_size = fs::copy(source_path, file_path)
            .await
            .map_err(|err| AppException::internal(format!("上传文件到沙箱出错: {err}")))?;
        fs::remove_file(source_path)
            .await
            .map_err(|err| AppException::internal(format!("清理上传临时文件失败: {err}")))?;

        Ok(FileUploadResult {
            file_path: file_path.to_string(),
            file_size,
            success: true,
        })
    }

    /// 传递 filepath 用于确保当前文件存在
    pub async fn ensure_file(&self, file_path: &str) -> Result<(), AppException> {
        let metadata = fs::metadata(file_path)
            .await
            .map_err(|_| AppException::not_found(format!("该文件不存在: {file_path}")))?;

        if !metadata.is_file() {
            return Err(AppException::not_found(format!(
                "该文件不存在: {file_path}"
            )));
        }

        Ok(())
    }

    /// 根据传递的路径判断文件是否存在
    pub async fn check_file_exists(
        &self,
        file_path: &str,
    ) -> Result<FileCheckResult, AppException> {
        let exists = fs::try_exists(file_path)
            .await
            .map_err(|err| AppException::internal(format!("检查文件是否存在失败: {err}")))?;

        Ok(FileCheckResult {
            file_path: file_path.to_string(),
            exists,
        })
    }

    /// 根据传递的路径删除指定文件
    pub async fn delete_file(&self, file_path: &str) -> Result<FileDeleteResult, AppException> {
        // 1.判断文件是否存在
        self.ensure_file(file_path).await?;
        // 2.调用命令删除文件
        fs::remove_file(file_path)
            .await
            .map_err(|err| AppException::internal(format!("删除文件{file_path}失败: {err}")))?;

        Ok(FileDeleteResult {
            file_path: file_path.to_string(),
            deleted: true,
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

    #[tokio::test]
    async fn replaces_every_non_overlapping_occurrence_in_the_complete_file() {
        let file = TestFile::create("Agent开发\nAgent工具\n").await;

        let result = FileService::new()
            .replace_in_file(file.as_str(), "Agent", "Manus", Some(false))
            .await
            .expect("matching content should be replaceable");

        assert_eq!(result.file_path, file.as_str());
        assert_eq!(result.replaced_count, 2);
        assert_eq!(
            fs::read_to_string(&file.0).await.unwrap(),
            "Manus开发\nManus工具\n"
        );
    }

    #[tokio::test]
    async fn search_matches_from_the_start_of_each_line_and_returns_zero_based_indexes() {
        let file = TestFile::create("LLM产品\nAgent开发\n前缀LLM\nLLM应用开发\n").await;

        let result = FileService::new()
            .search_in_file(file.as_str(), "LLM", Some(false))
            .await
            .expect("valid regular expression should be searchable");

        assert_eq!(result.matches, ["LLM产品", "LLM应用开发"]);
        assert_eq!(result.line_numbers, [0, 3]);
    }

    #[tokio::test]
    async fn search_rejects_an_invalid_regular_expression() {
        let file = TestFile::create("Agent开发").await;

        let error = FileService::new()
            .search_in_file(file.as_str(), "[", Some(false))
            .await
            .expect_err("invalid regular expression should be rejected");

        assert_eq!(error.status_code, StatusCode::BAD_REQUEST);
        assert!(error.msg.contains("传递正则表达式[["));
    }

    #[tokio::test]
    async fn finds_files_with_recursive_glob_rules() {
        let directory = TestDirectory::new();
        let root_file = directory.file_path("main.rs");
        let nested_file = directory.file_path("src/lib.rs");
        fs::create_dir_all(nested_file.parent().unwrap())
            .await
            .unwrap();
        fs::write(&root_file, "fn main() {}").await.unwrap();
        fs::write(&nested_file, "pub fn run() {}").await.unwrap();
        fs::write(directory.file_path("src/readme.md"), "docs")
            .await
            .unwrap();

        let result = FileService::new()
            .find_files(directory.0.to_str().unwrap(), "**/*.rs")
            .await
            .expect("recursive glob should find matching files");

        assert_eq!(result.dir_path, directory.0.to_str().unwrap());
        assert_eq!(result.files.len(), 2);
        assert!(
            result
                .files
                .contains(&root_file.to_string_lossy().into_owned())
        );
        assert!(
            result
                .files
                .contains(&nested_file.to_string_lossy().into_owned())
        );
    }

    #[tokio::test]
    async fn finalizes_a_chunked_upload_into_the_requested_directory() {
        let directory = TestDirectory::new();
        let source = directory.file_path("upload.tmp");
        let target = directory.file_path("nested/demo.bin");
        fs::create_dir_all(&directory.0).await.unwrap();
        fs::write(&source, b"chunk-onechunk-two").await.unwrap();

        let result = FileService::new()
            .upload_file(&source, target.to_str().unwrap())
            .await
            .expect("uploaded file should be finalized");

        assert_eq!(result.file_path, target.to_str().unwrap());
        assert_eq!(result.file_size, 18);
        assert!(result.success);
        assert_eq!(fs::read(&target).await.unwrap(), b"chunk-onechunk-two");
        assert!(!fs::try_exists(&source).await.unwrap());
    }

    #[tokio::test]
    async fn checks_and_deletes_an_existing_file() {
        let file = TestFile::create("待删除").await;

        let checked = FileService::new()
            .check_file_exists(file.as_str())
            .await
            .expect("file existence should be checkable");
        let deleted = FileService::new()
            .delete_file(file.as_str())
            .await
            .expect("existing file should be deletable");

        assert!(checked.exists);
        assert!(deleted.deleted);
        assert!(!fs::try_exists(&file.0).await.unwrap());
    }

    #[tokio::test]
    async fn ensure_file_reports_a_missing_path() {
        let path = std::env::temp_dir().join(format!("sandbox-missing-{}", Uuid::new_v4()));

        let error = FileService::new()
            .ensure_file(path.to_str().unwrap())
            .await
            .expect_err("missing file should be rejected");

        assert_eq!(error.status_code, StatusCode::NOT_FOUND);
        assert!(error.msg.contains("该文件不存在"));
    }
}
