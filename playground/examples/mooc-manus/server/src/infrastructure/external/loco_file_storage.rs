use std::{path::Path, sync::Arc};

use anyhow::{ensure, Context, Result};
use async_trait::async_trait;
use chrono::Local;
use loco_rs::storage::Storage;

use crate::domain::{
    external::{FileNotFound, FileStorage, FileStream, InvalidFileUpload, UploadFile},
    models::File,
    repositories::FileRepository,
};

/// 基于 Loco Storage 的文件存储桶，复用应用配置好的本地、内存或 R2 驱动。
pub struct LocoFileStorage {
    storage: Arc<Storage>,
    file_repository: Arc<dyn FileRepository>,
}

impl LocoFileStorage {
    /// 构造函数，完成文件存储桶初始化。
    pub fn new(storage: Arc<Storage>, file_repository: Arc<dyn FileRepository>) -> Self {
        Self {
            storage,
            file_repository,
        }
    }
}

#[async_trait]
impl FileStorage for LocoFileStorage {
    /// 根据传递的文件源上传文件后返回文件信息。
    async fn upload_file(&self, upload_file: UploadFile) -> Result<File> {
        // 1.生成随机的 UUID 作为文件 id 并获取文件扩展名。
        // 大小以实际内容为准；写入对象存储前先检查数据库字段范围。
        // 同时兼容浏览器提供的 Unix/Windows 路径，只保留最后的文件名。
        let filename = upload_file
            .filename
            .rsplit(['/', '\\'])
            .next()
            .unwrap_or_default();
        ensure!(
            !filename.trim().is_empty()
                && filename != "."
                && filename != ".."
                && !filename.chars().any(char::is_control),
            InvalidFileUpload("文件名无效")
        );
        ensure!(
            filename.chars().count() <= 255,
            InvalidFileUpload("文件名不能超过 255 个字符")
        );
        let size = upload_file.content.len();
        ensure!(
            i32::try_from(size).is_ok(),
            InvalidFileUpload("文件大小超出数据库 size 的 i32 范围")
        );
        let mime_type = upload_file
            .mime_type
            .as_deref()
            .map(str::trim)
            .filter(|value| !value.is_empty())
            .unwrap_or_default();
        ensure!(
            mime_type.chars().count() <= 255 && !mime_type.chars().any(char::is_control),
            InvalidFileUpload("文件 MIME 类型无效")
        );
        // 扩展名保留前导点；.env 这类只有前导点的文件不算有扩展名。
        let extension = filename
            .rsplit_once('.')
            .filter(|(stem, _)| stem.chars().any(|character| character != '.'))
            .map(|(_, extension)| format!(".{extension}"))
            .unwrap_or_default();
        let mut file = File {
            filename: filename.to_owned(),
            extension,
            mime_type: mime_type.to_owned(),
            size,
            ..File::default()
        };

        // 2.生成日期路径并拼接最终 key，UUID 避免同名文件覆盖和用户路径穿越。
        // filepath 留空，后续文件进入沙箱时才由对应流程设置。
        file.key = format!(
            "{}/{}{}",
            Local::now().format("%Y/%m/%d"),
            file.id,
            file.extension
        );
        ensure!(
            file.key.chars().count() <= 255,
            InvalidFileUpload("文件扩展名过长，生成的对象 key 不能超过 255 个字符")
        );
        let path = Path::new(&file.key);

        // 3.使用 Loco 原生异步 API 上传文件，不占用阻塞线程池。
        self.storage
            .upload(path, &upload_file.content)
            .await
            .context("上传文件内容失败")?;

        // 4.构建文件模型后将数据存储到数据库中。
        // 数据库与对象存储不能共用事务，保存失败时补偿删除对象。
        if let Err(error) = self.file_repository.save(file.clone()).await {
            if let Err(cleanup_error) = self.storage.delete(path).await {
                // 仅记录对象标识，避免驱动错误中的连接配置进入日志；原始错误保留在返回值。
                tracing::error!(key = %file.key, "文件信息保存失败，清理已上传对象也失败");
                return Err(error.context(format!(
                    "保存文件信息失败，清理对象[{}]也失败: {cleanup_error}",
                    file.key
                )));
            }
            return Err(error.context("保存文件信息失败，已清理上传对象"));
        }
        tracing::info!(filename = %file.filename, file_id = %file.id, "文件上传成功");
        Ok(file)
    }

    /// 根据传递的文件 id 下载文件，并返回文件源 + 文件信息。
    async fn download_file(&self, file_id: &str) -> Result<(FileStream, File)> {
        // 1.查询对应的文件记录是否存在，由仓库完成业务 UUID 到数据库记录的映射。
        let file = self
            .file_repository
            .get_by_id(file_id)
            .await
            .context("获取待下载文件信息失败")?
            .ok_or_else(|| FileNotFound::new(file_id))?;

        // 2.使用 Loco 原生异步 API 下载；读取记录中的 key，兼容之前上传的文件。
        // 不能把沙箱 filepath 当作存储路径。
        ensure!(!file.key.is_empty(), "文件[{file_id}]缺少对象存储 key");
        let stream = self
            .storage
            .download_stream(Path::new(&file.key))
            .await
            .with_context(|| format!("下载文件[{file_id}]失败"))?;

        // 3.返回异步文件源和领域模型；接口层可用 Body::from_stream 构造响应。
        Ok((Box::pin(stream), file))
    }
}
