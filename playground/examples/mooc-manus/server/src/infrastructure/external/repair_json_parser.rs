use anyhow::Result;
use async_trait::async_trait;
use llm_json::{loads, RepairOptions};
use serde_json::Value;
use tracing::info;

use crate::domain::external::JsonParser;

/// 基于修复逻辑的 JSON 解析器。
/// JSON parser backed by repair logic.
pub struct RepairJsonParser;

#[async_trait]
impl JsonParser for RepairJsonParser {
    /// 传递文本，并使用 JSON 修复库进行修复和解析。
    /// Repair and parse the provided JSON text.
    async fn invoke(&self, text: &str, default_value: Option<Value>) -> Result<Value> {
        info!("解析 JSON 文本: {}", text);

        if text.trim().is_empty() {
            return default_value.ok_or_else(|| anyhow::anyhow!("JSON 文本为空，且无默认值"));
        }

        let options = RepairOptions {
            ensure_ascii: false,
            ..RepairOptions::default()
        };

        Ok(loads(text, &options)?)
    }
}
