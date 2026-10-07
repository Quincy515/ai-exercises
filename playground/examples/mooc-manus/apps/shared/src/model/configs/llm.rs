//! 模型提供商状态：编辑草稿、校验、保存与手动重试。

use std::fmt;

use crux_core::{Command, render::render};
use crux_http::{Response, Url};
use facet::Facet;
use serde::{Deserialize, Serialize};

use crate::{
    api::configs::llm::{LlmConfig, LlmConfigRequest, get_llm_config, update_llm_config},
    effects::Effect,
};

#[derive(Facet, Serialize, Deserialize, PartialEq)]
#[repr(C)]
pub enum LlmConfigEvent {
    Get {
        base_url: String,
    },
    Edit {
        field: LlmConfigField,
        value: String,
    },
    Reset,
    Save {
        base_url: String,
    },

    #[serde(skip)]
    #[facet(skip)]
    Received(#[facet(opaque)] crux_http::Result<Response<LlmConfig>>),
    #[serde(skip)]
    #[facet(skip)]
    Saved(#[facet(opaque)] crux_http::Result<Response<LlmConfig>>),
}

impl fmt::Debug for LlmConfigEvent {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        // 输入、服务响应与传输错误都可能包含密钥，日志只记录事件名称。
        formatter.write_str(match self {
            Self::Get { .. } => "LlmConfigEvent::Get",
            Self::Edit { .. } => "LlmConfigEvent::Edit",
            Self::Reset => "LlmConfigEvent::Reset",
            Self::Save { .. } => "LlmConfigEvent::Save",
            Self::Received(_) => "LlmConfigEvent::Received",
            Self::Saved(_) => "LlmConfigEvent::Saved",
        })
    }
}

#[derive(Facet, Serialize, Deserialize, Debug, Clone, Copy, PartialEq, Eq)]
#[repr(C)]
pub enum LlmConfigField {
    BaseUrl,
    ApiKey,
    ModelName,
    Temperature,
    MaxTokens,
}

/// 可回显的字符串草稿保留清空和输入中的状态；密钥单独保存在内部 Model。
#[derive(Facet, Serialize, Deserialize, Debug, Clone, Default, PartialEq, Eq)]
pub struct LlmConfigDraft {
    pub base_url: String,
    pub model_name: String,
    pub temperature: String,
    pub max_tokens: String,
}

impl From<&LlmConfig> for LlmConfigDraft {
    fn from(config: &LlmConfig) -> Self {
        Self {
            base_url: config.base_url.clone().unwrap_or_default(),
            model_name: config.model_name.clone().unwrap_or_default(),
            temperature: config
                .temperature
                .map(|value| value.to_string())
                .unwrap_or_default(),
            max_tokens: config
                .max_tokens
                .map(|value| value.to_string())
                .unwrap_or_default(),
        }
    }
}

impl LlmConfigDraft {
    fn validate(&self, api_key: &str) -> Result<LlmConfigRequest, String> {
        let base_url = optional_text(&self.base_url);
        if let Some(base_url) = &base_url {
            let valid = Url::parse(base_url).ok().is_some_and(|url| {
                matches!(url.scheme(), "http" | "https")
                    && url.host_str().is_some()
                    && url.username().is_empty()
                    && url.password().is_none()
            });
            if !valid {
                return Err(
                    "提供商地址须为 HTTP 或 HTTPS 网址，且不能包含用户名或密码。".to_string(),
                );
            }
        }
        let temperature = optional_text(&self.temperature)
            .map(|value| {
                value
                    .parse::<f32>()
                    .ok()
                    .filter(|value| value.is_finite() && (-2.0..=2.0).contains(value))
                    .ok_or_else(|| "温度必须是 -2 到 2 之间的有限数值，或留空。".to_string())
            })
            .transpose()?;
        let max_tokens = optional_text(&self.max_tokens)
            .map(|value| {
                value
                    .parse::<u64>()
                    .ok()
                    // 与后端 llm_configs 的数据库 i64 范围保持一致。
                    .filter(|value| i64::try_from(*value).is_ok())
                    .ok_or_else(|| {
                        "最大输出 Tokens 必须是 0–9223372036854775807 的整数，或留空。".to_string()
                    })
            })
            .transpose()?;
        Ok(LlmConfigRequest {
            base_url,
            api_key: optional_text(api_key),
            model_name: optional_text(&self.model_name),
            temperature,
            max_tokens,
        })
    }
}

fn optional_text(value: &str) -> Option<String> {
    let value = value.trim();
    (!value.is_empty()).then(|| value.to_string())
}

#[derive(Default)]
pub struct LlmConfigModel {
    pub(crate) data: Option<LlmConfig>,
    pub(crate) draft: LlmConfigDraft,
    api_key: String,
    pub(crate) loading: bool,
    pub(crate) saving: bool,
    pub(crate) error: Option<String>,
    pub(crate) saved: bool,
}

impl LlmConfigModel {
    pub fn update(&mut self, event: LlmConfigEvent) -> Command<Effect, LlmConfigEvent> {
        match event {
            LlmConfigEvent::Get { base_url } => self.fetch(&base_url),
            LlmConfigEvent::Edit { field, value } => self.edit(field, value),
            LlmConfigEvent::Reset => self.reset(),
            LlmConfigEvent::Save { base_url } => self.save(&base_url),
            LlmConfigEvent::Received(result) => {
                self.loading = false;
                self.receive(result, false)
            }
            LlmConfigEvent::Saved(result) => {
                self.saving = false;
                self.receive(result, true)
            }
        }
    }

    pub(crate) fn api_key_changed(&self) -> bool {
        !self.api_key.trim().is_empty()
    }

    pub(crate) fn dirty(&self) -> bool {
        self.data.as_ref().is_some_and(|config| {
            self.draft != LlmConfigDraft::from(config) || self.api_key_changed()
        })
    }

    pub(crate) fn can_save(&self) -> bool {
        self.dirty() && !self.loading && !self.saving
    }

    fn fetch(&mut self, base_url: &str) -> Command<Effect, LlmConfigEvent> {
        if self.loading || self.saving || self.dirty() {
            return Command::done();
        }
        let request = match get_llm_config(base_url) {
            Ok(request) => request,
            Err(error) => {
                self.error = Some(error);
                return render();
            }
        };
        self.loading = true;
        self.error = None;
        self.saved = false;
        render().and(request.build().then_send(LlmConfigEvent::Received))
    }

    fn edit(&mut self, field: LlmConfigField, value: String) -> Command<Effect, LlmConfigEvent> {
        if self.data.is_none() || self.loading || self.saving {
            return Command::done();
        }
        match field {
            LlmConfigField::BaseUrl => self.draft.base_url = value,
            LlmConfigField::ApiKey => self.api_key = value,
            LlmConfigField::ModelName => self.draft.model_name = value,
            LlmConfigField::Temperature => self.draft.temperature = value,
            LlmConfigField::MaxTokens => self.draft.max_tokens = value,
        }
        self.error = None;
        self.saved = false;
        render()
    }

    fn reset(&mut self) -> Command<Effect, LlmConfigEvent> {
        if self.loading || self.saving {
            return Command::done();
        }
        self.draft = self
            .data
            .as_ref()
            .map(LlmConfigDraft::from)
            .unwrap_or_default();
        self.api_key.clear();
        self.error = None;
        self.saved = false;
        render()
    }

    fn save(&mut self, base_url: &str) -> Command<Effect, LlmConfigEvent> {
        if !self.can_save() {
            return Command::done();
        }
        let request = match self
            .draft
            .validate(&self.api_key)
            .and_then(|config| update_llm_config(base_url, &config))
        {
            Ok(request) => request,
            Err(error) => {
                self.error = Some(error);
                self.saved = false;
                return render();
            }
        };
        self.saving = true;
        self.error = None;
        self.saved = false;
        render().and(request.build().then_send(LlmConfigEvent::Saved))
    }

    fn receive(
        &mut self,
        result: crux_http::Result<Response<LlmConfig>>,
        saved: bool,
    ) -> Command<Effect, LlmConfigEvent> {
        match super::receive_config(result, saved, "模型提供商") {
            Ok(config) => {
                self.draft = LlmConfigDraft::from(&config);
                self.data = Some(config);
                self.api_key.clear();
                self.error = None;
                self.saved = saved;
            }
            Err(error) => self.error = Some(error),
        }
        render()
    }
}

#[cfg(test)]
mod tests;
