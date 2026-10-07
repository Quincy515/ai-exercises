//! 模型提供商的安全展示数据；密钥草稿始终留在内部 Model。

use facet::Facet;
use serde::{Deserialize, Serialize};

use crate::{
    api::configs::llm::LlmConfig,
    model::configs::llm::{LlmConfigDraft, LlmConfigModel},
};

#[derive(Facet, Serialize, Deserialize, Debug, Clone, Default, PartialEq)]
pub struct LlmConfigViewModel {
    pub data: Option<LlmConfig>,
    pub draft: LlmConfigDraft,
    pub loading: bool,
    pub saving: bool,
    pub error: Option<String>,
    pub saved: bool,
    pub dirty: bool,
    pub can_save: bool,
    pub api_key_changed: bool,
}

impl From<&LlmConfigModel> for LlmConfigViewModel {
    fn from(model: &LlmConfigModel) -> Self {
        Self {
            data: model.data.clone(),
            draft: model.draft.clone(),
            loading: model.loading,
            saving: model.saving,
            error: model.error.clone(),
            saved: model.saved,
            dirty: model.dirty(),
            can_save: model.can_save(),
            api_key_changed: model.api_key_changed(),
        }
    }
}
