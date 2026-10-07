//! 页面需要的数据：从内部 Model 转换为两端共享的展示结构。

pub mod configs;

use facet::Facet;
use serde::{Deserialize, Serialize};

use crate::model::Model;
pub use configs::{AgentConfigViewModel, LlmConfigViewModel};

#[derive(Facet, Serialize, Deserialize, Debug, Clone)]
pub struct ViewModel {
    pub text: String,
    pub confirmed: bool,
    pub agent_config: AgentConfigViewModel,
    pub llm_config: LlmConfigViewModel,
}

impl From<&Model> for ViewModel {
    fn from(model: &Model) -> Self {
        let suffix = model.count.updated_at.map_or_else(
            || " (pending)".to_string(),
            |updated_at| format!(" ({updated_at})"),
        );

        Self {
            text: model.count.value.to_string() + &suffix,
            confirmed: model.count.updated_at.is_some(),
            agent_config: (&model.configs.agent).into(),
            llm_config: (&model.configs.llm).into(),
        }
    }
}
