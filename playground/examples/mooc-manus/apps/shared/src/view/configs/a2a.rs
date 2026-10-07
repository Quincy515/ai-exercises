//! A2A 列表与操作状态，两端直接消费同一视图。

use facet::Facet;
use serde::{Deserialize, Serialize};

use crate::{api::configs::a2a::A2aServer, model::configs::a2a::A2aConfigModel};

#[derive(Facet, Serialize, Deserialize, Debug, Clone, Default, PartialEq, Eq)]
pub struct A2aConfigViewModel {
    pub servers: Vec<A2aServer>,
    pub draft_url: String,
    pub loaded: bool,
    pub loading: bool,
    pub saving: bool,
    pub error: Option<String>,
    pub notice: Option<String>,
    pub created: bool,
    pub write_uncertain: bool,
}

impl From<&A2aConfigModel> for A2aConfigViewModel {
    fn from(model: &A2aConfigModel) -> Self {
        Self {
            servers: model.servers.clone(),
            draft_url: model.draft_url.clone(),
            loaded: model.loaded,
            loading: model.loading,
            saving: model.saving,
            error: model.error.clone(),
            notice: model.notice.clone(),
            created: model.created,
            write_uncertain: model.write_uncertain,
        }
    }
}
