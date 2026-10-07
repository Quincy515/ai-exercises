//! 远程 Agent 配置：读取、新增、启停和删除；写成功后统一刷新列表。

use std::fmt;

use crux_core::{Command, render::render};
use crux_http::{HttpError, Response, Url, command::RequestBuilder};
use facet::Facet;
use serde::{Deserialize, Serialize};

use crate::{
    api::configs::a2a::{
        A2aServer, A2aServerList, create_a2a_server, delete_a2a_server, get_a2a_servers,
        set_a2a_server_enabled,
    },
    effects::Effect,
};

#[derive(Facet, Serialize, Deserialize, PartialEq)]
#[repr(C)]
pub enum A2aConfigEvent {
    Get {
        base_url: String,
    },
    EditUrl {
        value: String,
    },
    ResetDraft,
    Create {
        base_url: String,
    },
    SetEnabled {
        base_url: String,
        id: String,
        enabled: bool,
    },
    Delete {
        base_url: String,
        id: String,
    },
    #[serde(skip)]
    #[facet(skip)]
    Received(#[facet(opaque)] crux_http::Result<Response<A2aServerList>>),
    #[serde(skip)]
    #[facet(skip)]
    MutationFinished(#[facet(opaque)] crux_http::Result<Response<()>>),
}

impl fmt::Debug for A2aConfigEvent {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        // 输入 URL、远程卡片和传输错误保持在业务状态中，日志只记录事件名称。
        formatter.write_str(match self {
            Self::Get { .. } => "A2aConfigEvent::Get",
            Self::EditUrl { .. } => "A2aConfigEvent::EditUrl",
            Self::ResetDraft => "A2aConfigEvent::ResetDraft",
            Self::Create { .. } => "A2aConfigEvent::Create",
            Self::SetEnabled { .. } => "A2aConfigEvent::SetEnabled",
            Self::Delete { .. } => "A2aConfigEvent::Delete",
            Self::Received(_) => "A2aConfigEvent::Received",
            Self::MutationFinished(_) => "A2aConfigEvent::MutationFinished",
        })
    }
}

enum Mutation {
    Create,
    SetEnabled { id: String, enabled: bool },
    Delete { id: String },
}

struct PendingMutation {
    base_url: String,
    mutation: Mutation,
}

#[derive(Default)]
pub struct A2aConfigModel {
    pub(crate) servers: Vec<A2aServer>,
    pub(crate) draft_url: String,
    pub(crate) loaded: bool,
    pub(crate) loading: bool,
    pub(crate) saving: bool,
    pub(crate) error: Option<String>,
    pub(crate) notice: Option<String>,
    pub(crate) created: bool,
    pub(crate) write_uncertain: bool,
    pending: Option<PendingMutation>,
}

impl A2aConfigModel {
    pub fn update(&mut self, event: A2aConfigEvent) -> Command<Effect, A2aConfigEvent> {
        match event {
            A2aConfigEvent::Get { base_url } => {
                if self.loading || self.saving {
                    return Command::done();
                }
                self.fetch(&base_url)
            }
            A2aConfigEvent::EditUrl { value } => {
                if self.loading || self.saving {
                    return Command::done();
                }
                self.draft_url = value;
                self.created = false;
                self.error = None;
                render()
            }
            A2aConfigEvent::ResetDraft => {
                if self.loading || self.saving {
                    return Command::done();
                }
                self.draft_url.clear();
                self.error = None;
                self.created = false;
                render()
            }
            A2aConfigEvent::Create { base_url } => self.create(&base_url),
            A2aConfigEvent::SetEnabled {
                base_url,
                id,
                enabled,
            } => {
                if !self.can_mutate()
                    || self
                        .servers
                        .iter()
                        .any(|s| s.id == id && s.enabled == enabled)
                {
                    return Command::done();
                }
                if !self.has_server(&id) {
                    return self.missing_server();
                }
                let request = set_a2a_server_enabled(&base_url, &id, enabled);
                self.start_mutation(request, &base_url, Mutation::SetEnabled { id, enabled })
            }
            A2aConfigEvent::Delete { base_url, id } => {
                if !self.can_mutate() {
                    return Command::done();
                }
                if !self.has_server(&id) {
                    return self.missing_server();
                }
                let request = delete_a2a_server(&base_url, &id);
                self.start_mutation(request, &base_url, Mutation::Delete { id })
            }
            A2aConfigEvent::Received(result) => self.receive(result),
            A2aConfigEvent::MutationFinished(result) => self.mutation_finished(result),
        }
    }

    fn can_mutate(&self) -> bool {
        self.loaded && !self.loading && !self.saving && !self.write_uncertain
    }

    fn has_server(&self, id: &str) -> bool {
        self.servers.iter().any(|server| server.id == id)
    }

    fn missing_server(&mut self) -> Command<Effect, A2aConfigEvent> {
        self.error = Some("远程 Agent 已不存在，请刷新列表后重试。".to_string());
        render()
    }

    fn fetch(&mut self, base_url: &str) -> Command<Effect, A2aConfigEvent> {
        let request = match get_a2a_servers(base_url) {
            Ok(request) => request,
            Err(error) => {
                self.error = Some(error);
                self.saving = false;
                return render();
            }
        };
        self.loading = true;
        self.error = None;
        render().and(request.build().then_send(A2aConfigEvent::Received))
    }

    fn receive(
        &mut self,
        result: crux_http::Result<Response<A2aServerList>>,
    ) -> Command<Effect, A2aConfigEvent> {
        if !self.loading {
            return Command::done();
        }
        self.loading = false;
        self.saving = false;
        match super::receive_config(result, false, "A2A Agent") {
            Ok(list) => {
                self.servers = list.a2a_servers;
                self.loaded = true;
                if self.write_uncertain {
                    self.notice = Some(
                        "已刷新列表，请核对上次操作结果后继续；再次添加可能重复。".to_string(),
                    );
                }
                self.write_uncertain = false;
                self.error = None;
            }
            Err(error) => self.error = Some(error),
        }
        render()
    }

    fn create(&mut self, base_url: &str) -> Command<Effect, A2aConfigEvent> {
        if !self.can_mutate() {
            return Command::done();
        }
        let request =
            validate_remote_url(&self.draft_url).and_then(|url| create_a2a_server(base_url, &url));
        self.start_mutation(request, base_url, Mutation::Create)
    }

    fn start_mutation(
        &mut self,
        request: Result<RequestBuilder<Effect, A2aConfigEvent, ()>, String>,
        base_url: &str,
        mutation: Mutation,
    ) -> Command<Effect, A2aConfigEvent> {
        let request = match request {
            Ok(request) => request,
            Err(error) => {
                self.error = Some(error);
                return render();
            }
        };
        self.pending = Some(PendingMutation {
            base_url: base_url.to_string(),
            mutation,
        });
        self.saving = true;
        self.created = false;
        self.error = None;
        self.notice = None;
        render().and(request.build().then_send(A2aConfigEvent::MutationFinished))
    }

    fn mutation_finished(
        &mut self,
        result: crux_http::Result<Response<()>>,
    ) -> Command<Effect, A2aConfigEvent> {
        let Some(pending) = self.pending.take() else {
            return Command::done();
        };
        let uncertain = match &result {
            Err(HttpError::Http { code, .. }) => *code >= 500,
            Err(HttpError::Url(_)) => false,
            _ => true,
        };
        let missing = matches!(&result, Err(HttpError::Http { code: 404, .. }));
        if let Err(error) = super::receive_config(result, true, "A2A Agent") {
            self.saving = false;
            self.write_uncertain = uncertain;
            self.error = Some(if missing {
                "远程 Agent 已不存在，请刷新列表后重试。".to_string()
            } else {
                error
            });
            if uncertain {
                self.notice =
                    Some("操作结果尚未确认，请刷新列表核对后再操作，避免重复添加。".to_string());
            }
            return render();
        }
        self.notice = Some(match pending.mutation {
            Mutation::Create => {
                self.draft_url.clear();
                self.created = true;
                "已添加远程Agent配置。".to_string()
            }
            Mutation::SetEnabled { id, enabled } => {
                if let Some(server) = self.servers.iter_mut().find(|server| server.id == id) {
                    server.enabled = enabled;
                }
                "已更新远程 Agent 启用状态。".to_string()
            }
            Mutation::Delete { id } => {
                self.servers.retain(|server| server.id != id);
                "已删除远程 Agent 配置。".to_string()
            }
        });
        self.fetch(&pending.base_url)
    }
}

fn validate_remote_url(value: &str) -> Result<String, String> {
    let value = value.trim();
    let url = Url::parse(value)
        .ok()
        .filter(|url| {
            !value.chars().any(char::is_control)
                && matches!(url.scheme(), "http" | "https")
                && url.host_str().is_some()
                && url.username().is_empty()
                && url.password().is_none()
                && url.query().is_none()
                && url.fragment().is_none()
        })
        .ok_or_else(|| {
            "远程 Agent 地址须为 HTTP 或 HTTPS 网址，且不能包含用户名、密码、查询参数或片段。"
                .to_string()
        })?;
    Ok(url.as_str().trim_end_matches('/').to_string())
}

#[cfg(test)]
mod tests;
