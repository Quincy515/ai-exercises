//! MCP 配置：私有 JSON 草稿、批量完整更新与串行写后刷新。

mod validation;

use std::{collections::BTreeMap, fmt};

use crux_core::{Command, render::render};
use crux_http::{HttpError, Response, command::RequestBuilder};
use facet::Facet;
use serde::{Deserialize, Serialize};

use crate::{
    api::configs::mcp::{
        McpConfigAck, McpServer, McpServerAck, McpServerList, create_mcp_servers,
        delete_mcp_server, get_mcp_servers, set_mcp_server_enabled,
    },
    effects::Effect,
};

#[derive(Facet, Serialize, Deserialize, PartialEq)]
#[repr(C)]
pub enum McpConfigEvent {
    Get {
        base_url: String,
    },
    EditJson {
        value: String,
    },
    ResetDraft,
    Create {
        base_url: String,
    },
    SetEnabled {
        base_url: String,
        server_name: String,
        enabled: bool,
    },
    Delete {
        base_url: String,
        server_name: String,
    },
    #[serde(skip)]
    #[facet(skip)]
    Received(#[facet(opaque)] crux_http::Result<Response<McpServerList>>),
    #[serde(skip)]
    #[facet(skip)]
    MutationFinished(#[facet(opaque)] crux_http::Result<Response<McpConfigAck>>),
}

impl fmt::Debug for McpConfigEvent {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        // JSON、响应及错误可能携带密钥，日志始终只记录事件名称。
        formatter.write_str(match self {
            Self::Get { .. } => "McpConfigEvent::Get",
            Self::EditJson { .. } => "McpConfigEvent::EditJson",
            Self::ResetDraft => "McpConfigEvent::ResetDraft",
            Self::Create { .. } => "McpConfigEvent::Create",
            Self::SetEnabled { .. } => "McpConfigEvent::SetEnabled",
            Self::Delete { .. } => "McpConfigEvent::Delete",
            Self::Received(_) => "McpConfigEvent::Received",
            Self::MutationFinished(_) => "McpConfigEvent::MutationFinished",
        })
    }
}

enum Mutation {
    Create {
        expected: BTreeMap<String, McpServerAck>,
    },
    SetEnabled {
        server_name: String,
        enabled: bool,
    },
    Delete {
        server_name: String,
    },
}

impl Mutation {
    fn confirmed_by(&self, ack: &McpConfigAck) -> bool {
        match self {
            Self::Create { expected } => expected
                .iter()
                .all(|(name, config)| ack.mcp_servers.get(name) == Some(config)),
            Self::SetEnabled {
                server_name,
                enabled,
            } => ack
                .mcp_servers
                .get(server_name)
                .is_some_and(|config| config.enabled == *enabled),
            Self::Delete { server_name } => !ack.mcp_servers.contains_key(server_name),
        }
    }
}

struct PendingMutation {
    base_url: String,
    mutation: Mutation,
}

#[derive(Default)]
pub struct McpConfigModel {
    pub(crate) servers: Vec<McpServer>,
    draft_json: String,
    pub(crate) loaded: bool,
    pub(crate) loading: bool,
    pub(crate) saving: bool,
    pub(crate) error: Option<String>,
    pub(crate) notice: Option<String>,
    pub(crate) created: bool,
    pub(crate) write_uncertain: bool,
    pending: Option<PendingMutation>,
}

impl McpConfigModel {
    pub fn update(&mut self, event: McpConfigEvent) -> Command<Effect, McpConfigEvent> {
        match event {
            McpConfigEvent::Get { base_url } => {
                if self.loading || self.saving {
                    return Command::done();
                }
                self.fetch(&base_url)
            }
            McpConfigEvent::EditJson { value } => {
                if self.loading || self.saving {
                    return Command::done();
                }
                self.draft_json = value;
                self.created = false;
                self.error = None;
                render()
            }
            McpConfigEvent::ResetDraft => {
                if self.saving {
                    return Command::done();
                }
                self.draft_json.clear();
                self.error = None;
                self.created = false;
                render()
            }
            McpConfigEvent::Create { base_url } => self.create(&base_url),
            McpConfigEvent::SetEnabled {
                base_url,
                server_name,
                enabled,
            } => {
                if !self.can_mutate()
                    || self
                        .servers
                        .iter()
                        .any(|s| s.server_name == server_name && s.enabled == enabled)
                {
                    return Command::done();
                }
                if !self.has_server(&server_name) {
                    return self.missing_server();
                }
                let request = set_mcp_server_enabled(&base_url, &server_name, enabled);
                self.start_mutation(
                    request,
                    &base_url,
                    Mutation::SetEnabled {
                        server_name,
                        enabled,
                    },
                )
            }
            McpConfigEvent::Delete {
                base_url,
                server_name,
            } => {
                if !self.can_mutate() {
                    return Command::done();
                }
                if !self.has_server(&server_name) {
                    return self.missing_server();
                }
                let request = delete_mcp_server(&base_url, &server_name);
                self.start_mutation(request, &base_url, Mutation::Delete { server_name })
            }
            McpConfigEvent::Received(result) => self.receive(result),
            McpConfigEvent::MutationFinished(result) => self.mutation_finished(result),
        }
    }

    pub(crate) fn draft_present(&self) -> bool {
        !self.draft_json.trim().is_empty()
    }

    fn can_mutate(&self) -> bool {
        self.loaded && !self.loading && !self.saving && !self.write_uncertain
    }

    fn has_server(&self, name: &str) -> bool {
        self.servers.iter().any(|server| server.server_name == name)
    }

    fn missing_server(&mut self) -> Command<Effect, McpConfigEvent> {
        self.error = Some("MCP 服务器已不存在，请刷新列表后重试。".to_string());
        render()
    }

    fn fetch(&mut self, base_url: &str) -> Command<Effect, McpConfigEvent> {
        let request = match get_mcp_servers(base_url) {
            Ok(request) => request,
            Err(error) => {
                self.error = Some(error);
                self.saving = false;
                return render();
            }
        };
        self.loading = true;
        self.error = None;
        render().and(request.build().then_send(McpConfigEvent::Received))
    }

    fn receive(
        &mut self,
        result: crux_http::Result<Response<McpServerList>>,
    ) -> Command<Effect, McpConfigEvent> {
        if !self.loading {
            return Command::done();
        }
        self.loading = false;
        self.saving = false;
        match super::receive_config(result, false, "MCP") {
            Ok(list) => {
                self.servers = list.mcp_servers;
                self.loaded = true;
                if self.write_uncertain {
                    self.notice = Some(
                        "已刷新列表，请核对上次操作结果后继续；同名配置会完整更新。".to_string(),
                    );
                }
                self.write_uncertain = false;
                self.error = None;
            }
            Err(error) => self.error = Some(error),
        }
        render()
    }

    fn create(&mut self, base_url: &str) -> Command<Effect, McpConfigEvent> {
        if !self.can_mutate() {
            return Command::done();
        }
        let (request, expected) = match validation::parse_config(&self.draft_json) {
            Ok(config) => config,
            Err(error) => {
                self.error = Some(error);
                return render();
            }
        };
        self.start_mutation(
            create_mcp_servers(base_url, &request),
            base_url,
            Mutation::Create { expected },
        )
    }

    fn start_mutation(
        &mut self,
        request: Result<RequestBuilder<Effect, McpConfigEvent, McpConfigAck>, String>,
        base_url: &str,
        mutation: Mutation,
    ) -> Command<Effect, McpConfigEvent> {
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
        render().and(request.build().then_send(McpConfigEvent::MutationFinished))
    }

    fn mutation_finished(
        &mut self,
        result: crux_http::Result<Response<McpConfigAck>>,
    ) -> Command<Effect, McpConfigEvent> {
        let Some(pending) = self.pending.take() else {
            return Command::done();
        };
        let uncertain = match &result {
            Err(HttpError::Http { code, .. }) => *code >= 500,
            Err(HttpError::Url(_)) => false,
            _ => true,
        };
        let missing = matches!(&result, Err(HttpError::Http { code: 404, .. }));
        let ack = match super::receive_config(result, true, "MCP") {
            Ok(ack) => ack,
            Err(error) => {
                return self.mutation_failed(
                    if missing {
                        "MCP 服务器已不存在，请刷新列表后重试。".to_string()
                    } else {
                        error
                    },
                    uncertain,
                );
            }
        };
        if !pending.mutation.confirmed_by(&ack) {
            return self.mutation_failed(
                "配置写入响应与本次操作不一致，请刷新列表核对。".to_string(),
                true,
            );
        }
        self.apply_confirmation(ack, &pending.mutation);
        self.notice = Some(match pending.mutation {
            Mutation::Create { .. } => {
                self.draft_json.clear();
                self.created = true;
                "已保存 MCP 服务器配置。".to_string()
            }
            Mutation::SetEnabled { .. } => "已更新 MCP 服务器启用状态。".to_string(),
            Mutation::Delete { .. } => "已删除 MCP 服务器配置。".to_string(),
        });
        self.fetch(&pending.base_url)
    }

    fn mutation_failed(
        &mut self,
        error: String,
        uncertain: bool,
    ) -> Command<Effect, McpConfigEvent> {
        self.saving = false;
        self.error = Some(error);
        self.write_uncertain = uncertain;
        if uncertain {
            self.notice =
                Some("操作结果尚未确认，请刷新列表核对后再操作，避免覆盖已有配置。".to_string());
        }
        render()
    }

    fn apply_confirmation(&mut self, ack: McpConfigAck, mutation: &Mutation) {
        self.servers = ack
            .mcp_servers
            .into_iter()
            .map(|(server_name, config)| {
                let tools = self
                    .servers
                    .iter()
                    .find(|server| {
                        server.server_name == server_name
                            && config.enabled
                            && server.transport == config.transport
                            && !matches!(mutation, Mutation::Create { expected } if expected.contains_key(&server_name))
                    })
                    .map(|server| server.tools.clone())
                    .unwrap_or_default();
                McpServer {
                    server_name,
                    enabled: config.enabled,
                    transport: config.transport,
                    tools,
                }
            })
            .collect();
    }
}

#[cfg(test)]
mod tests;
