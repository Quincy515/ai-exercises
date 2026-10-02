//! 使用本机随机端口验证 VNC 双向转发；生产路由错误分支使用私有 PostgreSQL。

use crate::test_database as file_database;

use std::{future::Future, net::SocketAddr, time::Duration};

use anyhow::{bail, ensure, Context, Result};
use axum::{extract::State, routing::get, Router};
use futures::{SinkExt, StreamExt};
use loco_rs::{app::Hooks, config::Config, environment::Environment};
use migration::{Migrator, MigratorTrait, SchemaManager};
use serde_json::json;
use tokio::{
    io::{AsyncRead, AsyncReadExt, AsyncWrite},
    net::{TcpListener, TcpStream},
    task::JoinHandle,
    time::timeout,
};
use tokio_tungstenite::{
    accept_hdr_async, connect_async,
    tungstenite::{
        client::IntoClientRequest,
        handshake::server::{Request, Response},
        protocol::{frame::coding::CloseCode, CloseFrame},
        Message,
    },
    MaybeTlsStream, WebSocketStream,
};

use super::{close_vnc_with_error, forward_vnc, WebSocketUpgrade};
use crate::{
    app::App,
    application::shutdown::ShutdownSignal,
    domain::{models::Session, repositories::SessionRepository},
    infrastructure::repositories::SeaOrmSessionRepository,
};

const LIMIT: Duration = Duration::from_secs(5);
type Client = WebSocketStream<MaybeTlsStream<TcpStream>>;
type Upstream = WebSocketStream<TcpStream>;

struct RunningServer {
    address: SocketAddr,
    task: JoinHandle<std::io::Result<()>>,
}

impl RunningServer {
    async fn start(router: Router) -> Result<Self> {
        let listener = TcpListener::bind("127.0.0.1:0").await?;
        let address = listener.local_addr()?;
        let task = tokio::spawn(async move { axum::serve(listener, router).await });
        Ok(Self { address, task })
    }

    fn url(&self, path: &str) -> String {
        format!("ws://{}{path}", self.address)
    }
}

impl Drop for RunningServer {
    fn drop(&mut self) {
        self.task.abort();
    }
}

async fn bounded<T>(future: impl Future<Output = T>) -> Result<T> {
    timeout(LIMIT, future).await.context("WebSocket 操作超时")
}

async fn next_message<S>(socket: &mut WebSocketStream<S>) -> Result<Message>
where
    S: AsyncRead + AsyncWrite + Unpin,
{
    bounded(socket.next())
        .await?
        .context("连接在预期消息前已结束")?
        .context("读取 WebSocket 消息失败")
}

async fn binary_message<S>(socket: &mut WebSocketStream<S>) -> Result<Vec<u8>>
where
    S: AsyncRead + AsyncWrite + Unpin,
{
    // 协议层可以自动回复 Pong；它应留在自己的连接上，不进入 RFB 二进制数据。
    for _ in 0..4 {
        match next_message(socket).await? {
            Message::Binary(data) => return Ok(data.to_vec()),
            Message::Pong(_) => {}
            other => bail!("预期二进制消息或本连接的 Pong，实际为 {other:?}"),
        }
    }
    bail!("没有收到二进制消息")
}

async fn close_frame<S>(socket: &mut WebSocketStream<S>) -> Result<Option<CloseFrame>>
where
    S: AsyncRead + AsyncWrite + Unpin,
{
    match next_message(socket).await? {
        Message::Close(frame) => Ok(frame),
        other => bail!("预期关闭帧，实际为 {other:?}"),
    }
}

fn proxy_router(upstream_url: String, shutdown: ShutdownSignal) -> Router {
    Router::new()
        .route(
            "/vnc",
            get(
                |State((url, shutdown)): State<(String, ShutdownSignal)>,
                 ws: WebSocketUpgrade| async move {
                    ws.on_upgrade(move |mut socket| async move {
                        if let Err(error) = forward_vnc(&mut socket, &url, &shutdown).await {
                            close_vnc_with_error(&mut socket, error.to_string()).await;
                        }
                    })
                },
            ),
        )
        .with_state((upstream_url, shutdown))
}

async fn connected_proxy() -> Result<(RunningServer, Client, Upstream, ShutdownSignal)> {
    let listener = TcpListener::bind("127.0.0.1:0").await?;
    let upstream_url = format!("ws://{}", listener.local_addr()?);
    let shutdown = ShutdownSignal::default();
    let server = RunningServer::start(proxy_router(upstream_url, shutdown.clone())).await?;
    // 握手同时推进，两端均使用本测试创建的随机端口。
    let (client, upstream) = tokio::join!(
        bounded(connect_async(server.url("/vnc"))),
        bounded(async {
            let (stream, _) = listener.accept().await?;
            // Tungstenite 回调固定返回 HTTP Response 错误类型，此处保持库的签名。
            #[allow(clippy::result_large_err)]
            let verify_protocol = |request: &Request, response: Response| {
                // 课程仅协商浏览器侧子协议，连接沙箱时沿用无子协议的握手。
                assert!(!request.headers().contains_key("sec-websocket-protocol"));
                Ok(response)
            };
            accept_hdr_async(stream, verify_protocol)
                .await
                .context("模拟沙箱握手失败")
        })
    );
    Ok((server, client??.0, upstream??, shutdown))
}

#[tokio::test]
async fn forwards_binary_frames_both_ways_without_changing_bytes_or_boundaries() -> Result<()> {
    let (_server, mut client, mut upstream, _shutdown) = connected_proxy().await?;
    let requests = vec![vec![0, 255, 128, 1], Vec::new(), b"RFB request".to_vec()];
    let replies = vec![b"RFB 003.008\n".to_vec(), vec![254, 0, 129], Vec::new()];

    let (sent_requests, sent_replies) = tokio::join!(
        async {
            for data in &requests {
                bounded(client.send(Message::Binary(data.clone().into()))).await??;
            }
            Result::<()>::Ok(())
        },
        async {
            for data in &replies {
                bounded(upstream.send(Message::Binary(data.clone().into()))).await??;
            }
            Result::<()>::Ok(())
        }
    );
    sent_requests?;
    sent_replies?;
    for (request, reply) in requests.iter().zip(&replies) {
        let (received_request, received_reply) =
            tokio::join!(binary_message(&mut upstream), binary_message(&mut client));
        assert_eq!(&received_request?, request);
        assert_eq!(&received_reply?, reply);
    }
    bounded(client.close(None)).await??;
    close_frame(&mut upstream).await?;
    Ok(())
}

#[tokio::test]
async fn client_close_releases_the_upstream_connection() -> Result<()> {
    let (_server, mut client, mut upstream, _shutdown) = connected_proxy().await?;
    bounded(client.close(None)).await??;
    close_frame(&mut upstream).await?;
    close_frame(&mut client).await?;
    Ok(())
}

#[tokio::test]
async fn upstream_close_releases_the_client_connection() -> Result<()> {
    let (_server, mut client, mut upstream, _shutdown) = connected_proxy().await?;
    bounded(upstream.close(None)).await??;
    close_frame(&mut client).await?;
    close_frame(&mut upstream).await?;
    Ok(())
}

#[tokio::test]
async fn control_frames_stay_on_their_connection_and_text_ends_forwarding() -> Result<()> {
    let (_server, mut client, mut upstream, _shutdown) = connected_proxy().await?;
    bounded(client.send(Message::Ping(vec![1].into()))).await??;
    bounded(upstream.send(Message::Ping(vec![2].into()))).await??;
    // 分别确认各自的 Pong，避免将尚未消费的控制帧当成后续关闭帧。
    assert_eq!(
        next_message(&mut client).await?,
        Message::Pong(vec![1].into())
    );
    assert_eq!(
        next_message(&mut upstream).await?,
        Message::Pong(vec![2].into())
    );
    bounded(client.send(Message::Binary(vec![3].into()))).await??;
    bounded(upstream.send(Message::Binary(vec![4].into()))).await??;
    assert_eq!(binary_message(&mut upstream).await?, vec![3]);
    assert_eq!(binary_message(&mut client).await?, vec![4]);
    bounded(client.send(Message::Text("只接受二进制数据".into()))).await??;
    close_frame(&mut upstream).await?;
    close_frame(&mut client).await?;

    // 同样覆盖沙箱返回文本的情况，避免将它误当作 RFB 字节。
    let (_server, mut client, mut upstream, _shutdown) = connected_proxy().await?;
    bounded(upstream.send(Message::Text("invalid RFB payload".into()))).await??;
    close_frame(&mut client).await?;
    close_frame(&mut upstream).await?;
    Ok(())
}

#[tokio::test]
async fn application_shutdown_closes_both_connected_sockets() -> Result<()> {
    let (_server, mut client, mut upstream, shutdown) = connected_proxy().await?;
    bounded(client.send(Message::Binary(b"RFB request".to_vec().into()))).await??;
    assert_eq!(binary_message(&mut upstream).await?, b"RFB request");

    shutdown.notify();
    let (client_close, upstream_close) =
        tokio::join!(close_frame(&mut client), close_frame(&mut upstream));
    assert_eq!(client_close?, None);
    assert_eq!(upstream_close?, None);
    Ok(())
}

#[tokio::test]
async fn application_shutdown_interrupts_the_upstream_handshake() -> Result<()> {
    let listener = TcpListener::bind("127.0.0.1:0").await?;
    let upstream_url = format!("ws://{}", listener.local_addr()?);
    let shutdown = ShutdownSignal::default();
    let server = RunningServer::start(proxy_router(upstream_url, shutdown.clone())).await?;
    let (client, upstream) = tokio::join!(
        bounded(connect_async(server.url("/vnc"))),
        bounded(listener.accept())
    );
    let mut client = client??.0;
    let mut upstream = upstream??.0;
    // 沙箱已经收到握手请求，但故意保持 HTTP 响应未完成。
    let mut request = [0; 1024];
    assert!(bounded(upstream.read(&mut request)).await?? > 0);

    shutdown.notify();
    let mut remaining = Vec::new();
    let (client_close, upstream_closed) = tokio::join!(
        close_frame(&mut client),
        bounded(upstream.read_to_end(&mut remaining))
    );
    assert_eq!(client_close?, None);
    // EOF 证明连接 future 被取消时，尚未完成握手的 TCP 也得到释放。
    upstream_closed??;
    Ok(())
}

#[tokio::test]
async fn application_shutdown_before_forwarding_skips_the_upstream_connection() -> Result<()> {
    let listener = TcpListener::bind("127.0.0.1:0").await?;
    let upstream_url = format!("ws://{}", listener.local_addr()?);
    let shutdown = ShutdownSignal::default();
    shutdown.notify();
    let server = RunningServer::start(proxy_router(upstream_url, shutdown)).await?;
    let mut client = bounded(connect_async(server.url("/vnc"))).await??.0;

    assert_eq!(close_frame(&mut client).await?, None);
    // 关闭状态会保留；晚到的转发直接结束，沙箱监听端不会收到 TCP 连接。
    assert!(timeout(Duration::from_millis(100), listener.accept())
        .await
        .is_err());
    Ok(())
}

#[tokio::test]
async fn handshake_failure_closes_with_1011_and_a_valid_utf8_reason() -> Result<()> {
    let listener = TcpListener::bind("127.0.0.1:0").await?;
    let upstream_url = format!("ws://{}", listener.local_addr()?);
    let reason = format!("连接失败:{}", "沙箱环境异常".repeat(30));
    let expected = reason.clone();
    let router = Router::new().route(
        "/vnc",
        get(move |ws: WebSocketUpgrade| {
            let url = upstream_url.clone();
            let reason = reason.clone();
            async move {
                ws.on_upgrade(move |mut socket| async move {
                    let shutdown = ShutdownSignal::default();
                    assert!(forward_vnc(&mut socket, &url, &shutdown).await.is_err());
                    close_vnc_with_error(&mut socket, reason).await;
                })
            }
        }),
    );
    let server = RunningServer::start(router).await?;
    let (client, upstream) = tokio::join!(
        bounded(connect_async(server.url("/vnc"))),
        bounded(async {
            // TCP 可达，模拟沙箱在 WebSocket 握手完成前断开。
            let (stream, _) = listener.accept().await?;
            drop(stream);
            Result::<()>::Ok(())
        })
    );
    upstream??;
    let mut client = client??.0;
    let frame = close_frame(&mut client)
        .await?
        .context("错误关闭缺少原因")?;
    assert_eq!(frame.code, CloseCode::Error);
    let reason = frame.reason.as_str();
    assert!(reason.len() <= 123);
    assert!(expected.starts_with(reason));
    assert!(reason.len() >= 121, "原因应只在协议字节上限处截断");
    Ok(())
}

#[tokio::test]
async fn production_route_negotiates_protocol_then_closes_missing_session_or_sandbox() -> Result<()>
{
    let database = file_database::TestDatabase::new().await?;
    let manager = SchemaManager::new(&database.db);
    let migrations = Migrator::migrations()
        .into_iter()
        .filter(|migration| {
            matches!(
                migration.name(),
                "m20260720_184611_sessions" | "m20260720_191303_fix_sessions_table"
            )
        })
        .collect::<Vec<_>>();
    ensure!(migrations.len() == 2, "没有找到会话表迁移");
    for migration in migrations {
        migration.up(&manager).await?;
    }
    let repository = SeaOrmSessionRepository::new(database.db.clone());
    let session = Session::default();
    repository.save(session.clone()).await?;
    let stored_before = repository.get_by_id(&session.id).await?;
    let config: Config = serde_json::from_value(json!({
        "logger": {"enable": false, "level": "info", "format": "compact"},
        "server": {"port": 0, "host": "http://localhost"},
        "cache": {"kind": "Null"},
        "database": {
            "uri": "unused", "enable_logging": false,
            "min_connections": 1, "max_connections": 1,
            "connect_timeout": 10, "idle_timeout": 10
        }
    }))?;
    let ctx =
        loco_rs::app::AppContext::builder(Environment::Test, database.db.clone(), config).build();
    let router = App::routes(&ctx).to_router::<App>(ctx, Router::new())?;
    let server = RunningServer::start(router).await?;
    let missing_id = uuid::Uuid::new_v4().to_string();

    for (id, requested, selected, error) in [
        (
            &missing_id,
            Some("base64, binary"),
            Some("binary"),
            "当前会话不存在",
        ),
        (
            &session.id,
            Some("base64"),
            Some("base64"),
            "当前会话无沙箱环境",
        ),
        (
            &session.id,
            Some("binary"),
            Some("binary"),
            "当前会话无沙箱环境",
        ),
        (&missing_id, None, None, "当前会话不存在"),
    ] {
        let mut request = server
            .url(&format!("/api/sessions/{id}/vnc"))
            .into_client_request()?;
        if let Some(protocols) = requested {
            request
                .headers_mut()
                .insert("sec-websocket-protocol", protocols.parse()?);
        }
        let (mut client, response) = bounded(connect_async(request)).await??;
        assert_eq!(response.status(), 101);
        assert_eq!(
            response
                .headers()
                .get("sec-websocket-protocol")
                .map(|value| value.to_str().unwrap()),
            selected
        );
        let frame = close_frame(&mut client).await?.context("缺少错误关闭帧")?;
        assert_eq!(frame.code, CloseCode::Error);
        assert!(frame.reason.contains(error), "{frame:?}");
    }
    assert_eq!(repository.get_by_id(&session.id).await?, stored_before);
    Ok(())
}
