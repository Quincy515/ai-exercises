#[cfg(unix)]
use std::time::Duration;

#[cfg(unix)]
use reqwest::{Client, header::CONTENT_TYPE};
use tracing::error;

use crate::{exceptions::AppException, models::ProcessInfo};

const SUPERVISOR_SOCKET_PATH: &str = "/tmp/supervisor.sock";
const SUPERVISOR_RPC_URL: &str = "http://localhost/RPC2";
const GET_ALL_PROCESS_INFO_METHOD: &str = "supervisor.getAllProcessInfo";
const MAX_RPC_RESPONSE_BYTES: usize = 1024 * 1024;
#[cfg(all(unix, test))]
const RPC_TIMEOUT: Duration = Duration::from_millis(100);
#[cfg(all(unix, not(test)))]
const RPC_TIMEOUT: Duration = Duration::from_secs(10);

/*
1. Supervisor 启动后，通过 Unix Socket 文件实现 RPC 通信。
2. 连接 /tmp/supervisor.sock，建立 XML-RPC 连接。
3. 使用 reqwest 将 HTTP 请求完整转换到 Unix Socket 传输。
4. 连接成功后调用 supervisor.getAllProcessInfo() 获取全部进程状态。
*/

/// Supervisor 服务
#[derive(Debug)]
pub struct SupervisorService {
    pub rpc_url: String,
}

impl Default for SupervisorService {
    fn default() -> Self {
        Self::new()
    }
}

impl SupervisorService {
    /// 构造函数，保存 Supervisor Unix Socket 路径。
    ///
    /// 这里对应客户端初始化阶段；实际连接会在 `call_rpc()` 调用
    /// `send().await` 时异步建立。
    pub fn new() -> Self {
        Self::with_rpc_url(SUPERVISOR_SOCKET_PATH)
    }

    pub(crate) fn with_rpc_url(rpc_url: impl Into<String>) -> Self {
        Self {
            rpc_url: rpc_url.into(),
        }
    }

    /// 获取当前 Supervisor 管理的所有进程信息。
    pub async fn get_all_processes(&self) -> Result<Vec<ProcessInfo>, AppException> {
        let response = self.call_rpc().await.map_err(Self::process_error)?;
        parse_rpc_response(&response)
            .map_err(|err| Self::process_error(format!("RPC方法调用失败: {err}")))
    }

    /// 根据方法名调用 XML-RPC 服务。
    #[cfg(unix)]
    async fn call_rpc(&self) -> Result<String, String> {
        let request = build_get_all_process_info_request()?;

        // 这段配置承担 `_connect_rpc()` 的职责：
        // 1. `unix_socket()` 把 HTTP 传输固定到 `/tmp/supervisor.sock`；
        // 2. `http1_only()` 使用 Supervisor XML-RPC 所需的 HTTP/1.1；
        // 3. `build()` 创建异步 HTTP 客户端，此时只完成连接配置。
        let client = Client::builder()
            .unix_socket(self.rpc_url.as_str())
            .http1_only()
            .timeout(RPC_TIMEOUT)
            .build()
            .map_err(map_reqwest_error)?;

        // URL 用于生成 HTTP 请求路径 `/RPC2` 和 Host 头。
        // reqwest 会跳过 DNS，并在 `send().await` 时真正连接上面的 Unix Socket。
        let mut response = client
            .post(SUPERVISOR_RPC_URL)
            .header(CONTENT_TYPE, "text/xml")
            .body(request)
            .send()
            .await
            .map_err(map_reqwest_error)?;

        if !response.status().is_success() {
            return Err(format!(
                "RPC方法调用失败: Supervisor HTTP 请求失败: {}",
                response.status()
            ));
        }
        if response
            .content_length()
            .is_some_and(|length| length > MAX_RPC_RESPONSE_BYTES as u64)
        {
            return Err(response_too_large());
        }

        let mut body = Vec::new();
        while let Some(chunk) = response.chunk().await.map_err(map_reqwest_error)? {
            if body.len().saturating_add(chunk.len()) > MAX_RPC_RESPONSE_BYTES {
                return Err(response_too_large());
            }
            body.extend_from_slice(&chunk);
        }

        String::from_utf8(body).map_err(|err| format!("RPC方法调用失败: 响应不是 UTF-8: {err}"))
    }

    #[cfg(not(unix))]
    async fn call_rpc(&self) -> Result<String, String> {
        Err("RPC方法调用失败: Supervisor Unix Socket 仅支持 Unix 系统".to_string())
    }

    fn process_error(err: String) -> AppException {
        let msg = format!("获取进程信息失败: {err}");
        error!(error = %msg, "获取 Supervisor 进程信息失败");
        AppException::internal(msg)
    }
}

fn build_get_all_process_info_request() -> Result<String, String> {
    serde_xmlrpc::request_to_string(
        GET_ALL_PROCESS_INFO_METHOD,
        std::iter::empty::<serde_xmlrpc::Value>(),
    )
    .map_err(|err| format!("RPC方法调用失败: 构造请求失败: {err}"))
}

fn parse_rpc_response(response: &str) -> Result<Vec<ProcessInfo>, String> {
    if !response.trim_end().ends_with("</methodResponse>") {
        return Err("Supervisor XML-RPC methodResponse 根节点不完整或包含多余内容".to_string());
    }

    let mut processes: Vec<ProcessInfo> = match serde_xmlrpc::response_from_str(response) {
        Ok(processes) => processes,
        Err(serde_xmlrpc::Error::Fault(mut fault)) => {
            fault.fault_string = decode_xml_entities(&fault.fault_string)?;
            return Err(format!("Supervisor XML-RPC 响应解析失败: {fault}"));
        }
        Err(err) => return Err(format!("Supervisor XML-RPC 响应解析失败: {err}")),
    };
    for process in &mut processes {
        decode_process_strings(process)?;
    }
    Ok(processes)
}

fn decode_process_strings(process: &mut ProcessInfo) -> Result<(), String> {
    for value in [
        &mut process.name,
        &mut process.group,
        &mut process.description,
        &mut process.statename,
        &mut process.spawnerr,
        &mut process.logfile,
        &mut process.stdout_logfile,
        &mut process.stderr_logfile,
    ] {
        *value = decode_xml_entities(value)?;
    }
    Ok(())
}

/// serde_xmlrpc 0.3.0-alpha.2 会保留字符串中的 XML 实体，这里统一解码一次。
fn decode_xml_entities(value: &str) -> Result<String, String> {
    let mut decoded = String::with_capacity(value.len());
    let mut remaining = value;

    while let Some(index) = remaining.find('&') {
        decoded.push_str(&remaining[..index]);
        let entity = &remaining[index + 1..];
        let end = entity
            .find(';')
            .ok_or_else(|| format!("Supervisor XML-RPC 包含无效实体: {remaining}"))?;
        let entity_name = &entity[..end];
        let character = match entity_name {
            "amp" => '&',
            "lt" => '<',
            "gt" => '>',
            "quot" => '"',
            "apos" => '\'',
            value if value.starts_with("#x") => u32::from_str_radix(&value[2..], 16)
                .ok()
                .and_then(char::from_u32)
                .ok_or_else(|| format!("Supervisor XML-RPC 包含无效实体: &{value};"))?,
            value if value.starts_with('#') => value[1..]
                .parse::<u32>()
                .ok()
                .and_then(char::from_u32)
                .ok_or_else(|| format!("Supervisor XML-RPC 包含无效实体: &{value};"))?,
            value => return Err(format!("Supervisor XML-RPC 包含未知实体: &{value};")),
        };
        decoded.push(character);
        remaining = &entity[end + 1..];
    }

    decoded.push_str(remaining);
    Ok(decoded)
}

#[cfg(unix)]
fn map_reqwest_error(err: reqwest::Error) -> String {
    if err.is_timeout() {
        format!("RPC方法调用失败: 超时({}ms)", RPC_TIMEOUT.as_millis())
    } else {
        format!("RPC方法调用失败: {err}")
    }
}

#[cfg(unix)]
fn response_too_large() -> String {
    format!("RPC方法调用失败: Supervisor 响应过大(上限 {MAX_RPC_RESPONSE_BYTES} 字节)")
}

#[cfg(test)]
mod tests {
    use std::path::PathBuf;

    use super::*;

    const PROCESS_ARRAY: &str = r#"
        <array><data>
          <value><struct>
            <member><name>name</name><value><string>xvfb</string></value></member>
            <member><name>group</name><value><string>services</string></value></member>
            <member><name>description</name><value><string>pid 10, uptime 0:00:03</string></value></member>
            <member><name>start</name><value><int>1750000000</int></value></member>
            <member><name>stop</name><value><int>0</int></value></member>
            <member><name>now</name><value><int>1750000003</int></value></member>
            <member><name>state</name><value><int>20</int></value></member>
            <member><name>statename</name><value><string>RUNNING</string></value></member>
            <member><name>spawnerr</name><value><string></string></value></member>
            <member><name>exitstatus</name><value><int>0</int></value></member>
            <member><name>logfile</name><value><string>/tmp/xvfb.log</string></value></member>
            <member><name>stdout_logfile</name><value><string>/dev/stdout</string></value></member>
            <member><name>stderr_logfile</name><value><string>/dev/stderr</string></value></member>
            <member><name>pid</name><value><int>10</int></value></member>
          </struct></value>
          <value><struct>
            <member><name>name</name><value><string>chrome</string></value></member>
            <member><name>group</name><value><string>services</string></value></member>
            <member><name>description</name><value><string>ready &amp; listening</string></value></member>
            <member><name>start</name><value><int>1750000001</int></value></member>
            <member><name>stop</name><value><int>0</int></value></member>
            <member><name>now</name><value><int>1750000003</int></value></member>
            <member><name>state</name><value><int>20</int></value></member>
            <member><name>statename</name><value><string>RUNNING</string></value></member>
            <member><name>spawnerr</name><value><string></string></value></member>
            <member><name>exitstatus</name><value><int>0</int></value></member>
            <member><name>logfile</name><value><string>/tmp/chrome.log</string></value></member>
            <member><name>stdout_logfile</name><value><string>/dev/stdout</string></value></member>
            <member><name>stderr_logfile</name><value><string>/dev/stderr</string></value></member>
            <member><name>pid</name><value><int>11</int></value></member>
          </struct></value>
        </data></array>
    "#;

    fn success_xml(body: &str) -> String {
        format!(
            "<?xml version=\"1.0\"?><methodResponse><params><param><value>{body}</value></param></params></methodResponse>"
        )
    }

    fn http_response(xml: &str) -> String {
        format!(
            "HTTP/1.1 200 OK\r\nContent-Type: text/xml\r\nContent-Length: {}\r\n\r\n{xml}",
            xml.len()
        )
    }

    #[cfg(unix)]
    async fn call_mock_supervisor(response: String) -> Result<Vec<ProcessInfo>, AppException> {
        use tokio::{
            io::{AsyncReadExt, AsyncWriteExt},
            net::UnixListener,
        };

        let socket_path = PathBuf::from(format!(
            "/tmp/sandbox-supervisor-test-{}.sock",
            uuid::Uuid::new_v4()
        ));
        let listener = UnixListener::bind(&socket_path).expect("test socket should bind");
        let server = tokio::spawn(async move {
            let (mut stream, _) = listener.accept().await.expect("client should connect");
            let mut request = Vec::new();
            let mut chunk = [0_u8; 1024];
            loop {
                let read = stream.read(&mut chunk).await.expect("request should read");
                if read == 0 {
                    break;
                }
                request.extend_from_slice(&chunk[..read]);
                if request
                    .windows(b"</methodCall>".len())
                    .any(|window| window == b"</methodCall>")
                {
                    break;
                }
            }
            let request = String::from_utf8_lossy(&request);
            assert!(request.starts_with("POST /RPC2 HTTP/1.1"));
            assert!(request.contains("supervisor.getAllProcessInfo"));
            stream
                .write_all(response.as_bytes())
                .await
                .expect("response should write");
        });

        let service = SupervisorService::with_rpc_url(socket_path.display().to_string());
        let result = service.get_all_processes().await;
        server.await.expect("mock server should finish");
        std::fs::remove_file(socket_path).expect("test socket should be removed");
        result
    }

    #[test]
    fn default_service_uses_supervisor_socket() {
        assert_eq!(SupervisorService::default().rpc_url, "/tmp/supervisor.sock");
    }

    #[test]
    fn rpc_request_calls_course_method_without_parameters() {
        let request = build_get_all_process_info_request().expect("request should serialize");
        let (method, parameters) =
            serde_xmlrpc::request_from_str(&request).expect("request should parse");

        assert_eq!(method, "supervisor.getAllProcessInfo");
        assert!(parameters.is_empty());
    }

    #[test]
    fn rpc_response_maps_all_fields_and_preserves_order() {
        let processes = parse_rpc_response(&success_xml(PROCESS_ARRAY))
            .expect("valid Supervisor response should parse");

        assert_eq!(processes.len(), 2);
        assert_eq!(processes[0].name, "xvfb");
        assert_eq!(processes[0].start, 1_750_000_000);
        assert_eq!(processes[0].pid, 10);
        assert_eq!(processes[1].name, "chrome");
        assert_eq!(processes[1].description, "ready & listening");
        assert_eq!(processes[1].stdout_logfile, "/dev/stdout");
    }

    #[test]
    fn rpc_response_rejects_process_with_missing_required_field() {
        let body = PROCESS_ARRAY.replacen(
            "<member><name>group</name><value><string>services</string></value></member>",
            "",
            1,
        );

        let error = parse_rpc_response(&success_xml(&body))
            .expect_err("missing group should reject the complete response");

        assert!(error.contains("group"), "unexpected error: {error}");
    }

    #[test]
    fn rpc_response_rejects_truncated_array() {
        let xml = success_xml("<array><data><value><struct><member><name>name</name>");

        let error = parse_rpc_response(&xml)
            .expect_err("truncated array should reject the complete response");

        assert!(error.contains("解析失败"), "unexpected error: {error}");
    }

    #[test]
    fn rpc_response_rejects_partial_process_list() {
        let (truncated, _) = PROCESS_ARRAY
            .rsplit_once("</struct></value>")
            .expect("fixture should contain a final process");

        let error = parse_rpc_response(&success_xml(truncated))
            .expect_err("truncated second process should reject the complete response");

        assert!(error.contains("解析失败"), "unexpected error: {error}");
    }

    #[test]
    fn rpc_response_rejects_content_after_method_response_root() {
        let xml = format!("{}<broken", success_xml(PROCESS_ARRAY));

        let error = parse_rpc_response(&xml)
            .expect_err("content after the XML root should fail document validation");

        assert!(error.contains("根节点"), "unexpected error: {error}");
    }

    #[test]
    fn rpc_fault_is_reported_with_fault_message() {
        let body = r#"<fault><value><struct>
            <member><name>faultCode</name><value><int>1</int></value></member>
            <member><name>faultString</name><value><string>UNKNOWN &amp; METHOD</string></value></member>
        </struct></value></fault>"#;
        let xml = format!("<?xml version=\"1.0\"?><methodResponse>{body}</methodResponse>");

        let error = parse_rpc_response(&xml).expect_err("XML-RPC fault should be an error");

        assert!(
            error.contains("UNKNOWN & METHOD"),
            "unexpected error: {error}"
        );
    }

    #[cfg(unix)]
    #[tokio::test]
    async fn get_all_processes_uses_unix_socket_and_returns_typed_models() {
        let xml = success_xml(PROCESS_ARRAY);

        let processes = call_mock_supervisor(http_response(&xml))
            .await
            .expect("mock Supervisor call should succeed");

        assert_eq!(processes.len(), 2);
        assert_eq!(processes[0].name, "xvfb");
        assert_eq!(processes[1].name, "chrome");
    }

    #[cfg(unix)]
    #[tokio::test]
    async fn get_all_processes_accepts_chunked_http_response() {
        let xml = success_xml(PROCESS_ARRAY);
        let response = format!(
            "HTTP/1.1 200 OK\r\nContent-Type: text/xml\r\nTransfer-Encoding: chunked\r\n\r\n{:X}\r\n{xml}\r\n0\r\n\r\n",
            xml.len()
        );

        let processes = call_mock_supervisor(response)
            .await
            .expect("reqwest should decode chunked HTTP framing");

        assert_eq!(processes.len(), 2);
        assert_eq!(processes[0].name, "xvfb");
        assert_eq!(processes[1].name, "chrome");
    }

    #[cfg(unix)]
    #[tokio::test]
    async fn get_all_processes_rejects_mismatched_content_length() {
        let xml = success_xml(PROCESS_ARRAY);
        let response = format!(
            "HTTP/1.1 200 OK\r\nContent-Type: text/xml\r\nContent-Length: {}\r\n\r\n{xml}",
            xml.len() + 1
        );

        let error = call_mock_supervisor(response)
            .await
            .expect_err("truncated HTTP response should fail");

        assert!(
            error.msg.contains("RPC方法调用失败"),
            "unexpected error: {}",
            error.msg
        );
    }

    #[cfg(unix)]
    #[tokio::test]
    async fn get_all_processes_rejects_conflicting_content_length_headers() {
        let xml = success_xml(PROCESS_ARRAY);
        let response = format!(
            "HTTP/1.1 200 OK\r\nContent-Type: text/xml\r\nContent-Length: {}\r\nContent-Length: {}\r\n\r\n{xml}",
            xml.len(),
            xml.len() + 1
        );

        let error = call_mock_supervisor(response)
            .await
            .expect_err("conflicting Content-Length headers should fail");

        assert!(
            error.msg.contains("RPC方法调用失败"),
            "unexpected error: {}",
            error.msg
        );
    }

    #[cfg(unix)]
    #[tokio::test]
    async fn get_all_processes_rejects_oversized_response() {
        let response = format!(
            "HTTP/1.1 200 OK\r\nContent-Type: text/xml\r\nContent-Length: {}\r\n\r\n",
            MAX_RPC_RESPONSE_BYTES + 1
        );

        let error = call_mock_supervisor(response)
            .await
            .expect_err("oversized response should fail before reading its body");

        assert!(
            error.msg.contains("响应过大"),
            "unexpected error: {}",
            error.msg
        );
    }

    #[cfg(unix)]
    #[tokio::test]
    async fn get_all_processes_rejects_oversized_chunked_response_while_streaming() {
        let first_chunk = "x".repeat(MAX_RPC_RESPONSE_BYTES);
        let response = format!(
            "HTTP/1.1 200 OK\r\nContent-Type: text/xml\r\nTransfer-Encoding: chunked\r\n\r\n{:X}\r\n{first_chunk}\r\n1\r\nx\r\n0\r\n\r\n",
            first_chunk.len()
        );

        let error = call_mock_supervisor(response)
            .await
            .expect_err("streamed response should enforce the same size limit");

        assert!(
            error.msg.contains("响应过大"),
            "unexpected error: {}",
            error.msg
        );
    }

    #[cfg(unix)]
    #[tokio::test]
    async fn get_all_processes_wraps_connection_failure_as_app_error() {
        let socket_path = PathBuf::from(format!(
            "/tmp/missing-supervisor-{}.sock",
            uuid::Uuid::new_v4()
        ));
        let service = SupervisorService::with_rpc_url(socket_path.display().to_string());

        let error = service
            .get_all_processes()
            .await
            .expect_err("missing socket should fail");

        assert_eq!(
            error.status_code,
            axum::http::StatusCode::INTERNAL_SERVER_ERROR
        );
        assert!(error.msg.contains("获取进程信息失败"));
        assert!(error.msg.contains("RPC方法调用失败"));
    }

    #[cfg(unix)]
    #[tokio::test]
    async fn get_all_processes_times_out_when_supervisor_stalls() {
        use tokio::{io::AsyncReadExt, net::UnixListener};

        let socket_path = PathBuf::from(format!(
            "/tmp/stalled-supervisor-{}.sock",
            uuid::Uuid::new_v4()
        ));
        let listener = UnixListener::bind(&socket_path).expect("test socket should bind");
        let server = tokio::spawn(async move {
            let (mut stream, _) = listener.accept().await.expect("client should connect");
            let mut request = vec![0_u8; 4096];
            let _ = stream
                .read(&mut request)
                .await
                .expect("request should read");
            std::future::pending::<()>().await;
        });
        let service = SupervisorService::with_rpc_url(socket_path.display().to_string());

        let result = tokio::time::timeout(
            std::time::Duration::from_millis(300),
            service.get_all_processes(),
        )
        .await
        .expect("service should enforce an RPC timeout");
        let error = result.expect_err("stalled Supervisor should time out");

        assert!(
            error.msg.contains("超时"),
            "unexpected error: {}",
            error.msg
        );
        server.abort();
        let _ = server.await;
        std::fs::remove_file(socket_path).expect("test socket should be removed");
    }
}
