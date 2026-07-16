use serde::{Deserialize, Serialize};
use utoipa::ToSchema;

/// 进程信息模型
#[derive(Debug, Clone, Deserialize, Serialize, ToSchema, PartialEq, Eq)]
pub struct ProcessInfo {
    /// 进程名字
    pub name: String,
    /// 进程分组
    pub group: String,
    /// 进程描述
    pub description: String,
    /// 进程开始时间戳
    pub start: i64,
    /// 进程结束时间戳
    pub stop: i64,
    /// 当前时间戳
    pub now: i64,
    /// 状态代码
    pub state: i32,
    /// 状态名字
    pub statename: String,
    /// Spawn错误
    pub spawnerr: String,
    /// 退出状态
    pub exitstatus: i32,
    /// 日志文件
    pub logfile: String,
    /// 标准输出日志文件
    pub stdout_logfile: String,
    /// 标准错误日志文件
    pub stderr_logfile: String,
    /// 进程id(Process ID)
    pub pid: i64,
}

/// Supervisor批量启动或停止单个进程的结果
#[derive(Debug, Clone, Deserialize, Serialize, ToSchema, PartialEq, Eq)]
pub struct SupervisorProcessAction {
    /// 进程名字
    pub name: String,
    /// 进程分组
    pub group: String,
    /// 操作后的状态代码
    pub status: i32,
    /// 操作结果描述
    pub description: String,
}

/// Supervisor动作/执行结果
#[derive(Debug, Clone, Default, Deserialize, Serialize, ToSchema, PartialEq, Eq)]
pub struct SupervisorActionResult {
    /// 执行状态
    pub status: String,
    /// 执行结果
    pub result: Option<Vec<SupervisorProcessAction>>,
    /// 停止结果
    pub stop_result: Option<Vec<SupervisorProcessAction>>,
    /// 开始结果
    pub start_result: Option<Vec<SupervisorProcessAction>>,
    /// 关闭结果
    pub shutdown_result: Option<bool>,
}

impl SupervisorActionResult {
    /// 构造停止全部子进程的结果。
    pub fn stopped(result: Vec<SupervisorProcessAction>) -> Self {
        Self {
            status: "stopped".to_string(),
            result: Some(result),
            ..Self::default()
        }
    }

    /// 构造关闭 supervisord 主进程的结果。
    pub fn shutdown(shutdown_result: bool) -> Self {
        Self {
            status: "shutdown".to_string(),
            shutdown_result: Some(shutdown_result),
            ..Self::default()
        }
    }

    /// 构造已交给独立辅助进程执行的重启结果。
    pub fn restart_scheduled() -> Self {
        Self {
            status: "restart_scheduled".to_string(),
            ..Self::default()
        }
    }

    /// 构造先停止、再启动全部子进程的重启结果。
    pub fn restarted(
        stop_result: Vec<SupervisorProcessAction>,
        start_result: Vec<SupervisorProcessAction>,
    ) -> Self {
        Self {
            status: "restarted".to_string(),
            stop_result: Some(stop_result),
            start_result: Some(start_result),
            ..Self::default()
        }
    }
}

/// Supervisor超时销毁模型
#[derive(Debug, Clone, Default, Deserialize, Serialize, ToSchema, PartialEq, Eq)]
pub struct SupervisorTimeout {
    /// 超时设置状态
    pub status: Option<String>,
    /// 超时销毁是否激活
    #[serde(default)]
    pub active: bool,
    /// 销毁时间，使用 ISO-8601 UTC 字符串表示
    pub shutdown_time: Option<String>,
    /// 超时时间, 单位为分钟
    pub timeout_minutes: Option<usize>,
    /// 超时剩余秒数
    pub remaining_seconds: Option<usize>,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn supervisor_action_result_builders_match_course_response_shapes() {
        let stop_result = vec![SupervisorProcessAction {
            name: "app".to_string(),
            group: "services".to_string(),
            status: 80,
            description: "Stopped".to_string(),
        }];
        let start_result = vec![SupervisorProcessAction {
            name: "app".to_string(),
            group: "services".to_string(),
            status: 80,
            description: "Started".to_string(),
        }];

        let stopped = SupervisorActionResult::stopped(stop_result.clone());
        assert_eq!(stopped.status, "stopped");
        assert_eq!(stopped.result, Some(stop_result.clone()));
        assert_eq!(stopped.stop_result, None);

        let shutdown = SupervisorActionResult::shutdown(true);
        assert_eq!(shutdown.status, "shutdown");
        assert_eq!(shutdown.shutdown_result, Some(true));

        let scheduled = SupervisorActionResult::restart_scheduled();
        assert_eq!(scheduled.status, "restart_scheduled");
        assert_eq!(scheduled.stop_result, None);
        assert_eq!(scheduled.start_result, None);

        let restarted =
            SupervisorActionResult::restarted(stop_result.clone(), start_result.clone());
        assert_eq!(restarted.status, "restarted");
        assert_eq!(restarted.stop_result, Some(stop_result));
        assert_eq!(restarted.start_result, Some(start_result));
        assert_eq!(restarted.result, None);
    }

    #[test]
    fn process_info_supports_unix_timestamps_beyond_i32() {
        let process = ProcessInfo {
            name: "app".to_string(),
            group: "services".to_string(),
            description: "pid 42, uptime 0:00:10".to_string(),
            start: 2_200_000_000,
            stop: 0,
            now: 2_200_000_010,
            state: 20,
            statename: "RUNNING".to_string(),
            spawnerr: String::new(),
            exitstatus: 0,
            logfile: "/tmp/app.log".to_string(),
            stdout_logfile: "/dev/stdout".to_string(),
            stderr_logfile: "/dev/stderr".to_string(),
            pid: 42,
        };

        assert_eq!(process.start, 2_200_000_000);
        assert_eq!(process.now, 2_200_000_010);
        assert_eq!(process.pid, 42);
    }

    #[test]
    fn process_info_deserializes_supervisor_integer_pid() {
        let xml = r#"<?xml version="1.0" encoding="utf-8"?>
            <methodResponse><params><param><value><array><data><value><struct>
                <member><name>name</name><value><string>app</string></value></member>
                <member><name>group</name><value><string>services</string></value></member>
                <member><name>description</name><value><string>pid 42, uptime 0:00:10</string></value></member>
                <member><name>start</name><value><int>1750000000</int></value></member>
                <member><name>stop</name><value><int>0</int></value></member>
                <member><name>now</name><value><int>1750000010</int></value></member>
                <member><name>state</name><value><int>20</int></value></member>
                <member><name>statename</name><value><string>RUNNING</string></value></member>
                <member><name>spawnerr</name><value><string></string></value></member>
                <member><name>exitstatus</name><value><int>0</int></value></member>
                <member><name>logfile</name><value><string>/dev/stdout</string></value></member>
                <member><name>stdout_logfile</name><value><string>/dev/stdout</string></value></member>
                <member><name>stderr_logfile</name><value><string>/dev/stderr</string></value></member>
                <member><name>pid</name><value><int>42</int></value></member>
            </struct></value></data></array></value></param></params></methodResponse>"#;

        let processes: Vec<ProcessInfo> =
            serde_xmlrpc::response_from_str(xml).expect("Supervisor response should deserialize");

        assert_eq!(processes.len(), 1);
        assert_eq!(processes[0].name, "app");
        assert_eq!(processes[0].pid, 42);
    }

    #[test]
    fn supervisor_timeout_default_represents_an_inactive_timer() {
        let timeout = SupervisorTimeout::default();

        assert_eq!(timeout.status, None);
        assert!(!timeout.active);
        assert_eq!(timeout.shutdown_time, None);
        assert_eq!(timeout.timeout_minutes, None);
        assert_eq!(timeout.remaining_seconds, None);
    }
}
