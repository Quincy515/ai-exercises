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

#[cfg(test)]
mod tests {
    use super::*;

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
}
