use serde::{Deserialize, Serialize};

/// 沙箱 API 服务基础配置信息
#[derive(Debug, Serialize, Deserialize, Clone)]
pub struct Settings {
    /// 日志等级
    pub log_level: String,
    /// 服务超时时间单位为分钟
    pub server_timeout_minutes: usize,
}

impl Default for Settings {
    fn default() -> Self {
        Self {
            log_level: "INFO".to_string(),
            server_timeout_minutes: 60,
        }
    }
}

impl Settings {
    pub fn load() -> Self {
        let mut settings = Settings::default();
        let _ = dotenvy::from_path(".env");
        settings.log_level =
            dotenvy::var("LOG_LEVEL").unwrap_or_else(|_| settings.log_level.clone());
        settings.server_timeout_minutes = dotenvy::var("SERVER_TIMEOUT_MINUTES")
            .unwrap_or_else(|_| settings.server_timeout_minutes.to_string())
            .parse()
            .unwrap_or(settings.server_timeout_minutes);
        settings
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn default_settings_match_course_config() {
        let settings = Settings::default();

        assert_eq!(settings.log_level, "INFO");
        assert_eq!(settings.server_timeout_minutes, 60);
    }
}
