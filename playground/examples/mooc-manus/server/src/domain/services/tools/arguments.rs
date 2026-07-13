use anyhow::{anyhow, Result};
use serde_json::Value;

use super::base::ToolArguments;

/// 读取必填字符串参数。空字符串是否有效由具体工具决定。
pub(super) fn required_str<'a>(kwargs: &'a ToolArguments, name: &str) -> Result<&'a str> {
    match kwargs.get(name) {
        Some(Value::String(value)) => Ok(value),
        Some(Value::Null) | None => Err(anyhow!("工具参数[{name}]缺失")),
        Some(_) => Err(anyhow!("工具参数[{name}]必须是字符串")),
    }
}

/// 读取不能为空或纯空白的必填字符串参数。
pub(super) fn required_non_empty_str<'a>(kwargs: &'a ToolArguments, name: &str) -> Result<&'a str> {
    let value = required_str(kwargs, name)?;
    if value.trim().is_empty() {
        Err(anyhow!("工具参数[{name}]缺失"))
    } else {
        Ok(value)
    }
}

/// 读取必填布尔参数。
pub(super) fn required_bool(kwargs: &ToolArguments, name: &str) -> Result<bool> {
    optional_bool(kwargs, name)?.ok_or_else(|| anyhow!("工具参数[{name}]缺失"))
}

/// 读取必填浮点数参数。
pub(super) fn required_f32(kwargs: &ToolArguments, name: &str) -> Result<f32> {
    optional_f32(kwargs, name)?.ok_or_else(|| anyhow!("工具参数[{name}]缺失"))
}

/// 读取必填非负整数参数。
pub(super) fn required_usize(kwargs: &ToolArguments, name: &str) -> Result<usize> {
    optional_usize(kwargs, name)?.ok_or_else(|| anyhow!("工具参数[{name}]缺失"))
}

/// 读取可选布尔参数。
pub(super) fn optional_bool(kwargs: &ToolArguments, name: &str) -> Result<Option<bool>> {
    match kwargs.get(name) {
        Some(Value::Null) | None => Ok(None),
        Some(value) => value
            .as_bool()
            .map(Some)
            .ok_or_else(|| anyhow!("工具参数[{name}]必须是布尔值")),
    }
}

/// 读取可选浮点数参数。
pub(super) fn optional_f32(kwargs: &ToolArguments, name: &str) -> Result<Option<f32>> {
    match kwargs.get(name) {
        Some(Value::Null) | None => Ok(None),
        Some(value) => value
            .as_f64()
            .map(|value| value as f32)
            .map(Some)
            .ok_or_else(|| anyhow!("工具参数[{name}]必须是数字")),
    }
}

/// 读取可选非负整数参数。
pub(super) fn optional_usize(kwargs: &ToolArguments, name: &str) -> Result<Option<usize>> {
    match kwargs.get(name) {
        Some(Value::Null) | None => Ok(None),
        Some(value) => value
            .as_u64()
            .and_then(|value| usize::try_from(value).ok())
            .map(Some)
            .ok_or_else(|| anyhow!("工具参数[{name}]必须是非负整数")),
    }
}

#[cfg(test)]
mod tests {
    use serde_json::{json, Map};

    use super::*;

    #[test]
    fn parses_required_argument_types() {
        let kwargs = Map::from_iter([
            ("text".to_string(), json!("  value  ")),
            ("enabled".to_string(), json!(true)),
            ("coordinate".to_string(), json!(12.5)),
            ("index".to_string(), json!(3)),
        ]);

        assert_eq!(required_str(&kwargs, "text").unwrap(), "  value  ");
        assert_eq!(
            required_non_empty_str(&kwargs, "text").unwrap(),
            "  value  "
        );
        assert!(required_bool(&kwargs, "enabled").unwrap());
        assert_eq!(required_f32(&kwargs, "coordinate").unwrap(), 12.5);
        assert_eq!(required_usize(&kwargs, "index").unwrap(), 3);
    }

    #[test]
    fn distinguishes_empty_strings_from_missing_strings() {
        let kwargs = Map::from_iter([
            ("empty".to_string(), json!("")),
            ("blank".to_string(), json!("   ")),
        ]);

        assert_eq!(required_str(&kwargs, "empty").unwrap(), "");
        assert_eq!(
            required_non_empty_str(&kwargs, "blank")
                .unwrap_err()
                .to_string(),
            "工具参数[blank]缺失"
        );
        assert_eq!(
            required_str(&kwargs, "missing").unwrap_err().to_string(),
            "工具参数[missing]缺失"
        );
    }

    #[test]
    fn optional_arguments_accept_missing_null_and_valid_values() {
        let kwargs = Map::from_iter([
            ("null".to_string(), Value::Null),
            ("enabled".to_string(), json!(false)),
            ("coordinate".to_string(), json!(8.25)),
            ("index".to_string(), json!(7)),
        ]);

        assert_eq!(optional_bool(&kwargs, "missing").unwrap(), None);
        assert_eq!(optional_usize(&kwargs, "null").unwrap(), None);
        assert_eq!(optional_bool(&kwargs, "enabled").unwrap(), Some(false));
        assert_eq!(optional_f32(&kwargs, "coordinate").unwrap(), Some(8.25));
        assert_eq!(optional_usize(&kwargs, "index").unwrap(), Some(7));
    }

    #[test]
    fn reports_invalid_argument_types() {
        let kwargs = Map::from_iter([
            ("text".to_string(), json!(1)),
            ("enabled".to_string(), json!("yes")),
            ("coordinate".to_string(), json!("left")),
            ("negative".to_string(), json!(-1)),
            ("fraction".to_string(), json!(1.5)),
        ]);

        assert_eq!(
            required_str(&kwargs, "text").unwrap_err().to_string(),
            "工具参数[text]必须是字符串"
        );
        assert_eq!(
            required_bool(&kwargs, "enabled").unwrap_err().to_string(),
            "工具参数[enabled]必须是布尔值"
        );
        assert_eq!(
            required_f32(&kwargs, "coordinate").unwrap_err().to_string(),
            "工具参数[coordinate]必须是数字"
        );
        for name in ["negative", "fraction"] {
            assert_eq!(
                optional_usize(&kwargs, name).unwrap_err().to_string(),
                format!("工具参数[{name}]必须是非负整数")
            );
        }
    }
}
