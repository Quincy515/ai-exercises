//! 中英文提示词使用相同的常量名和占位符，默认导出中文版本。
//! 如需切换整套英文提示词，将下面三个 `pub use` 替换为 `pub use en::*;`。

pub mod en;
pub mod planner;
pub mod react;
pub mod system;

pub use planner::*;
pub use react::*;
pub use system::*;

/// 只替换原始模板中的命名占位符，不再次解析用户消息、附件或步骤中的内容。
/// 模板是原始字符串，JSON / TypeScript 的普通大括号直接保留，无须双写。
pub(crate) fn render_prompt(template: &str, variables: &[(&str, &str)]) -> String {
    let mut rendered = String::with_capacity(template.len());
    let mut cursor = 0;
    for (start, _) in template.match_indices('{') {
        if start < cursor {
            continue;
        }
        if let Some((placeholder, value)) = variables
            .iter()
            .find(|(placeholder, _)| template[start..].starts_with(placeholder))
        {
            rendered.push_str(&template[cursor..start]);
            rendered.push_str(value);
            cursor = start + placeholder.len();
        }
    }
    rendered.push_str(&template[cursor..]);
    rendered
}

#[cfg(test)]
mod tests;
