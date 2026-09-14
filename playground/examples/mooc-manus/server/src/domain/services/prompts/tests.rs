use std::collections::BTreeSet;

use regex::Regex;
use serde_json::Value;

use super::{en, planner, react, render_prompt};
use crate::domain::models::{Message, Plan, Step};

fn example(prompt: &str) -> Value {
    let (_, example) = prompt
        .split_once("JSON 输出示例：")
        .or_else(|| prompt.split_once("EXAMPLE JSON OUTPUT:"))
        .expect("模板必须提供 JSON 示例");
    // 只读取示例对象，后面的输入说明不是 JSON。
    serde_json::Deserializer::from_str(example)
        .into_iter::<Value>()
        .next()
        .expect("必须存在 JSON 对象")
        .expect("发送给模型的示例必须是合法 JSON")
}

#[test]
fn both_languages_render_examples_accepted_by_domain_models() {
    let variables = [
        ("{message}", "整理销售报告"),
        ("{attachments}", "/home/ubuntu/销售.csv"),
        ("{language}", "中文"),
        ("{step}", "分析销售额"),
        ("{plan}", r#"{"goal":"完成分析","steps":[]}"#),
    ];
    for (create, update, execute, summarize) in [
        (
            planner::CREATE_PLAN_PROMPT,
            planner::UPDATE_PLAN_PROMPT,
            react::EXECUTION_PROMPT,
            react::SUMMARIZE_PROMPT,
        ),
        (
            en::CREATE_PLAN_PROMPT,
            en::UPDATE_PLAN_PROMPT,
            en::EXECUTION_PROMPT,
            en::SUMMARIZE_PROMPT,
        ),
    ] {
        let create = example(&render_prompt(create, &variables));
        for field in ["message", "language", "goal", "title"] {
            assert!(create[field].is_string(), "缺少字符串字段 {field}");
        }
        let plan: Plan = serde_json::from_value(create).unwrap();
        assert_eq!(plan.steps[0].id, "1");
        assert!(!plan.steps[0].description.is_empty());

        let updated: Plan =
            serde_json::from_value(example(&render_prompt(update, &variables))).unwrap();
        assert_eq!(updated.steps[0].id, "1");

        let execution = example(&render_prompt(execute, &variables));
        assert!(execution["success"].is_boolean());
        assert!(execution["result"].is_string());
        let step: Step = serde_json::from_value(execution).unwrap();
        assert!(step.success);
        assert_eq!(step.attachments.len(), 2);

        // 汇总模板没有变量，ReActAgent 直接使用原始字符串。
        let summary = example(summarize);
        assert!(summary["message"].is_string());
        let message: Message = serde_json::from_value(summary).unwrap();
        assert_eq!(message.attachments.len(), 2);
    }
}

#[test]
fn both_languages_keep_the_same_placeholder_contract() {
    let placeholder = Regex::new(r"\{[a-z]+\}").unwrap();
    for (chinese, english, expected) in [
        (
            planner::CREATE_PLAN_PROMPT,
            en::CREATE_PLAN_PROMPT,
            vec!["{message}", "{attachments}"],
        ),
        (
            planner::UPDATE_PLAN_PROMPT,
            en::UPDATE_PLAN_PROMPT,
            vec!["{plan}", "{step}"],
        ),
        (
            react::EXECUTION_PROMPT,
            en::EXECUTION_PROMPT,
            vec!["{message}", "{attachments}", "{language}", "{step}"],
        ),
        (react::SUMMARIZE_PROMPT, en::SUMMARIZE_PROMPT, vec![]),
        (super::system::SYSTEM_PROMPT, en::SYSTEM_PROMPT, vec![]),
        (
            planner::PLANNER_SYSTEM_PROMPT,
            en::PLANNER_SYSTEM_PROMPT,
            vec![],
        ),
        (react::REACT_SYSTEM_PROMPT, en::REACT_SYSTEM_PROMPT, vec![]),
    ] {
        let expected: BTreeSet<_> = expected.into_iter().collect();
        for template in [chinese, english] {
            let actual: BTreeSet<_> = placeholder
                .find_iter(template)
                .map(|value| value.as_str())
                .collect();
            assert_eq!(actual, expected);
            assert!(!template.contains("{{"));
            assert!(!template.contains("}}"));
        }
    }
}

#[test]
fn rendering_preserves_literal_braces_and_does_not_reparse_inserted_text() {
    let rendered = render_prompt(
        "interface Response { id: string; }\n{message}\n{attachments}\n{step}\n{message}",
        &[
            ("{message}", "请解释 {attachments}、{step} 和 {{原始括号}}"),
            ("{attachments}", "/home/ubuntu/{step}.md"),
            ("{step}", "阅读 {message}"),
        ],
    );
    assert_eq!(
        rendered,
        "interface Response { id: string; }\n请解释 {attachments}、{step} 和 {{原始括号}}\n/home/ubuntu/{step}.md\n阅读 {message}\n请解释 {attachments}、{step} 和 {{原始括号}}"
    );
}
