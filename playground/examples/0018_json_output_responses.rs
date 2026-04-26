use anyhow::{Context, ensure};
use async_openai::config::OpenAIConfig;
use async_openai::{
    Client,
    types::responses::{
        CreateResponseArgs, InputContent, InputMessage, InputRole, TextResponseFormatConfiguration,
    },
};
use serde::Deserialize;
use std::env;
use std::io;
use std::io::Write;
use tokio::io::{AsyncBufReadExt, BufReader};
use validator::Validate;

const DEFAULT_MODEL: &str = "gpt-5.4-mini";
const SYSTEM_MESSAGE: &str = r#"
用户将提问一个问题，请拆解这个问题为多个串联的小任务，拆解的小任务数量不超过10个，你可以使用任何假设的工具、LLM、代码等。
并以json格式输出，其中task_count字段代表拆分任务的总数，tasks为拆分的任务数组(tasks数组内的每个元素都是一个字符串，有顺序之分)。

示例输入:
今天广州的天气怎样?

示例输出:
{
    "task_count": 3,
    "tasks": ["调用浏览器搜索今天的时间", "调用浏览器搜索广州的天气", "综合搜索的结果/内容调用LLM整理答案并回复用户"]
}
"#;

#[derive(Debug, Deserialize, Validate)]
struct SplitTask {
    #[validate(range(min = 1, max = 10))]
    task_count: usize,
    tasks: Vec<String>,
}

#[tokio::main]
async fn main() -> Result<(), anyhow::Error> {
    dotenvy::dotenv().ok();
    let client = Client::new();
    chat_loop(&client).await
}

async fn chat(client: &Client<OpenAIConfig>, user_prompt: &str) -> anyhow::Result<SplitTask> {
    let request = CreateResponseArgs::default()
        .model(env::var("OPENAI_MODEL").unwrap_or_else(|_| DEFAULT_MODEL.to_string()))
        .input(vec![
            InputMessage {
                role: InputRole::System,
                content: vec![InputContent::from(SYSTEM_MESSAGE)],
                status: None,
            },
            InputMessage {
                role: InputRole::User,
                content: vec![InputContent::from(user_prompt)],
                status: None,
            },
        ])
        .text(TextResponseFormatConfiguration::JsonObject)
        .build()?;

    let response = client.responses().create(request).await?;
    let content = response
        .output_text()
        .context("模型没有返回可解析的 JSON 内容")?;

    SplitTask::model_validate_json(&content)
}

/// 交互式 REPL 循环；输入 `quit` 退出。
async fn chat_loop(client: &Client<OpenAIConfig>) -> Result<(), anyhow::Error> {
    // 异步读取标准输入，避免阻塞 tokio 运行时
    let stdin = tokio::io::stdin();
    let mut reader = BufReader::new(stdin);

    loop {
        print!("\nQuery: ");
        // 立即刷新提示符，保证在等待输入前可见
        io::stdout().flush().ok();

        // 部分 IDE 控制台可能传入非 UTF-8 字节，按字节读取可以避免 REPL 直接退出。
        let mut line = Vec::new();
        if reader.read_until(b'\n', &mut line).await? == 0 {
            break;
        }

        let query = String::from_utf8_lossy(&line);
        let query = query.trim();
        if query.is_empty() {
            continue;
        }
        if query.eq_ignore_ascii_case("quit") {
            break;
        }

        // 单轮出错不应中断整个会话，只打印错误信息继续下一轮
        match chat(client, query).await {
            Ok(split_task) => {
                println!("拆解任务数: {}", split_task.task_count);
                for (idx, task) in split_task.tasks.iter().enumerate() {
                    println!("{:02}.{}", idx + 1, task);
                }
                println!("===============");
            }
            Err(e) => eprintln!("\nError: {e:#}"),
        }
    }
    Ok(())
}

impl SplitTask {
    fn model_validate_json(content: &str) -> anyhow::Result<Self> {
        let split_task: Self = serde_json::from_str(content)
            .with_context(|| format!("模型回复不是有效的 SplitTask JSON: {content}"))?;
        split_task.validate()?;
        ensure!(
            split_task.task_count == split_task.tasks.len(),
            "task_count={} 与 tasks 数量={} 不一致",
            split_task.task_count,
            split_task.tasks.len()
        );
        Ok(split_task)
    }
}
