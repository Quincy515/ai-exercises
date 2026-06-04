use anyhow::{anyhow, Result};
use async_trait::async_trait;
use serde_json::{json, Value};

use crate::domain::{
    external::SearchEngine,
    models::{SearchResults, ToolResult},
    services::tools::{tool, BaseTool, ToolArguments, ToolDefinition},
};

/// 搜索工具包，提供与搜索引擎交互的能力。
pub struct SearchTool {
    search_engine: Box<dyn SearchEngine>,
    definitions: Vec<ToolDefinition>,
}

impl SearchTool {
    /// 构造函数，完成搜索工具包的初始化。
    pub fn new(search_engine: Box<dyn SearchEngine>) -> Self {
        Self {
            search_engine,
            definitions: vec![tool(
                "search_web",
                "全网搜索引擎工具。当需要获取实时信息（如突发新闻、天气）、补充内部知识库未涵盖的内容或进行事实核查时使用。该工具会返回相关的网页摘要和链接。",
                ToolArguments::from_iter([
                    (
                        "query".to_string(),
                        json!({
                            "type": "string",
                            "description": "针对搜索引擎优化的查询字符串。请提取问题中的核心实体和关键词（3-5个），避免使用完整的自然语言问句（例如将'今天北京的天气怎么样'转换为'北京 天气'）。"
                        }),
                    ),
                    (
                        "date_range".to_string(),
                        json!({
                            "type": "string",
                            "enum": [
                                "all",
                                "past_hour",
                                "past_day",
                                "past_week",
                                "past_month",
                                "past_year"
                            ],
                            "description": "（可选）搜索结果的时间范围过滤。当用户询问特定时效性的新闻或事件时（如'昨天'、'上周'），必须指定此参数。默认为 'all'。"
                        }),
                    ),
                ]),
                vec!["query".to_string()],
            )],
        }
    }

    /// 调用搜索引擎获取搜索结果后返回。
    async fn search_web(
        &self,
        query: String,
        date_range: Option<String>,
    ) -> Result<ToolResult<SearchResults>> {
        self.search_engine.invoke(query, date_range).await
    }
}

#[async_trait]
impl BaseTool for SearchTool {
    fn name(&self) -> &str {
        "search"
    }

    fn tool_definitions(&self) -> &[ToolDefinition] {
        &self.definitions
    }

    async fn call_tool(&self, tool_name: &str, kwargs: ToolArguments) -> Result<ToolResult<Value>> {
        match tool_name {
            "search_web" => {
                let query = kwargs
                    .get("query")
                    .and_then(Value::as_str)
                    .filter(|query| !query.trim().is_empty())
                    .ok_or_else(|| anyhow!("工具参数[query]缺失"))?
                    .to_string();
                let date_range = kwargs
                    .get("date_range")
                    .and_then(Value::as_str)
                    .filter(|date_range| !date_range.trim().is_empty())
                    .map(str::to_owned);

                let result = self.search_web(query, date_range).await?;
                let data = result.data.map(serde_json::to_value).transpose()?;

                Ok(ToolResult {
                    success: result.success,
                    message: result.message,
                    data,
                })
            }
            _ => Err(anyhow!("工具[{tool_name}]未找到")),
        }
    }
}

#[cfg(test)]
mod tests {
    use std::sync::{Arc, Mutex};

    use async_trait::async_trait;
    use serde_json::{json, Map};

    use super::*;
    use crate::domain::models::SearchResultItem;

    type Calls = Arc<Mutex<Vec<(String, Option<String>)>>>;

    struct MockSearchEngine {
        calls: Calls,
    }

    #[async_trait]
    impl SearchEngine for MockSearchEngine {
        async fn invoke(
            &self,
            query: String,
            date_range: Option<String>,
        ) -> Result<ToolResult<SearchResults>> {
            self.calls
                .lock()
                .unwrap()
                .push((query.clone(), date_range.clone()));
            Ok(ToolResult {
                data: Some(SearchResults {
                    query,
                    date_range,
                    total_results: 1,
                    results: vec![SearchResultItem {
                        url: "https://example.com".to_string(),
                        title: "Example".to_string(),
                        snippet: "Search result".to_string(),
                    }],
                }),
                ..ToolResult::default()
            })
        }
    }

    fn search_tool() -> (SearchTool, Calls) {
        let calls = Arc::new(Mutex::new(Vec::new()));
        let search_engine = MockSearchEngine {
            calls: Arc::clone(&calls),
        };
        (SearchTool::new(Box::new(search_engine)), calls)
    }

    #[test]
    fn exposes_search_web_schema_like_python_tool_decorator() {
        let (tool, _) = search_tool();

        assert_eq!(tool.name(), "search");
        assert!(tool.has_tool("search_web"));
        assert!(!tool.has_tool("missing"));

        let tools = tool.get_tools();
        assert_eq!(tools.len(), 1);
        assert_eq!(tools[0]["type"], "function");
        assert_eq!(tools[0]["function"]["name"], "search_web");
        assert_eq!(
            tools[0]["function"]["parameters"]["properties"]["query"]["type"],
            "string"
        );
        assert_eq!(
            tools[0]["function"]["parameters"]["properties"]["date_range"]["enum"],
            json!([
                "all",
                "past_hour",
                "past_day",
                "past_week",
                "past_month",
                "past_year"
            ])
        );
        assert_eq!(
            tools[0]["function"]["parameters"]["required"],
            json!(["query"])
        );
    }

    #[tokio::test]
    async fn invokes_search_engine_with_query_and_date_range() {
        let (tool, calls) = search_tool();
        let mut kwargs = Map::new();
        kwargs.insert("query".to_string(), json!("北京 天气"));
        kwargs.insert("date_range".to_string(), json!("past_day"));
        kwargs.insert("extra".to_string(), json!("ignored"));

        let result = tool.invoke("search_web", kwargs).await.unwrap();

        assert!(result.success);
        assert_eq!(
            *calls.lock().unwrap(),
            vec![("北京 天气".to_string(), Some("past_day".to_string()))]
        );
        let data = result.data.unwrap();
        assert_eq!(data["query"], "北京 天气");
        assert_eq!(data["date_range"], "past_day");
        assert_eq!(data["total_results"], 1);
        assert_eq!(data["results"][0]["url"], "https://example.com");
        assert!(data.get("extra").is_none());
    }

    #[tokio::test]
    async fn omits_date_range_when_argument_is_absent() {
        let (tool, calls) = search_tool();
        let mut kwargs = Map::new();
        kwargs.insert("query".to_string(), json!("OpenAI news"));

        let result = tool.invoke("search_web", kwargs).await.unwrap();

        assert_eq!(
            *calls.lock().unwrap(),
            vec![("OpenAI news".to_string(), None)]
        );
        let data = result.data.unwrap();
        assert_eq!(data["query"], "OpenAI news");
        assert!(data["date_range"].is_null());
    }

    #[tokio::test]
    async fn rejects_missing_query() {
        let (tool, _) = search_tool();
        let error = tool.invoke("search_web", Map::new()).await.unwrap_err();

        assert_eq!(error.to_string(), "工具参数[query]缺失");
    }
}
