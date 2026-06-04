use std::{
    collections::HashMap,
    sync::Arc,
    time::{Duration, SystemTime, UNIX_EPOCH},
};

use anyhow::{anyhow, Result};
use async_trait::async_trait;
use regex::Regex;
use reqwest::{
    cookie::Jar,
    header::{HeaderMap, HeaderName, HeaderValue},
    redirect::Policy,
};
use scraper::{Html, Selector};
use tracing::error;

use crate::domain::{
    external::SearchEngine,
    models::{SearchResultItem, SearchResults, ToolResult},
};

/// Bing 搜索引擎
pub struct BingSearchEngine {
    pub base_url: String,
    pub headers: HashMap<String, String>,
    pub cookies: Arc<Jar>,
}

#[allow(clippy::new_without_default)]
impl BingSearchEngine {
    /// 构造函数，初始化 Bing 搜索引擎的相关信息
    pub fn new() -> Self {
        Self {
            base_url: "https://www.bing.com/search".to_string(),
            headers: serde_json::from_str(r#"
                {
                "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/122.0.0.0 Safari/537.36 Edg/122.0.0.0",
                "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8",
                "Accept-Language": "en-US,en;q=0.5",
                "Accept-Encoding": "gzip, deflate",
                "Connection": "keep-alive",
                "Upgrade-Insecure-Requests": "1"
                }
            "#).unwrap(),
            cookies: Arc::new(Jar::default()),
        }
    }
}

#[async_trait]
impl SearchEngine for BingSearchEngine {
    /// 根据传递的 query+date_range 调用 Bing 搜索获取搜索内容
    async fn invoke(
        &self,
        query: String,
        date_range: Option<String>,
    ) -> Result<ToolResult<SearchResults>> {
        // 1. 构建请求参数
        let mut params = HashMap::from([
            ("q".to_string(), query.clone()),
            // 固定英文搜索市场，避免 Bing 根据出口 IP 或 Cookie 推断出不稳定的区域
            ("mkt".to_string(), "en-US".to_string()),
            ("setlang".to_string(), "en-US".to_string()),
            ("cc".to_string(), "US".to_string()),
        ]);

        // 2. 判断 date_range 是否存在并提取真实检索数据
        if let Some(date_range) = date_range
            .as_deref()
            .filter(|date_range| *date_range != "all")
        {
            // 3. 获取当前日期的天数距离 1970-01-01 的天数
            let days_since_epoch = SystemTime::now()
                .duration_since(UNIX_EPOCH)
                .unwrap_or_default()
                .as_secs()
                / (24 * 60 * 60);

            // 4. 创建日期检索数据类型映射
            let date_mapping = HashMap::from([
                ("past_hour", r#"ex1%3a"ez1""#.to_string()),
                ("past_day", r#"ex1%3a"ez1""#.to_string()),
                ("past_week", r#"ex1%3a"ez2""#.to_string()),
                ("past_month", r#"ex1%3a"ez3""#.to_string()),
                (
                    "past_year",
                    format!(
                        r#"ex1%3a"ez5_{}_{}""#,
                        days_since_epoch.saturating_sub(365),
                        days_since_epoch
                    ),
                ),
            ]);

            // 5. 判断是否传递了 date_range 补全 params 参数
            if let Some(filters) = date_mapping.get(date_range) {
                params.insert("filters".to_string(), filters.clone());
            }
        }

        let search_result: Result<SearchResults> = async {
            // 6. 使用 reqwest 创建异步客户端
            let headers = self.headers.iter().try_fold(
                HeaderMap::new(),
                |mut headers, (name, value)| -> Result<HeaderMap> {
                    headers.insert(
                        HeaderName::from_bytes(name.as_bytes())?,
                        HeaderValue::from_str(value)?,
                    );
                    Ok(headers)
                },
            )?;
            let client = reqwest::Client::builder()
                .default_headers(headers)
                .cookie_provider(Arc::clone(&self.cookies))
                .timeout(Duration::from_secs(60))
                .redirect(Policy::limited(10))
                .build()?;

            // 7. 调用客户端发起请求
            let response = client
                .get(&self.base_url)
                .query(&params)
                .send()
                .await?
                .error_for_status()?;

            // 8. reqwest 会将响应 Cookie 自动更新到共享的 Cookie Jar

            // 9. 使用 scraper 解析 HTML 内容
            let response_text = response.text().await?;
            let soup = Html::parse_document(&response_text);

            // 10. 定义搜索结果并解析 li.b_algo 对应的 DOM 元素
            let mut search_results = Vec::new();
            let result_item_selector = Selector::parse("li.b_algo")
                .map_err(|err| anyhow!("解析 CSS 选择器 [li.b_algo] 失败: {err}"))?;
            let title_selector = Selector::parse("h2 a")
                .map_err(|err| anyhow!("解析 CSS 选择器 [h2 a] 失败: {err}"))?;
            let a_selector = Selector::parse("a")
                .map_err(|err| anyhow!("解析 CSS 选择器 [a] 失败: {err}"))?;
            let snippet_selector = Selector::parse(
                "p.b_lineclamp, div.b_lineclamp, p.b_descript, div.b_descript, p.b_caption, div.b_caption",
            )
            .map_err(|err| anyhow!("解析摘要 CSS 选择器失败: {err}"))?;
            let p_selector = Selector::parse("p")
                .map_err(|err| anyhow!("解析 CSS 选择器 [p] 失败: {err}"))?;
            let sentence_regex = Regex::new(r"[.!?\n。！]")?;
            let count_selector = Selector::parse(
                "span.sb_count, div.sb_count, p.sb_count, span.b_focusTextMedium, div.b_focusTextMedium, p.b_focusTextMedium",
            )
            .map_err(|err| anyhow!("解析结果数量 CSS 选择器失败: {err}"))?;

            // 11. 循环遍历所有匹配的 DOM
            for item in soup.select(&result_item_selector) {
                // 12. 定义变量存储数据标题+url 连接
                let mut title = String::new();
                let mut url = String::new();

                // 13. 解析搜索结果中的 h2 并提取 title 标题与 URL 链接
                if let Some(a_tag) = item.select(&title_selector).next() {
                    title = a_tag.text().map(str::trim).collect::<String>();
                    url = a_tag.value().attr("href").unwrap_or_default().to_string();
                }

                // 14. 判断标题如果不存在提取该 DOM 下的 a 标签
                if title.is_empty() {
                    for a_tag in item.select(&a_selector) {
                        // 15. 提取标签中的文本并判断文本的长度是否超过 10 且不以 http 开头
                        let text = a_tag.text().map(str::trim).collect::<String>();
                        if text.chars().count() > 10 && !text.starts_with("http") {
                            title = text;
                            url = a_tag.value().attr("href").unwrap_or_default().to_string();
                            break;
                        }
                    }
                }

                // 16. 如果两种查询方式都找不到 title 则跳过这次数据
                if title.is_empty() {
                    continue;
                }

                // 17. 提取检索数据的摘要信息
                let mut snippet = item
                    .select(&snippet_selector)
                    .next()
                    .map(|element| element.text().map(str::trim).collect::<String>())
                    .unwrap_or_default();

                // 18. 如果未找到摘要则查询所有 p 标签（段落标签）
                if snippet.is_empty() {
                    for p in item.select(&p_selector) {
                        let text = p.text().map(str::trim).collect::<String>();
                        if text.chars().count() > 20 {
                            snippet = text;
                            break;
                        }
                    }
                }

                // 19. 如果还找不到摘要数据，则提取选项中的所有文本并使用常见的分隔符分割
                if snippet.is_empty() {
                    let all_text = item.text().map(str::trim).collect::<String>();

                    // 20. 将所有文本分割成对应的句子，并循环遍历取出长度 > 20 的句子
                    for sentence in sentence_regex.split(&all_text) {
                        let clean_sentence = sentence.trim();
                        if clean_sentence.chars().count() > 20 && clean_sentence != title {
                            snippet = clean_sentence.to_string();
                            break;
                        }
                    }
                }

                // 21. 补全相对路径的 URL 链接与缺失协议的部分
                if !url.is_empty() && !url.starts_with("http") {
                    if url.starts_with("//") {
                        url = format!("https:{url}");
                    } else if url.starts_with('/') {
                        url = format!("https://www.bing.com{url}");
                    }
                }

                // 22. 如果标题存在则添加数据
                search_results.push(SearchResultItem {
                    title,
                    url,
                    snippet,
                });
            }

            // 23. scraper 的选择器和文本提取不会抛出单条结果解析异常，无法解析的条目已直接跳过

            // 24. 提取整个页面的内容并查找 `results` 对应的文本
            let mut total_results = 0;
            let total_results_regex = Regex::new(r"([\d,]+)\s*results")?;
            for stat in soup.root_element().text() {
                // 25. 匹配出对应的数字分组
                if let Some(captures) = total_results_regex.captures(stat) {
                    // 26. 取出匹配的分组内容，去除逗号后转换为整型
                    if let Some(count) = captures
                        .get(1)
                        .and_then(|value| value.as_str().replace(',', "").parse::<usize>().ok())
                    {
                        total_results = count;
                        break;
                    }
                }
            }

            // 27. 如果使用正则匹配找不到 results（有可能是页面结构不一致）则使用新逻辑
            if total_results == 0 {
                // 28. 使用类元素查找器
                for element in soup.select(&count_selector) {
                    // 29. 提取 DOM 的文本并获取数字
                    let text = element.text().map(str::trim).collect::<String>();
                    if let Some(count) = total_results_regex
                        .captures(&text)
                        .and_then(|captures| captures.get(1))
                        .and_then(|value| value.as_str().replace(',', "").parse::<usize>().ok())
                    {
                        total_results = count;
                        break;
                    }
                }
            }

            // 30. 返回搜索结果
            Ok(SearchResults {
                query: query.clone(),
                date_range: date_range.clone(),
                total_results,
                results: search_results,
            })
        }
        .await;

        match search_result {
            Ok(results) => Ok(ToolResult {
                data: Some(results),
                ..ToolResult::default()
            }),
            Err(err) => {
                // 31. 记录日志并返回错误工具调用结果
                let message = format!("Bing搜索出错: {err}");
                error!("{message}");
                Ok(ToolResult {
                    success: false,
                    message: Some(message),
                    data: Some(SearchResults {
                        query,
                        date_range,
                        total_results: 0,
                        results: Vec::new(),
                    }),
                })
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    #[ignore = "requires live Bing access"]
    async fn searches_live_bing() -> Result<()> {
        let search_engine = BingSearchEngine::new();
        let result = search_engine
            .invoke("小米股价".to_string(), Some("past_day".to_string()))
            .await?;

        println!("{result:#?}");
        assert!(
            result.success,
            "{}",
            result.message.as_deref().unwrap_or("Bing 搜索失败")
        );

        let results = result.data.expect("Bing 搜索结果缺少 data");
        for item in &results.results {
            println!("{item:#?}");
        }

        assert!(!results.results.is_empty(), "Bing 搜索未解析到结果条目");
        assert!(
            results
                .results
                .iter()
                .any(|item| item.title.contains("小米") || item.snippet.contains("小米")),
            "Bing 搜索结果与查询词不相关"
        );
        Ok(())
    }
}
