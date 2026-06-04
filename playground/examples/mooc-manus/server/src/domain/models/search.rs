use serde::{Deserialize, Serialize};

/// 搜索结果条目数据类型
#[derive(Debug, Clone, Default, Serialize, Deserialize, PartialEq, Eq)]
#[serde(default)]
pub struct SearchResultItem {
    /// 搜索条目 URL 连接
    pub url: String,
    /// 搜索条目标题
    pub title: String,
    /// 搜索条目再要信息
    pub snippet: String,
}

/// 搜索结果数据类型
#[derive(Debug, Clone, Default, Serialize, Deserialize, PartialEq, Eq)]
#[serde(default)]
pub struct SearchResults {
    /// 搜索查询
    pub query: String,
    /// 日期筛选范围
    pub date_range: Option<String>,
    /// 搜索结果条数
    pub total_results: usize,
    /// 搜索结果
    pub results: Vec<SearchResultItem>,
}
