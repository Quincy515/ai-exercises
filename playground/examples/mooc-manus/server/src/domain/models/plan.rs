use serde::{Deserialize, Serialize};
use uuid::Uuid;

/// 规划 / 任务执行的状态
/// Status for plan or step execution.
#[derive(Debug, Clone, Copy, Default, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "lowercase")]
pub enum ExecutionStatus {
    /// 空闲或等待中
    /// Idle or waiting.
    #[default]
    Pending,
    /// 执行中
    /// Running now.
    Running,
    /// 执行完成
    /// Finished successfully.
    Completed,
    /// 失败
    /// Finished with failure.
    Failed,
}

/// 计划中的每一个步骤 / 子任务
/// A step or subtask inside a plan.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(default)]
pub struct Step {
    /// 子任务 id
    /// Step id.
    pub id: String,
    /// 步骤的描述信息
    /// Step description.
    pub description: String,
    /// 子任务的执行状态
    /// Step execution status.
    pub status: ExecutionStatus,
    /// 结果
    /// Result text.
    pub result: Option<String>,
    /// 错误信息
    /// Error message.
    pub error: Option<String>,
    /// 是否执行成功
    /// Whether the step succeeded.
    pub success: bool,
    /// 附件列表信息
    /// Attachment list.
    pub attachments: Vec<String>,
}

impl Default for Step {
    fn default() -> Self {
        Self {
            id: Uuid::new_v4().to_string(),
            description: String::new(),
            status: ExecutionStatus::Pending,
            result: None,
            error: None,
            success: false,
            attachments: Vec::new(),
        }
    }
}

impl Step {
    pub fn new(description: impl Into<String>) -> Self {
        Self {
            description: description.into(),
            ..Self::default()
        }
    }

    /// 只读属性，返回步骤是否结束
    /// Returns whether the step has ended.
    pub fn done(&self) -> bool {
        matches!(
            self.status,
            ExecutionStatus::Completed | ExecutionStatus::Failed
        )
    }
}

/// 规划 Domain 模型，用于存储用户传递消息拆分出来的子任务 / 子步骤
/// Domain model that stores subtasks split from the user's message.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(default)]
pub struct Plan {
    /// 计划 id
    /// Plan id.
    pub id: String,
    /// 任务标题
    /// Task title.
    pub title: String,
    /// 任务目标
    /// Task goal.
    pub goal: String,
    /// 工作语言
    /// Working language.
    pub language: String,
    /// 步骤列表 / 子任务列表
    /// Step or subtask list.
    pub steps: Vec<Step>,
    /// AI 传递的消息
    /// Message passed by the AI.
    pub message: String,
    /// 规划的状态
    /// Plan execution status.
    pub status: ExecutionStatus,
    /// 错误信息
    /// Error message.
    pub error: Option<String>,
}

impl Default for Plan {
    fn default() -> Self {
        Self {
            id: Uuid::new_v4().to_string(),
            title: String::new(),
            goal: String::new(),
            language: String::new(),
            steps: Vec::new(),
            message: String::new(),
            status: ExecutionStatus::Pending,
            error: None,
        }
    }
}

impl Plan {
    pub fn new(title: impl Into<String>, goal: impl Into<String>) -> Self {
        Self {
            title: title.into(),
            goal: goal.into(),
            ..Self::default()
        }
    }

    /// 只读属性，用于判断计划是否结束
    /// Returns whether the plan has ended.
    pub fn done(&self) -> bool {
        matches!(
            self.status,
            ExecutionStatus::Completed | ExecutionStatus::Failed
        )
    }

    /// 获取需要执行的下一个步骤
    /// Returns the next step that still needs execution.
    pub fn get_next_step(&self) -> Option<&Step> {
        self.steps.iter().find(|step| !step.done())
    }
}

#[cfg(test)]
mod tests {
    use super::{ExecutionStatus, Plan, Step};
    use uuid::Uuid;

    #[test]
    fn creates_python_like_defaults() {
        let step = Step::default();
        assert!(Uuid::parse_str(&step.id).is_ok());
        assert_eq!(step.description, "");
        assert_eq!(step.status, ExecutionStatus::Pending);
        assert_eq!(step.result, None);
        assert_eq!(step.error, None);
        assert!(!step.success);
        assert!(step.attachments.is_empty());

        let plan = Plan::default();
        assert!(Uuid::parse_str(&plan.id).is_ok());
        assert_eq!(plan.title, "");
        assert_eq!(plan.goal, "");
        assert_eq!(plan.language, "");
        assert!(plan.steps.is_empty());
        assert_eq!(plan.message, "");
        assert_eq!(plan.status, ExecutionStatus::Pending);
        assert_eq!(plan.error, None);
    }

    #[test]
    fn reports_done_for_terminal_statuses() {
        let mut step = Step::default();
        assert!(!step.done());

        step.status = ExecutionStatus::Running;
        assert!(!step.done());

        step.status = ExecutionStatus::Completed;
        assert!(step.done());

        let mut plan = Plan::default();
        assert!(!plan.done());

        plan.status = ExecutionStatus::Failed;
        assert!(plan.done());
    }

    #[test]
    fn returns_first_step_that_is_not_done() {
        let completed = Step {
            status: ExecutionStatus::Completed,
            ..Step::new("completed")
        };
        let running = Step {
            status: ExecutionStatus::Running,
            ..Step::new("running")
        };
        let pending = Step::new("pending");
        let plan = Plan {
            steps: vec![completed, running, pending],
            ..Plan::default()
        };

        assert_eq!(
            plan.get_next_step().map(|step| step.description.as_str()),
            Some("running")
        );
    }

    #[test]
    fn serializes_status_as_python_enum_values() {
        assert_eq!(
            serde_json::to_string(&ExecutionStatus::Completed).unwrap(),
            "\"completed\""
        );
    }

    #[test]
    fn deserializes_planner_output_with_python_defaults() {
        let plan: Plan = serde_json::from_value(serde_json::json!({
            "title": "发布应用",
            "goal": "完成应用发布",
            "language": "zh",
            "message": "计划已生成",
            "steps": [
                {
                    "id": "1",
                    "description": "执行发布检查"
                }
            ]
        }))
        .unwrap();

        assert_eq!(plan.status, ExecutionStatus::Pending);
        assert_eq!(plan.error, None);
        assert_eq!(plan.steps[0].status, ExecutionStatus::Pending);
        assert_eq!(plan.steps[0].result, None);
        assert_eq!(plan.steps[0].error, None);
        assert!(!plan.steps[0].success);
        assert!(plan.steps[0].attachments.is_empty());
    }
}
