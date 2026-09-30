import { ChatInput } from "./components/chat-input";
import { ChatMessage } from "./components/chat-message";
import { PlanPanel } from "./components/plan-panel";
import { SessionHeader } from "./components/session-header";

function Chat() {
  // 本课使用模拟消息列表，真实事件数据在后续接入。
  const messages = [
    { id: 1, type: "user" },
    { id: 2, type: "attachments", role: "user" },
    { id: 3, type: "assistant" },
    { id: 4, type: "tool" },
    { id: 5, type: "step" },
    { id: 6, type: "assistant" },
    { id: 7, type: "attachments", role: "assistant" },
  ];

  return (
    <section
      aria-label="会话任务详情"
      className="relative flex min-h-full min-w-0 flex-1 flex-col px-4"
    >
      {/* 顶部标题&操作按钮 */}
      <SessionHeader />
      {/* 中间内容 */}
      <div className="mx-auto flex w-full min-w-0 max-w-[768px] flex-1 flex-col">
        {/* 对话列表 */}
        <div className="flex w-full flex-1 flex-col gap-3 pt-3 pb-10">
          {messages.map((message) => (
            <ChatMessage key={message.id} message={message} />
          ))}
        </div>
        {/* 底部输入框&任务清单 */}
        <div className="sticky bottom-0 mt-auto bg-chat-background">
          {/* 规划列表 */}
          <PlanPanel className="mb-2" />
          {/* 输入框 */}
          <ChatInput className="mb-4" />
        </div>
      </div>
    </section>
  );
}
export default Chat;
