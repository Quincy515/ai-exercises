import { ChatInput } from "./components/chat-input";
import { PlanPanel } from "./components/plan-panel";
import { SessionHeader } from "./components/session-header";

function Chat() {
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
        <div className="pb-10">
          <p>中间对话内容列表</p>
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
