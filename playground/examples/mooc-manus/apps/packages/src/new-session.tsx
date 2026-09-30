import { ChatInput } from "./components/chat-input";
import { SuggestedQuestions } from "./components/suggested-questions";

export default function NewSession() {
  return (
    // 顶部header由共享布局提供，这里组合中间对话框。
    <section
      aria-label="新建会话任务"
      className="relative mx-auto mt-[20vh] w-full min-w-0 max-w-[768px] shrink-0 px-4 pb-6"
    >
      {/* 对话提示内容 */}
      <h1 className="mb-4 text-[32px] font-bold">
        <span className="block text-gray-700">您好, 慕学员</span>
        <span className="block text-gray-500">我能为您做什么?</span>
      </h1>
      {/* 对话框 */}
      <ChatInput className="mb-4" />
      {/* 推荐对话内容 */}
      <SuggestedQuestions />
    </section>
  );
}
