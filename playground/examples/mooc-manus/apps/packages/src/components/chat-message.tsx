interface ChatMessageProps {
  className?: string;
  message: {
    type: string;
    role?: string;
  };
}

export function ChatMessage({ className, message }: ChatMessageProps) {
  // 1.消息类型为user时
  if (message.type === "user") {
    return <div className={className}>用户消息</div>;
  } else if (message.type === "assistant") {
    return <div className={className}>AI消息</div>;
  } else if (message.type === "tool") {
    return <div className={className}>工具消息</div>;
  } else if (message.type === "step") {
    return <div className={className}>步骤/子任务消息</div>;
  } else if (message.type === "attachments") {
    return <div className={className}>附件消息</div>;
  }

  return null;
}
