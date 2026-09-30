import { cn } from "../lib/utils";
import { Button } from "./ui/button";

interface SuggestedQuestionsProps {
  className?: string;
}

export function SuggestedQuestions({ className }: SuggestedQuestionsProps) {
  // 本课展示推荐问题，点击后的业务交互在后续接入。
  const questions = [
    "与最高的建筑相比，埃菲尔铁塔有多高？",
    "GitHub上最热门的存储库有哪些？",
    "如果看待中国的外卖大战？",
    "超加工食品与健康有关吗？超加工食品的历史怎样？",
  ];

  return (
    <div className={cn("flex flex-wrap gap-2", className)}>
      {questions.map((question) => (
        <Button
          key={question}
          type="button"
          variant="outline"
          className="h-auto min-h-8 max-w-full cursor-pointer justify-start py-1.5 text-left whitespace-normal"
        >
          {question}
        </Button>
      ))}
    </div>
  );
}
