const sections = {
  schedules: { title: "定时任务", description: "在这里查看和管理定时任务。" },
  library: { title: "资料库", description: "集中查看聊天中使用的资料。" },
};

// 先接通模块导航，后续再补充对应业务内容。
export default function SectionPage({
  section,
}: {
  section: keyof typeof sections;
}) {
  const { title, description } = sections[section];
  return (
    <section className="flex flex-1 flex-col items-center justify-center gap-3 px-6 text-center">
      <h1 className="text-2xl font-semibold">{title}</h1>
      <p className="text-sm text-muted-foreground">{description}</p>
    </section>
  );
}
