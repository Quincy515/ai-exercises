import Novnc from '@apps/frontend/novnc'
import { createFileRoute } from '@tanstack/react-router'

// 下划线让此页直接挂到根路由，避开会话详情组件的嵌套。
export const Route = createFileRoute('/sessions/$id_/novnc')({
  ssr: false,
  component: Novnc,
})
