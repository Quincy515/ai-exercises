import Chat from '@apps/frontend/chat'
import { createFileRoute } from '@tanstack/react-router'

export const Route = createFileRoute('/sessions/$id')({
  ssr: false,
  component: Chat,
})
