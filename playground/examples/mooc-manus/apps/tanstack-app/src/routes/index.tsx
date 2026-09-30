import NewSession from '@apps/frontend/new-session'
import { createFileRoute } from '@tanstack/react-router'

export const Route = createFileRoute('/')({
  ssr: false,
  component: NewSession,
})
