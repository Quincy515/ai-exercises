import SectionPage from '@apps/frontend/section-page'
import { createFileRoute } from '@tanstack/react-router'

export const Route = createFileRoute('/schedules')({
  ssr: false,
  component: () => <SectionPage section="schedules" />,
})
