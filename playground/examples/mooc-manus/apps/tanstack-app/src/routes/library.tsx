import SectionPage from '@apps/frontend/section-page'
import { createFileRoute } from '@tanstack/react-router'

export const Route = createFileRoute('/library')({
  ssr: false,
  component: () => <SectionPage section="library" />,
})
