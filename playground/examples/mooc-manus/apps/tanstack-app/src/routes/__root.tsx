import { TanStackDevtools } from '@tanstack/react-devtools'
import AppLayout from '@apps/frontend/layout'
import type { QueryClient } from '@tanstack/react-query'
import {
  createRootRouteWithContext,
  ClientOnly,
  HeadContent,
  Outlet,
  Scripts,
  useNavigate,
  useParams,
} from '@tanstack/react-router'
import { TanStackRouterDevtoolsPanel } from '@tanstack/react-router-devtools'
import TanStackQueryDevtools from '../integrations/tanstack-query/devtools'
import appCss from '../styles.css?url'

interface MyRouterContext {
  queryClient: QueryClient
}

export const Route = createRootRouteWithContext<MyRouterContext>()({
  head: () => ({
    meta: [
      {
        charSet: 'utf-8',
      },
      {
        name: 'viewport',
        content: 'width=device-width, initial-scale=1',
      },
      {
        title: 'Mooc Manus',
      },
    ],
    links: [
      {
        rel: 'stylesheet',
        href: appCss,
      },
    ],
  }),
  shellComponent: RootDocument,
  component: () => <ClientOnly><AppShell /></ClientOnly>,
})

function AppShell() {
  const { id: sessionId } = useParams({ strict: false })
  const navigate = useNavigate()

  return (
    <AppLayout
      sessionId={sessionId}
      onNewSession={() => void navigate({ to: '/' })}
      onSelectSession={(id) => void navigate({ to: '/sessions/$id', params: { id } })}
    >
      <Outlet />
    </AppLayout>
  )
}

function RootDocument({ children }: { children: React.ReactNode }) {
  return (
    <html lang="en">
      <head>
        <HeadContent />
      </head>
      <body>
        {children}
        <TanStackDevtools
          config={{
            position: 'bottom-right',
          }}
          plugins={[
            {
              name: 'Tanstack Router',
              render: <TanStackRouterDevtoolsPanel />,
            },
            TanStackQueryDevtools,
          ]}
        />
        <Scripts />
      </body>
    </html>
  )
}
