import Chat from "@apps/frontend/chat";
import AppLayout from "@apps/frontend/layout";
import {
  createHashHistory,
  createRootRoute,
  createRoute,
  createRouter,
  Outlet,
  RouterProvider,
  useNavigate,
  useParams,
  useLocation,
} from "@tanstack/react-router";
import { createRoot } from "react-dom/client";

function AppShell() {
  const { id } = useParams({ strict: false });
  const navigate = useNavigate();
  const pathname = useLocation({ select: (location) => location.pathname });

  return (
    <AppLayout
      showChatHeader={pathname === "/"}
      sessionId={id}
      onNewSession={() => void navigate({ to: "/" })}
      onSelectSession={(id) =>
        void navigate({ to: "/sessions/$id", params: { id } })
      }
    >
      <Outlet />
    </AppLayout>
  );
}

const rootRoute = createRootRoute({ component: AppShell });
const indexRoute = createRoute({
  getParentRoute: () => rootRoute,
  path: "/",
  component: Chat,
});
const sessionRoute = createRoute({
  getParentRoute: () => rootRoute,
  path: "/sessions/$id",
  component: Chat,
});

// Hash 路由让打包后的 file:// 页面也能切换会话并刷新。
const router = createRouter({
  routeTree: rootRoute.addChildren([indexRoute, sessionRoute]),
  history: createHashHistory(),
});

declare module "@tanstack/react-router" {
  interface Register {
    router: typeof router;
  }
}

const root = createRoot(document.getElementById("root")!);
root.render(<RouterProvider router={router} />);
