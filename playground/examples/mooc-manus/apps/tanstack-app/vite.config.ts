import tailwindcss from "@tailwindcss/vite";
import { devtools } from "@tanstack/devtools-vite";

import { tanstackStart } from "@tanstack/react-start/plugin/vite";

import viteReact from "@vitejs/plugin-react";
import { defineConfig } from "vite";
import { apiDevProxy } from "../vite.api.mts";

const config = defineConfig({
	// Crux typegen emits CommonJS into a linked workspace package.
	optimizeDeps: {
		include: ["@apps/frontend > shared_types/app.js", "@apps/frontend > shared_types/bincode/index.js", "@apps/frontend > shared > @boltffi/runtime", "@apps/frontend > @base-ui/react/scroll-area"],
		exclude: ["shared"],
	},
	resolve: { tsconfigPaths: true, dedupe: ["react", "react-dom"] },
	// TanStack Devtools already pipes console output between browser and server.
	server: { forwardConsole: false, proxy: apiDevProxy },
	plugins: [devtools(), tailwindcss(), tanstackStart({ spa: { enabled: true } }), viteReact()],
});

export default config;
