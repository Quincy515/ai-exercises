import tailwindcss from "@tailwindcss/vite";
import { devtools } from "@tanstack/devtools-vite";

import { tanstackStart } from "@tanstack/react-start/plugin/vite";

import viteReact from "@vitejs/plugin-react";
import { defineConfig } from "vite";

const config = defineConfig({
	resolve: { tsconfigPaths: true, dedupe: ["react", "react-dom"] },
	// TanStack Devtools already pipes console output between browser and server.
	server: { forwardConsole: false },
	plugins: [devtools(), tailwindcss(), tanstackStart(), viteReact()],
});

export default config;
