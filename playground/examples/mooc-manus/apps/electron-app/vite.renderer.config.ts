import tailwindcss from "@tailwindcss/vite";
import react from "@vitejs/plugin-react";
import { defineConfig } from "vite";

// https://vitejs.dev/config
export default defineConfig({
  // Prebundle generated CommonJS types; let Vite resolve the WASM asset URL.
  optimizeDeps: {
    include: ["@apps/frontend > shared_types/app.js", "@apps/frontend > shared_types/bincode/index.js", "@apps/frontend > shared > @boltffi/runtime", "@apps/frontend > @base-ui/react/scroll-area"],
    exclude: ["shared"],
  },
  build: {
    commonjsOptions: { include: [/node_modules/, /generated\/types\//] },
  },
  plugins: [react(), tailwindcss()],
  resolve: {
    preserveSymlinks: false,
    dedupe: ["react", "react-dom"],
  },
});
