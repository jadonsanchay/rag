import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";

export default defineConfig({
  plugins: [react()],
  server: {
    port: 5173,
    // Proxy the API so the browser sees one origin in dev. Streaming needs
    // buffering off, or the proxy holds SSE frames until the response ends.
    // No path rewrite: the backend serves these same routes under /api itself,
    // so dev and the production single-container deploy hit identical paths.
    proxy: {
      "/api": {
        target: "http://127.0.0.1:8000",
        changeOrigin: true,
      },
    },
  },
});
