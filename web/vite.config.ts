import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";

export default defineConfig({
  plugins: [react()],
  // onnxruntime-web finds its .wasm next to its own module; pre-bundling breaks that
  optimizeDeps: { exclude: ["onnxruntime-web"] },
});
