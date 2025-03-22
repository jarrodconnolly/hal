import { defineConfig } from 'vite';

export default defineConfig({
  root: 'src', // Serve from src/ (index.html lives here)
  server: {
    port: 5173,
    strictPort: true, // Tauri expects this
  },
  build: {
    outDir: '../dist', // Tauri’s distDir
    target: 'esnext', // Modern JS for Tauri WebView
    minify: 'esbuild',
    sourcemap: true, // Debug HAL’s spans
  },
});