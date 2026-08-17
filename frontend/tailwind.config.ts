import type { Config } from "tailwindcss";

const config: Config = {
  content: ["./src/app/**/*.{ts,tsx}", "./src/components/**/*.{ts,tsx}"],
  theme: {
    extend: {
      colors: {
        pass: "#22c55e",
        fail: "#ef4444",
        pending: "#eab308",
        running: "#3b82f6",
      },
    },
  },
  plugins: [],
};

export default config;
