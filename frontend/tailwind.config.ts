import type { Config } from "tailwindcss";

const config: Config = {
  content: ["./src/app/**/*.{ts,tsx}", "./src/components/**/*.{ts,tsx}"],
  theme: {
    extend: {
      colors: {
        pass: "#22c55e",
        fail: "#ef4444",
        pending: "#facc15",
        running: "#38bdf8",
      },
      boxShadow: {
        glow: "0 0 0 1px rgba(34,211,238,0.25), 0 20px 50px rgba(14,165,233,0.15)",
      },
    },
  },
  plugins: [],
};

export default config;
