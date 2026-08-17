import type { Metadata } from "next";
import Link from "next/link";
import "./globals.css";

export const metadata: Metadata = {
  title: "OmniTest",
  description: "Autonomous AI QA platform with a live map-reduce execution canvas",
};

export default function RootLayout({ children }: { children: React.ReactNode }) {
  return (
    <html lang="en">
      <body className="min-h-screen bg-neutral-950 text-neutral-100">
        <nav className="flex items-center gap-6 border-b border-neutral-800 px-6 py-4">
          <span className="font-semibold tracking-tight">OmniTest</span>
          <Link href="/run" className="text-sm text-neutral-400 hover:text-neutral-100">
            Run
          </Link>
          <Link href="/reports" className="text-sm text-neutral-400 hover:text-neutral-100">
            Reports
          </Link>
        </nav>
        <main className="p-6">{children}</main>
      </body>
    </html>
  );
}
