import type { Metadata } from "next";
import Link from "next/link";
import "./globals.css";
import SidebarNav from "@/components/SidebarNav";

export const metadata: Metadata = {
  title: "OmniTest",
  description: "Autonomous AI QA platform with a live execution canvas and evidence-driven reporting",
};

export default function RootLayout({ children }: { children: React.ReactNode }) {
  return (
    <html lang="en">
      <body className="min-h-screen bg-slate-950 text-slate-100 antialiased">
        <div className="flex min-h-screen">
          <aside className="glass-panel sticky top-0 hidden h-screen w-64 shrink-0 rounded-none border-b-0 border-l-0 border-t-0 md:flex md:flex-col">
            <SidebarNav />
          </aside>

          <div className="flex min-h-screen flex-1 flex-col">
            {/* Mobile top bar — the sidebar takes over from md: up */}
            <header className="glass-panel m-4 flex items-center justify-between rounded-3xl px-4 py-3 md:hidden">
              <div className="flex items-center gap-2">
                <div className="flex h-8 w-8 items-center justify-center rounded-xl bg-gradient-to-br from-cyan-500 to-violet-500 text-sm font-bold text-white">
                  O
                </div>
                <span className="text-sm font-semibold text-white">OmniTest</span>
              </div>
              <nav className="flex items-center gap-1 text-xs">
                <Link href="/" className="rounded-full border border-white/10 bg-slate-950/50 px-3 py-1.5 text-slate-200">
                  Home
                </Link>
                <Link href="/history" className="rounded-full border border-white/10 bg-slate-950/50 px-3 py-1.5 text-slate-200">
                  History
                </Link>
                <Link href="/reports" className="rounded-full border border-white/10 bg-slate-950/50 px-3 py-1.5 text-slate-200">
                  Reports
                </Link>
              </nav>
            </header>

            <div className="hidden justify-end px-4 pt-6 md:flex md:px-6 lg:px-8">
              <Link
                href="/"
                className="rounded-full bg-gradient-to-r from-cyan-500 to-violet-500 px-4 py-2 text-xs font-semibold text-white shadow-lg shadow-cyan-500/20 transition hover:brightness-110"
              >
                + New session
              </Link>
            </div>

            <div className="flex-1 px-4 pb-10 pt-4 md:px-6 md:pt-2 lg:px-8">
              <main>{children}</main>
            </div>
          </div>
        </div>
      </body>
    </html>
  );
}
