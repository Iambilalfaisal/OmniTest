import type { Metadata } from "next";
import Link from "next/link";
import "./globals.css";
import SidebarShell from "@/components/SidebarShell";
import OrbitMark from "@/components/OrbitMark";

export const metadata: Metadata = {
  title: "OmniTest",
  description: "Autonomous AI QA platform with a live execution canvas and evidence-driven reporting",
};

export default function RootLayout({ children }: { children: React.ReactNode }) {
  return (
    <html lang="en">
      <body className="min-h-screen text-slate-100 antialiased">
        <div className="flex min-h-screen">
          <SidebarShell />

          <div className="flex min-h-screen flex-1 flex-col">
            {/* Mobile top bar — the sidebar takes over from md: up */}
            <header className="glass-panel m-3 flex items-center justify-between rounded-2xl px-4 py-3 md:hidden">
              <div className="flex items-center gap-2">
                <OrbitMark size="sm" />
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

            <div className="hidden items-center justify-between px-6 pt-6 md:flex lg:px-8">
              <div className="flex items-center gap-2 text-xs text-slate-500"><span className="h-1.5 w-1.5 rounded-full bg-lime-300 shadow-[0_0_10px_#c5f36a]" />System nominal</div>
              <Link
                href="/"
                className="rounded-xl border border-cyan-300/30 bg-cyan-300/10 px-4 py-2 text-xs font-semibold text-cyan-100 shadow-lg shadow-cyan-500/10 transition hover:bg-cyan-300/20"
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
