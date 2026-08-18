import type { Metadata } from "next";
import Link from "next/link";
import "./globals.css";

export const metadata: Metadata = {
  title: "OmniTest",
  description: "Autonomous AI QA platform with a live execution canvas and evidence-driven reporting",
};

export default function RootLayout({ children }: { children: React.ReactNode }) {
  return (
    <html lang="en">
      <body className="min-h-screen bg-slate-950 text-slate-100 antialiased">
        <div className="mx-auto max-w-7xl px-4 pb-10 pt-6 md:px-6 lg:px-8">
          <header className="mb-8 rounded-3xl border border-white/10 bg-slate-900/80 px-5 py-4 shadow-2xl shadow-slate-950/40 backdrop-blur-xl">
            <div className="flex flex-col gap-4 md:flex-row md:items-center md:justify-between">
              <div className="flex items-center gap-3">
                <div className="flex h-10 w-10 items-center justify-center rounded-2xl bg-gradient-to-br from-cyan-500 to-violet-500 text-base font-bold text-white shadow-lg shadow-cyan-500/20">
                  O
                </div>
                <div>
                  <div className="text-xs uppercase tracking-[0.28em] text-cyan-300/80">AI QA</div>
                  <div className="text-lg font-semibold text-white">OmniTest</div>
                </div>
              </div>

              <nav className="flex items-center gap-2 text-sm">
                <Link href="/" className="rounded-full border border-white/10 bg-slate-950/50 px-4 py-2 text-slate-200 transition hover:border-cyan-500/30 hover:text-white">
                  Home
                </Link>
                <Link href="/run" className="rounded-full border border-white/10 bg-slate-950/50 px-4 py-2 text-slate-200 transition hover:border-cyan-500/30 hover:text-white">
                  Live run
                </Link>
                <Link href="/reports" className="rounded-full border border-white/10 bg-slate-950/50 px-4 py-2 text-slate-200 transition hover:border-cyan-500/30 hover:text-white">
                  Reports
                </Link>
              </nav>
            </div>
          </header>

          <main>{children}</main>
        </div>
      </body>
    </html>
  );
}
