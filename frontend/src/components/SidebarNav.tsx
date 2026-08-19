"use client";

import Link from "next/link";
import { usePathname } from "next/navigation";

const NAV_ITEMS = [
  { href: "/", label: "New session" },
  { href: "/history", label: "History" },
  { href: "/reports", label: "Reports" },
];

export default function SidebarNav() {
  const pathname = usePathname();

  return (
    <div className="flex h-full flex-col gap-8 p-5">
      <div className="flex items-center gap-3">
        <div className="flex h-10 w-10 items-center justify-center rounded-2xl bg-gradient-to-br from-cyan-500 to-violet-500 text-base font-bold text-white shadow-lg shadow-cyan-500/20">
          O
        </div>
        <div>
          <div className="text-xs uppercase tracking-[0.28em] text-cyan-300/80">AI QA</div>
          <div className="text-lg font-semibold text-white">OmniTest</div>
        </div>
      </div>

      <nav className="flex flex-col gap-1.5">
        {NAV_ITEMS.map((item) => {
          const active = item.href === "/" ? pathname === "/" : pathname?.startsWith(item.href);
          return (
            <Link
              key={item.href}
              href={item.href}
              className={`rounded-2xl border px-4 py-2.5 text-sm font-medium transition ${
                active
                  ? "border-cyan-400/30 bg-cyan-500/10 text-white"
                  : "border-transparent text-slate-300 hover:border-white/10 hover:bg-white/5 hover:text-white"
              }`}
            >
              {item.label}
            </Link>
          );
        })}
      </nav>
    </div>
  );
}
