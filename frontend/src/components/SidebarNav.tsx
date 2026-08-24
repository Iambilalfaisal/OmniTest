"use client";

import Link from "next/link";
import { usePathname } from "next/navigation";
import OrbitMark from "@/components/OrbitMark";

const NAV_ITEMS = [
  { href: "/",        label: "New session" },
  { href: "/discover", label: "Discovery"  },
  { href: "/run",      label: "Run"        },
  { href: "/history",  label: "History"    },
  { href: "/reports",  label: "Reports"    },
];

export default function SidebarNav() {
  const pathname = usePathname();

  return (
    <div className="flex h-full flex-col gap-8 p-6">
      <div className="flex items-center gap-3">
        <OrbitMark />
        <div>
          <div className="eyebrow">AI QA / 01</div>
          <div className="text-lg font-semibold text-white">OmniTest</div>
        </div>
      </div>

      <div className="rounded-2xl border border-white/10 bg-white/[0.03] p-4 text-xs leading-5 text-slate-400">
        <div className="mb-2 flex items-center justify-between text-[10px] uppercase tracking-[0.2em] text-slate-500">
          <span>Control plane</span><span className="text-lime-300">Online</span>
        </div>
        Autonomous exploration, human-guided planning, evidence-rich verification.
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
                  ? "border-cyan-300/30 bg-cyan-300/10 text-white shadow-[inset_3px_0_0_#39e7d3]"
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
