"use client";

import Link from "next/link";
import { usePathname } from "next/navigation";
import { useEffect, useState } from "react";
import OrbitMark from "@/components/OrbitMark";

const NAV_ITEMS = [
  { href: "/",         label: "New session", icon: "✦" },
  { href: "/discover", label: "Discovery",   icon: "◎" },
  { href: "/run",      label: "Run",         icon: "▶" },
  { href: "/history",  label: "History",     icon: "≡" },
  { href: "/reports",  label: "Reports",     icon: "◈" },
];

const STORAGE_KEY = "omnitest_sidebar_collapsed";

export default function SidebarShell() {
  const pathname = usePathname();
  const [collapsed, setCollapsed] = useState(false);
  const [mounted, setMounted] = useState(false);

  // Hydrate from localStorage after mount to avoid SSR mismatch
  useEffect(() => {
    const stored = localStorage.getItem(STORAGE_KEY);
    if (stored === "1") setCollapsed(true);
    setMounted(true);
  }, []);

  function toggle() {
    setCollapsed((c) => {
      const next = !c;
      localStorage.setItem(STORAGE_KEY, next ? "1" : "0");
      return next;
    });
  }

  // Avoid layout flash before localStorage is read
  if (!mounted) return <div className="glass-panel hidden h-screen w-72 shrink-0 rounded-none border-b-0 border-l-0 border-t-0 md:block" />;

  return (
    <aside
      className={[
        "glass-panel sticky top-0 hidden h-screen shrink-0 flex-col rounded-none border-b-0 border-l-0 border-t-0 md:flex",
        "transition-[width] duration-300 ease-in-out",
        collapsed ? "w-[4.5rem]" : "w-72",
      ].join(" ")}
    >
      <div className="flex h-full flex-col gap-6 overflow-hidden p-4">

        {/* Logo row */}
        <div className={`flex items-center gap-3 ${collapsed ? "justify-center" : ""}`}>
          <div className="shrink-0">
            <OrbitMark />
          </div>
          {!collapsed && (
            <div className="min-w-0 overflow-hidden">
              <div className="eyebrow truncate">AI QA / 01</div>
              <div className="truncate text-lg font-semibold text-white">OmniTest</div>
            </div>
          )}
        </div>

        {/* Status pill — hide when collapsed */}
        {!collapsed && (
          <div className="rounded-2xl border border-white/10 bg-white/[0.03] p-4 text-xs leading-5 text-slate-400">
            <div className="mb-2 flex items-center justify-between text-[10px] uppercase tracking-[0.2em] text-slate-500">
              <span>Control plane</span>
              <span className="text-lime-300">Online</span>
            </div>
            Autonomous exploration, human-guided planning, evidence-rich verification.
          </div>
        )}

        {/* Collapsed status dot */}
        {collapsed && (
          <div className="flex justify-center">
            <span className="h-2 w-2 rounded-full bg-lime-300 shadow-[0_0_8px_#c5f36a]" title="System online" />
          </div>
        )}

        {/* Nav */}
        <nav className="flex flex-col gap-1.5">
          {NAV_ITEMS.map((item) => {
            const active = item.href === "/" ? pathname === "/" : pathname?.startsWith(item.href);
            return (
              <Link
                key={item.href}
                href={item.href}
                title={collapsed ? item.label : undefined}
                className={[
                  "rounded-2xl border transition",
                  collapsed ? "flex items-center justify-center py-2.5 text-base" : "px-4 py-2.5 text-sm font-medium",
                  active
                    ? "border-cyan-300/30 bg-cyan-300/10 text-white shadow-[inset_3px_0_0_#39e7d3]"
                    : "border-transparent text-slate-300 hover:border-white/10 hover:bg-white/5 hover:text-white",
                ].join(" ")}
              >
                {collapsed ? (
                  <span className={active ? "text-cyan-300" : ""}>{item.icon}</span>
                ) : (
                  item.label
                )}
              </Link>
            );
          })}
        </nav>

        {/* Spacer */}
        <div className="flex-1" />

        {/* Toggle button */}
        <button
          onClick={toggle}
          title={collapsed ? "Expand sidebar" : "Collapse sidebar"}
          className={[
            "flex items-center rounded-2xl border border-white/10 bg-white/[0.03] py-2.5 text-xs text-slate-400 transition hover:border-white/20 hover:bg-white/[0.06] hover:text-slate-200",
            collapsed ? "justify-center" : "gap-2 px-4",
          ].join(" ")}
        >
          <svg
            className={`h-4 w-4 shrink-0 transition-transform duration-300 ${collapsed ? "rotate-180" : ""}`}
            fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={2}
          >
            <path strokeLinecap="round" strokeLinejoin="round" d="M11 19l-7-7 7-7M18 19l-7-7 7-7" />
          </svg>
          {!collapsed && <span>Collapse</span>}
        </button>
      </div>
    </aside>
  );
}
