export default function OrbitMark({ size = "md" }: { size?: "sm" | "md" | "lg" }) {
  const sizeClass = size === "lg" ? "h-28 w-28" : size === "sm" ? "h-8 w-8" : "h-12 w-12";

  return (
    <div className={`orbit-mark ${sizeClass}`} aria-hidden="true">
      <span className="orbit-mark__core" />
      <span className="orbit-mark__ring orbit-mark__ring--one" />
      <span className="orbit-mark__ring orbit-mark__ring--two" />
      <span className="orbit-mark__satellite" />
    </div>
  );
}
