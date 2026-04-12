import clsx from "clsx";

export function ConnectionIndicator({ connected }: { connected: boolean }) {
  return (
    <span className={clsx("conn-indicator", connected ? "is-on" : "is-off")}>
      <span className="conn-dot" aria-hidden />
      <span className="conn-label">{connected ? "online" : "offline"}</span>
    </span>
  );
}
