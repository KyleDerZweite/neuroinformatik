import clsx from "clsx";
import type { ReactNode } from "react";

interface BlockProps {
  id: string;
  title: string;
  subtitle?: ReactNode;
  meta?: ReactNode;
  className?: string;
  children: ReactNode;
}

export function Block({
  id,
  title,
  subtitle,
  meta,
  className,
  children,
}: BlockProps) {
  return (
    <section className={clsx("block", className)}>
      <header className="block-head">
        <span className="block-id">
          <span className="block-id-prefix">{id}</span>
          <span className="block-id-dot">·</span>
          <span className="block-id-title">{title}</span>
        </span>
        {(subtitle || meta) && (
          <span className="block-head-meta">
            {subtitle && <span className="block-subid">{subtitle}</span>}
            {meta && <span className="block-meta">{meta}</span>}
          </span>
        )}
      </header>
      {children}
    </section>
  );
}
