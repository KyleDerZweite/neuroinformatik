import { fmt } from "../lib/format";
import type { SessionState } from "../lib/api/types";

export function Telemetry({ state }: { state: SessionState | null }) {
  const training = state?.training;
  const epoch = training?.current_epoch ?? 0;
  const total = training?.total_samples ?? 0;
  const sampleIdx = training?.current_sample_index ?? 0;
  const sampleLabel = total
    ? `${Math.min(sampleIdx + 1, total)}/${total}`
    : "0/0";
  const phase = training?.phase ?? "idle";
  const latest = training?.epoch_losses?.length
    ? training.epoch_losses[training.epoch_losses.length - 1] ?? null
    : null;

  return (
    <section className="block telemetry">
      <header className="block-head">
        <span className="block-id">
          <span className="block-id-prefix">T·00</span>
          <span className="block-id-dot">·</span>
          <span className="block-id-title">TELEMETRY</span>
        </span>
        <span className="block-subid">
          {training?.mode === "run" ? "streaming · run mode" : "idle · inspect mode"}
        </span>
      </header>
      <div className="telemetry-grid">
        <Readout label="epoch" value={String(epoch)} />
        <ReadoutWire />
        <Readout label="sample" value={sampleLabel} />
        <ReadoutWire />
        <Readout label="phase" value={phase} />
        <ReadoutWire />
        <Readout
          label="loss · ℒ"
          value={latest === null ? "—" : fmt(latest, 6)}
          primary
        />
      </div>
    </section>
  );
}

function Readout({
  label,
  value,
  primary = false,
}: {
  label: string;
  value: string;
  primary?: boolean;
}) {
  return (
    <div className={primary ? "readout readout-primary" : "readout"}>
      <span className="readout-label">{label}</span>
      <strong className="readout-value">{value}</strong>
    </div>
  );
}

function ReadoutWire() {
  return <div className="readout-wire" aria-hidden />;
}
