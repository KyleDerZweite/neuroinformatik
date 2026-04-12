export function CornerMarks() {
  return (
    <>
      <div className="corner corner-tl" aria-hidden>
        <span>X·000 / Y·000 · ORIGIN</span>
      </div>
      <div className="corner corner-tr" aria-hidden>
        <span>SHEET 01 / 01 · REV B</span>
      </div>
      <div className="corner corner-bl" aria-hidden>
        <span>PROJECT N·I · MMXXVI</span>
      </div>
      <div className="corner corner-br" aria-hidden>
        <span>DRG-NN-002 · SCALE 1:1</span>
      </div>
    </>
  );
}
