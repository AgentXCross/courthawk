// Renders 2 thicker neon corner-bracket accents (top-left, bottom-right) for a .panel
// element. A background-image hack was tried first but sharp rectangles don't follow a
// rounded border-radius correctly — these are real elements instead, each with its own
// matching border-radius on just the one corner it occupies, so the bracket traces the
// same curve as the panel itself.
function PanelCorners() {
  return (
    <>
      <span className="panel-corner tl" />
      <span className="panel-corner br" />
    </>
  )
}

export default PanelCorners
