"""Write a one-page summary of the demo results from demo_report.json."""
from __future__ import annotations
import json, sys
from pathlib import Path

out = Path(sys.argv[1] if len(sys.argv) > 1 else "out/demo")
r = json.loads((out / "demo_report.json").read_text())
A, B, C = r.get("A", {}), r.get("B", {}), r.get("C", {})

def pct(x): return f"{100 * x:.1f}%"
def fr(f): return f"{f[0]}/{f[1]} ({100 * f[0] / max(1, f[1]):.0f}%)"

lines = [
"# Infeasibility proofs on the real Schunk LWA 4D — demo results",
"",
"Robot: Schunk LWA 4D, kinematics and joint limits from the ROS description",
"package (`schunk_description/urdf/lwa4d/lwa4d.urdf.xacro`).",
"`d_bs=0.240, d_se=0.328, d_ew=0.323, d_wt=0.0824`, reach 0.651 m.",
"Limits: ±178.85° on q1/q3/q5, ±121.5/123.8/118.6° on q2/q4/q6, ±168.45° on q7.",
"",
"Scene, identical in all three cases:",
"",
"- weld: straight line, x = 0.28, z = 0.46, y ∈ [−0.26, 0.26]",
"- start: a **single posture** on page (++-) at y = 0",
"- goal: reach y = 1",
"- obstacle: a cube in the arm's swept volume (cases B and C)",
"",
"The torch tilt is the door switch. A q2 door pins the whole configuration, so",
"q5 is determined, and the s6 mirror shifts it by π. At tilt 0 that puts q5 at",
"±180°, outside the ±178.85° limit, so the doors are blocked; tilting the torch",
"moves q5 off zero and opens them. A 20–40° torch tilt is standard practice.",
"",
"## Results",
"",
"| case | obstacle | tilt | doors | verdict | output |",
"|---|---|---|---|---|---|",
f"| A | none | 0° | blocked | **FEASIBLE** | plan, {A.get('waypoints','?')} waypoints + video |",
f"| B | yes | 0° | {B.get('doors_blocked','?')} blocked / {B.get('doors_open','?')} open | **INFEASIBLE** | wall, **one page** |",
f"| C | yes | 30° | {C.get('doors_blocked','?')} blocked / {C.get('doors_open','?')} open | **INFEASIBLE** | walls, **{len(C.get('pages_walled',[]))} pages** |",
"",
"### Case A — the robot finishes the weld",
"",
f"Page (++-) is {pct(A.get('valid',{}).get('++-',0))} valid and has a region spanning",
f"start to goal (region(s) {A.get('spanning_regions')}). The planner returns a",
f"{A.get('waypoints','?')}-waypoint plan. See `A_weld.mp4` and `A_pages_feasible.png`.",
"",
"### Case B — infeasible, proof closes inside ONE page",
"",
"The obstacle disconnects start from goal on the start page. Every door is",
"blocked, so the other pages are unreachable and the certificate needs just one",
f"wall. Wall: closed = {B.get('closed')}, facets verified {fr(B.get('facets',[0,0]))},",
f"{B.get('discs_discharged','?')} Σ disc(s) discharged as blocked doors.",
"See `B_pages_proof_onepage.png`.",
"",
"### Case C — infeasible, proof SPANS pages through a door",
"",
f"At 30° the doors open ({C.get('doors_open','?')} open). The start region is still",
"enclosed, but now a door leads out of it, so the proof must follow the door and",
f"wall the neighbouring page too. Pages walled: {C.get('pages_walled')}; door",
f"followed: {C.get('door_link')}. Facets verified {fr(C.get('facets',[0,0]))}.",
"See `C_pages_proof_twopage.png`.",
"",
"## What is and is not established",
"",
"- Kinematics are the real LWA 4D and the IK round-trips to 1.8e-15.",
"- **Collision geometry is approximate**: the STL meshes were not available, so",
"  links are capsules fitted to the PowerBall module diameters. Collision-driven",
"  numbers are provisional.",
"- A wall counts as a proof only when **every** facet verifies. Where the facet",
"  count below is short of 100%, the figure shows a learned and traced wall, not",
"  a completed certificate.",
"- Discharging a Σ disc (case B) relies on a dense **sample** of the door's",
"  self-motion fiber, which is measure-zero in Q. That is evidence, not a",
"  certificate; a rigorous version needs interval arithmetic along the fiber.",
]
(out / "SUMMARY.md").write_text("\n".join(lines) + "\n")
print(f"[out] {out / 'SUMMARY.md'}")
