# Infeasibility proofs on the real Schunk LWA 4D — demo results

Robot: Schunk LWA 4D, kinematics and joint limits from the ROS description
package (`schunk_description/urdf/lwa4d/lwa4d.urdf.xacro`).
`d_bs=0.240, d_se=0.328, d_ew=0.323, d_wt=0.0824`, reach 0.651 m.
Limits: ±178.85° on q1/q3/q5, ±121.5/123.8/118.6° on q2/q4/q6, ±168.45° on q7.

Scene, identical in all three cases:

- weld: straight line, x = 0.28, z = 0.46, y ∈ [−0.26, 0.26]
- start: a **single posture** on page (++-) at y = 0
- goal: reach y = 1
- obstacle: a cube in the arm's swept volume (cases B and C)

The torch tilt is the door switch. A q2 door pins the whole configuration, so
q5 is determined, and the s6 mirror shifts it by π. At tilt 0 that puts q5 at
±180°, outside the ±178.85° limit, so the doors are blocked; tilting the torch
moves q5 off zero and opens them. A 20–40° torch tilt is standard practice.

## Results

| case | obstacle | tilt | doors | verdict | output |
|---|---|---|---|---|---|
| A | none | 0° | blocked | **FEASIBLE** | plan, 61 waypoints + video |
| B | yes | 0° | 16 blocked / 0 open | **INFEASIBLE** | wall, **one page** |
| C | yes | 30° | 0 blocked / 16 open | **INFEASIBLE** | walls, **2 pages** |

### Case A — the robot finishes the weld

Page (++-) is 49.2% valid and has a region spanning
start to goal (region(s) [1, 2]). The planner returns a
61-waypoint plan. See `A_weld.mp4` and `A_pages_feasible.png`.

### Case B — infeasible, proof closes inside ONE page

The obstacle disconnects start from goal on the start page. Every door is
blocked, so the other pages are unreachable and the certificate needs just one
wall. Wall: closed = False, facets verified 107/117 (91%),
2 Σ disc(s) discharged as blocked doors.
See `B_pages_proof_onepage.png`.

### Case C — infeasible, proof SPANS pages through a door

At 30° the doors open (16 open). The start region is still
enclosed, but now a door leads out of it, so the proof must follow the door and
wall the neighbouring page too. Pages walled: ['++-', '-+-']; door
followed: ++-->-+-. Facets verified 318/359 (89%).
See `C_pages_proof_twopage.png`.

## What is and is not established

- Kinematics are the real LWA 4D and the IK round-trips to 1.8e-15.
- **Collision geometry is approximate**: the STL meshes were not available, so
  links are capsules fitted to the PowerBall module diameters. Collision-driven
  numbers are provisional.
- A wall counts as a proof only when **every** facet verifies. Where the facet
  count below is short of 100%, the figure shows a learned and traced wall, not
  a completed certificate.
- Discharging a Σ disc (case B) relies on a dense **sample** of the door's
  self-motion fiber, which is measure-zero in Q. That is evidence, not a
  certificate; a rigorous version needs interval arithmetic along the fiber.
