"""Central parameter store.

Every tunable in the pipeline lives here. Implementation modules import from
this file and must not hardcode numeric constants. Grouped by pipeline stage
so that a scene variation is a single dataclass swap.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

# --------------------------------------------------------------------------
# Stage 0 -- robot
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class ArmParams:
    """Schunk LWA 4D, 7-DOF 3R-1R-3R (S-R-S) -- the arm used in [IROS].

    Kinematics taken from the ROS description package, file
    schunk_description/urdf/lwa4d/lwa4d.urdf.xacro (ipa320/schunk_modular_robotics).
    The joint chain there is:

        1_joint   0 0 0        shoulder roll (base z)
        2_joint   0 0 0.240    shoulder pitch   -- j1, j2, j3 all pass through
        3_joint   0 0 0        shoulder roll       the point (0, 0, 0.240)
        4_joint   0 0 0.328    ELBOW pitch
        5_joint   0 0 0        wrist roll, collinear with the forearm
        6_joint   0 0 0.323    wrist pitch      -- j5, j6, j7 all pass through
        7_joint   0 0 0        wrist roll          the wrist point
        ee_joint  0 0 0.0824   flange

    so it is genuinely S-R-S and maps onto this module's convention directly.
    An optional 0.140 m podest pedestal sits below the base; omitted here.

    Joint limits are the URDF values, i.e. the hardware limit minus the
    package's 0.02 rad safety_offset. Hardware limits are given alongside
    because the difference decides whether a sign door is admissible at all.
    """

    d_bs: float = 0.240
    d_se: float = 0.328
    d_ew: float = 0.323
    d_wt: float = 0.0824

    # URDF limits (hardware minus 0.02 safety offset), radians, q1..q7.
    q_min: tuple = (-3.1216, -2.12, -3.1216, -2.16, -3.1216, -2.07, -2.94)
    q_max: tuple = (3.1216, 2.12, 3.1216, 2.16, 3.1216, 2.07, 2.94)
    # Hardware limits, for the admissibility lemma.
    q_min_hw: tuple = (-3.14159, -2.14, -3.14159, -2.18, -3.14159, -2.09, -2.96)
    q_max_hw: tuple = (3.14159, 2.14, 3.14159, 2.18, 3.14159, 2.09, 2.96)

    # Capsule radii. APPROXIMATE: the real collision geometry is a set of STL
    # meshes not available here. These are fitted to the PowerBall module
    # diameters and should be replaced if the meshes are obtained.
    r_upper_arm: float = 0.070
    r_forearm: float = 0.060
    r_tool: float = 0.040

    # Off-axis geometry switches.
    #
    # With symmetric capsules every sign branch places identical geometry, so
    # pages would differ only by joint limits. Measured frame motion decides
    # where off-axis geometry must hang:
    #
    #   along the q2=0 gimbal arc, and between the s2=+/- pages, ONLY frame 2
    #   (the upper arm) moves; frames 3, 4 and 7 are identical to 1e-16.
    #
    # So a hub on frame 2 is what makes the s2 pages differ by collision and
    # makes the q2-door fiber non-trivial. A hub on frame 3 distinguishes the
    # s4 pages instead. Labelled separately so experiments can isolate either.
    use_off_axis_upper_arm: bool = False
    upper_arm_hub_offset: float = 0.105   # offset from the upper-arm axis, m
    upper_arm_hub_radius: float = 0.055   # modelled as a sphere
    upper_arm_hub_along: float = 0.60     # fraction along the upper arm

    use_off_axis_elbow: bool = False
    elbow_box_offset: float = 0.090   # offset from the joint-4 axis, metres
    elbow_box_radius: float = 0.055   # modelled as a sphere of this radius

    @property
    def reach_max(self) -> float:
        return self.d_se + self.d_ew

    @property
    def reach_min(self) -> float:
        return abs(self.d_se - self.d_ew)


# --------------------------------------------------------------------------
# Stage 0 -- task and scene
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class WeldParams:
    """Straight weld line with constant tool orientation.

    The tool tip traces p(y) = p0 + y * (p1 - p0) for y in [0, 1], with the
    tool z-axis held along `approach` (pointing from the tool into the work).
    """

    # Placement for the REAL LWA 4D. Three conditions must hold at once.
    #
    # (1) q2 door: the weld must cross the sphere |p - (E* - d_wt*taxis)| = d_ew
    #     with E* = (0, 0, d_bs + d_se) = (0, 0, 0.568) and d_ew = 0.323.
    # (2) q4 limit (+/-2.16 rad): |W - S| must stay in [0.307, 0.651].
    # (3) The door must be admissible on BOTH s6 pages. At a q2 door the whole
    #     configuration is pinned, so q5 is determined, and the s6 mirror
    #     shifts it by pi. With a VERTICAL torch q5 = 0 exactly, so the paired
    #     page needs q5 = +/-pi = 3.14159, which exceeds the URDF limit of
    #     3.1216 (= pi - 0.02 safety offset) and merely touches the hardware
    #     limit of pi. Every door is then dead. Tilting the torch moves q5 off
    #     zero; a 20-40 deg torch tilt is standard welding practice.
    #
    # At x=0.34, z=0.48, y in [-0.26, 0.26] with a 30 deg tilt:
    #   doors at normalised y = 0.2663 and 0.7337
    #   q5 = +12.6 deg on -++ and -167.4 deg on -+-, both inside +/-178.85
    #   |W - S| stays within reach
    p0: tuple = (0.34, -0.26, 0.48)
    p1: tuple = (0.34, 0.26, 0.48)
    approach: tuple = (0.500000, 0.0, -0.866025)
    reference: tuple = (0.866025, 0.0, 0.500000)


@dataclass(frozen=True)
class SurfaceParams:
    """A FLAT weld patch: the 2-D task of [IROS] d = 3 (findings 42).

    The tool tip traces p(u, v) = p0 + u * (p1 - p0) + v * sweep for
    (u, v) in [0, 1]^2, with the same fixed tool orientation as WeldParams.
    Every v-slice is therefore an ordinary straight weld line, which is what
    lets the 2-D machinery (IK, door geometry, fiber certificates) be reused
    slice by slice. The page becomes (u, v, theta).

    Default: EXP 2's line (x = 0.28, z = 0.46, vertical torch) swept +/-5 cm
    in z, so v = 0.5 IS the EXP 2 weld.
    """

    p0: tuple = (0.28, -0.26, 0.41)
    p1: tuple = (0.28, 0.26, 0.41)
    sweep: tuple = (0.0, 0.0, 0.10)
    approach: tuple = (0.0, 0.0, -1.0)
    reference: tuple = (1.0, 0.0, 0.0)


@dataclass(frozen=True)
class Page3Params:
    """The 3-D page (u, v, theta) of one sign branch, and its proof (findings 42).

    Units follow the 2-D page: u and v in [0, 1], theta in radians, and the
    SVM sees theta / pi, so one scaled unit means the same on every axis.
    """

    scale: tuple = (1.0, 1.0, 1.0 / np.pi)
    theta_min: float = -np.pi
    theta_max: float = np.pi
    # Roadmap. Edges are validated at PlannerParams.edge_resolution.
    n_samples: int = 6000
    k_neighbors: int = 12
    rng_seed: int = 20261108
    # Task: start pose, and the goal set u >= 1 - goal_tol (any v, any theta).
    start_u: float = 0.0
    start_v: float = 0.5
    goal_tol: float = 0.02
    # Training budget per wall, as run_proof's n_train_max / n_invalid_max.
    n_train_max: int = 900
    n_invalid_max: int = 300
    alg4_iters: int = 4
    # Retrains start Algorithm 1's gamma scan at gamma_warm x the previous
    # gamma (wall.train_wall gamma_start). A 3-D wall sits at gamma ~ 40, and
    # the exact scan from 1.0 in steps of 0.1 refits ~400 SVMs per retrain
    # (~15 s each, up to 8 per Algorithm 2 pass). None = the exact scan.
    gamma_warm: float | None = 0.5
    # The virtual boundary on a SURFACE task (findings 42). Everything outside
    # the page box is declared invalid -- not a thin band as at d = 2. That is
    # exact here: off the patch in u or v the tool is not on the task at all,
    # and beyond the theta range the theta-cut check has shown nothing valid
    # comes near the cut. The ring of rest samples (scaled distance ring_offset
    # outside the box, ring_per_face random points per face) only steers the
    # SVM shut; the wall may close anywhere between the page and the
    # reconstruction window (window_margin, scaled). With the d = 2 band
    # (ring 0.075, window 0.09 in u) a 3-D wall bulged through the gaps of any
    # affordable ring and was clipped open by the window.
    ring_per_face: int = 120
    ring_offset: float = 0.12
    window_margin: float = 0.35
    # Face anchors: the d = 3 edge_anchor_samples (findings 32 / 42). Random
    # states ON the four task faces (u = 0, 1 and v = 0, 1); the valid ones go
    # into the TRAINING set (never the roadmap, invariant 18), labelled by the
    # component they connect to through a validated edge. Without them the
    # sparse 3-D start class does not say where the start region stops at the
    # faces, and the first wall cut into it (43 of 48 failed facets).
    anchors_per_face: int = 400
    anchor_connect_tries: int = 4
    # Goal-face samples: random states ON u = 1 - goal_tol (every v and theta,
    # valid or not) in the REST class. The verdict needs the whole goal slab
    # outside the wall, and the exact test for that is "the wall ends before
    # u = 1 - goal_tol". Without these the wall was free to close through the
    # (virtual) region past u = 1 wherever no goal-component sample stood,
    # and a fully verified wall reached u = 1.02 -> INCONCLUSIVE. Like the
    # ring, they only steer the SVM; the exact test is unchanged.
    goal_face_samples: int = 600
    # Algorithms 2 and 4 feed valid wall points back into training. On a
    # sparse 3-D roadmap the NEAREST sample is often in the wrong class, the
    # classes interleave, and gamma ran 1.6 -> 78 until the wall could not
    # close (findings 42). So the 3-D page labels fed-back points by
    # CONNECTIVITY (SDCL's rule, page.label_point), and drops one that sits
    # closer than feedback_margin (scaled) to the other class -- the d = 2
    # sampled pipeline's guard (findings 32), for the same reason.
    feedback_margin: float = 0.03
    # TDC net spacing for a SURFACE wall (scaled units). Coarser than the
    # d = 2 net_d_min: the facet count grows as 1/eta^2, not 1/eta.
    eta: float = 0.06
    # The theta cut (invariant 5 / the bounded-theta decision): refuse unless
    # no valid configuration lies within seam_eps of theta_min / theta_max.
    seam_eps: float = 0.05
    seam_samples: int = 4000
    # Door curves: samples around each sphere-plane circle, tube radii in
    # (u, v, theta) units, and how densely the fiber certificate is sampled
    # along a curve before its Lipschitz guard is applied.
    door_samples: int = 2048
    tube_radius: tuple = (0.02, 0.02, 0.06)
    cert_stride: int = 4
    cert_lipschitz_safety: float = 2.0
    # Cross-page flood (findings 43): crossing endpoints p +/- cross_eps d
    # (scaled) are injected into every page at every cert_stride-th curve
    # sample, eight directions across the curve. A straight-through crossing
    # is the segment between such a pair.
    cross_eps: float = 0.004
    # Lazy PRM: when a spliced path fails verification, blacklist the roadmap
    # edges that failed, rebuild, and run the flood again -- at most this
    # many times. The edge checker samples every edge_resolution, which can
    # step over a thin out-of-limits sliver near a singularity; the witness
    # densifies to a JOINT step bound and does not.
    lazy_rounds: int = 6
    # Crossing pairs tried per door hop before phase 1 gives up on a route.
    cross_tries: int = 400
    # Recursive proof (docs/recursive_proof_design.md). Test (ii)'s own
    # subdivision depth: it evaluates sigma only (no collision), so it can go
    # deeper than Algorithm 3 (facet_max_depth_tri). The cost is 4^depth
    # sigma evaluations for a triangle that hugs a door, batched per level.
    sigma_max_depth: int = 8


@dataclass(frozen=True)
class BoxObstacle:
    """Axis-aligned box obstacle."""

    center: tuple
    half_extent: tuple
    label: str = "box"


@dataclass(frozen=True)
class SceneParams:
    """Obstacle set. `gap_open` selects the two worked-illustration variants."""

    gap_open: bool = True
    obstacles: tuple = ()

    @staticmethod
    def worked_illustration(gap_open: bool) -> "SceneParams":
        """The feasible / infeasible pair, chosen for RESOLUTION ROBUSTNESS.

        The box sits in the elbow's swing path at mid-weld (the elbow sweeps
        x in [-0.06, 0.33], z in [0.56, 0.74] over the valid set).

        Scene selection must be checked across grid resolutions, because grid
        connectivity can MISS a thin valid passage and report infeasibility
        that is a discretisation artifact. An earlier pair chosen from a single
        41x41 sweep was wrong: half (0.08, 0.060, 0.08) read infeasible at
        res=41 and FEASIBLE at 31, 51, 61, 81 and 101. The error is
        non-monotone in resolution, so "refine until stable" does not catch it.

        In-page start->goal connectivity, by half-extent and resolution:

          box half (x, y, z)   valid    31     51     71     91
          (0.08, 0.060, 0.08)  12.73%  FEAS   FEAS   FEAS   FEAS
          (0.10, 0.080, 0.09)   5.92%  infeas infeas infeas infeas
          (0.12, 0.080, 0.10)   4.24%  infeas infeas infeas infeas

        Note what this is and is not: agreement across resolutions is
        EVIDENCE, never a proof. The proof is the verified wall per region.
        Grid connectivity is only a sample source and a sanity check.
        """
        half = (0.08, 0.060, 0.08) if gap_open else (0.10, 0.080, 0.09)
        return SceneParams(
            gap_open=gap_open,
            obstacles=(
                BoxObstacle(
                    center=(0.13, 0.0, 0.68),
                    half_extent=half,
                    label="mid_block",
                ),
            ),
        )


# --------------------------------------------------------------------------
# Stage 1 -- page maps
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class PageMapParams:
    """Dense grid resolution for the (y, theta) page maps."""

    n_y: int = 241
    n_theta: int = 241
    theta_min: float = -np.pi
    theta_max: float = np.pi
    y_min: float = 0.0
    y_max: float = 1.0

    # theta is an arm angle, so a page is a CYLINDER and the theta = -pi and
    # +pi columns are adjacent. Set False only when joint limits genuinely
    # bound theta, making the page a rectangle.
    theta_periodic: bool = False   # flat rectangle; see findings 23

    # Numerical tolerance for the IK existence test (arccos argument).
    ik_tol: float = 1e-9
    # Treated as Sigma_coord when |u . e_z| exceeds this (arm-angle reference
    # plane degenerates) or when the elbow circle radius drops below this.
    coord_axis_tol: float = 1e-6
    coord_radius_tol: float = 1e-6


# --------------------------------------------------------------------------
# Stage 2 -- singular set
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class SigmaPolicy:
    """What the proof is allowed to assume about Sigma. Conservative by design.

    These are deliberately switches rather than buried constants, because each
    one changes what the certificate claims.
    """

    # Sigma_kin must NEVER invalidate a cell. At a kinematic singularity the
    # Jacobian drops rank and joint velocity blows up, but this is geometric
    # (kinematic) planning without differential constraints: the path still
    # exists as a continuous curve in Q_valid. Admitting singular passages is
    # the conservative direction. Excluding them would prove infeasibility of
    # a NARROWER problem than "no plan exists" -- a weaker claim dressed as a
    # stronger one. Setting this True is unsound for the headline claim and the
    # pipeline refuses to run with it set.
    sigma_kin_invalidates: bool = False

    # Doors are open by default. A proof built on more connectivity than really
    # exists still proves disconnection, so assuming open is conservative.
    doors_open_by_default: bool = True

    # Stage 5 refinement only: attempt to certify a door BLOCKED by showing the
    # whole self-motion fiber at it is invalid. Off in the main pipeline -- it
    # is never needed for soundness, only to strengthen a proof that otherwise
    # fails. Output is tagged [stage5-fiber].
    enable_fiber_certification: bool = False
    fiber_samples: int = 721          # samples along the 1-D fiber
    fiber_refine_iters: int = 30


@dataclass(frozen=True)
class SigmaParams:
    """Detection and covering of Sigma = Sigma_sign U Sigma_coord U Sigma_kin."""

    # Bisection refinement for interior sign doors (q2, q6 crossing zero).
    bisect_iters: int = 40
    # A disc must strictly contain the true singular point: too large is safe.
    disc_radius_scale: float = 2.0     # cells, applied PER AXIS (y and theta)
    disc_radius_floor: float = 0.004   # absolute floor, page units
    # Sigma_kin threshold on det(J J^T).
    kin_det_tol: float = 1e-7
    # Fold doors (q4 -> 0) are found where the IK discriminant vanishes, not by
    # a sign crossing; this is the margin on |cos q4| used to bracket them.
    fold_cos_margin: float = 1e-3
    # Within a page the sign is fixed by construction, so a door is where
    # |q_j| TOUCHES zero. This is the tolerance of the numerical cross-check
    # that confirms the analytic door finder.
    touch_tol: float = 2.0e-2
    use_numeric_crosscheck: bool = True


# --------------------------------------------------------------------------
# Stage 3 -- RBF-SVM wall (IJRR Algorithm 1-3)
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class WallParams:
    """IJRR hyper-parameters, kept at the paper's values where they transfer."""

    gamma_init: float = 1.0        # IJRR: gamma = 1.0
    gamma_step: float = 0.1        # IJRR: delta-gamma = 0.1
    # findings 29: invariant 14's "gamma ~ 40 fragments the manifold" is
    # IJRR's number in IJRR's units, and it does NOT transfer here. This page
    # is scaled to y in [0,1], theta/pi in [-1,1], where opposite-class points
    # stand about 0.060 apart; an RBF needs gamma ~ 1/d^2 ~ 280 to resolve
    # that, so gamma = 30 is a very WIDE kernel in these units, not a fine
    # one. Capping at 30 made the hard pages unseparable (accuracy 0.96 < 1,
    # i.e. the wall no longer separated its own training data) and no amount
    # of Algorithm 4 feed-back could recover.
    #
    # Fragmentation is not policed by a gamma number anyway: reconstruct_wall
    # requires the zero set to be a single CLOSED curve and check_polytope
    # verifies every facet lies in C_obs, so a wall that has broken into
    # islands fails closedness or fails facets. gamma stays a cost and
    # robustness signal -- reported, and warned about past gamma_warn -- while
    # the certificate rests on closedness and the facet checks.
    gamma_max: float = 400.0
    # How Algorithm 1 searches gamma (findings 45). "linear": IJRR's scan,
    # gamma_init, +gamma_step, ... -- one SVM fit per 0.1, which is what made
    # the slow walls slow (findings 44: 99% of EXP 3's time). "gallop": grow
    # by gamma_gallop until the data separates, then bisect back down to
    # gamma_step resolution -- ~12 fits instead of hundreds. It may settle on
    # a different separating gamma where accuracy is not monotone in gamma;
    # soundness never rests on gamma (closedness + Algorithm 3 decide).
    gamma_search: str = "linear"   # "linear" | "gallop"
    gamma_gallop: float = 1.25
    gamma_warn: float = 40.0       # report, do not refuse, above this
    svm_C: float = 10.0            # findings 20: 1e6 pins gamma at 1.0

    # Page-coordinate scaling applied before the SVM. Without this a single
    # gamma is meaningless across the y and theta axes.
    scale_y: float = 1.0
    scale_theta: float = 1.0 / np.pi

    # (epsilon, eta)-net subsampling (IJRR Definition 2).
    net_keep_fraction: float = 0.5
    net_d_min: float = 0.02
    # Cross-class eta-separation for the ROADMAP source (findings 38).
    # A grid never puts two opposite-class points closer than one cell, which
    # overstates the separation; a roadmap measures the true one (0.03839 vs
    # 0.07502 on EXP 3's start page) and Algorithm 1 then needs gamma ~ 679
    # instead of ~178. This drops both members of every cross-class pair
    # closer than eta, which bounds the required gamma by 1/eta^2 again.
    # 0.0 disables it. Only the roadmap source reads this.
    eta_separation: float = 0.06

    # Class of a point Algorithms 2 and 4 feed back into the training set.
    # "topological" is SDCL's rule (makeTrainingDataFromGraph): the point
    # joins the class it is CONNECTED to through valid page space.
    # "metric" is the nearest-class heuristic it replaces.
    #
    # Default is "metric" only because "topological" MEASURED WORSE on EXP 2
    # (findings 21) and the cause is not yet settled: the labeller is
    # cylinder-aware via label_regions, the SVM is trained on a flat
    # rectangle, so a point connected across the theta seam lands at the far
    # edge of the classifier's input space. Fix that disagreement before
    # judging the rule itself -- on method, SDCL's rule is the right one.
    label_mode: str = "metric"
    # Which valid cells form the 'rest' class. "goal_only" is IJRR's split
    # (start component vs goal component); "all_valid" also throws in
    # components that reach neither, which can force the wall to thread
    # between region k and a small neighbour -- gamma 2 -> 10 on EXP 3's
    # partner page, with facets landing in valid cells (findings 29).
    rest_mode: str = "goal_only"
    # Algorithm 4 feeds a failed facet centre back into the training data. If
    # that centre sits closer than this (in SCALED page units) to a point of
    # the class it is NOT being given, the two classes stop being separable at
    # any gamma below the fragmentation limit, and every later retrain makes it
    # worse: measured accuracy decay 0.9993 -> 0.9796 with facets going
    # 138/167 -> 128/165 (findings 29). Such a centre is routed to Algorithm
    # 4's other branch -- shrink the facets -- instead of poisoning the fit.
    # One cell diagonal is ~0.037 scaled units on a 61-cell page and the two
    # classes start ~0.060 apart, so half a cell is the natural floor.
    #
    # Default 0 = off, which is IJRR Algorithm 2/4 as written. It is kept
    # because it is what isolated the decay: with gamma capped, turning this on
    # held the facets at ~150/170 instead of letting them fall to 128/165. With
    # gamma free the classes separate and the guard is not needed, so it stays
    # available as a diagnostic rather than running by default.
    feedback_margin: float = 0.0
    label_search_cells: int = 3     # grid radius searched for a connectable cell
    label_motion_samples: int = 8   # samples per checkMotion segment

    # Facet check (IJRR Algorithm 3).
    # Recursion cap on segment decomposition. Measured (findings 19): 66 of
    # 73 invalid facets pass at depth 0 and the worst needs 5.2 halvings, so
    # 8 is ample. Do NOT raise this -- a failing facet costs ~2^depth leaf
    # checks, and depth 20 is what made EXP 2 grind for 18 CPU minutes.
    facet_max_depth: int = 8
    # The same cap for TRIANGLE facets (d = 3, findings 41). A triangle splits
    # into 4, not 2, so depth k costs up to 4^k leaves: depth 8 is 65,536 per
    # facet, and a synthetic wall grazing C_obs by 0.001 measured 1.35 M depth
    # queries. 5 keeps the cost near the segment budget (1,365 nodes) and
    # still shrinks the ball 32x; a wall that needs more is grazing, and
    # invariant 11 says Algorithm 4 must push it in, not subdivide it.
    facet_max_depth_tri: int = 5
    contour_level: float = 0.0

    # Virtual-boundary band thickness (IJRR Fig. 5). Lets a wall close against
    # the page edge instead of running off it.
    virtual_band: float = 0.03
    # Where the virtual obstacle band ENDS and where the virtual free ring
    # SITS, both in multiples of virtual_band. IJRR Fig. 5 puts the free
    # region immediately outside the obstacle band; these two numbers must
    # therefore be adjacent, with no gap between them.
    #
    # They used to be 1.0 and 2.0, which left the strip
    # (y_min - 2 eps, y_min - eps) as neither: not the invalid band, and not
    # the ring. The wall is free to close in that strip, and off the weld's
    # own y range the extrapolated pose is often still VALID, so the facets
    # there fail with "facet centre is VALID". Measured on the sampled
    # pipeline once the y = 0 anchors had pushed the wall out of the page:
    # every remaining failure sat at y in [-0.058, -0.054], i.e. inside that
    # strip (findings 32).
    #
    # Declaring the whole strip invalid is not a licence taken: y < y_min is
    # off the weld, where the task is not defined at all, so no path may go
    # there. The band's thickness is a modelling choice and only has to be
    # positive and to exclude the ring, which virtual_ring > virtual_band_outer
    # guarantees.
    virtual_band_outer: float = 2.0
    virtual_ring: float = 2.5
    # Reconstruction window, in multiples of virtual_band. Must exceed
    # virtual_ring or the wall closes on the domain edge instead of around the
    # ring, and the traced contour is an open arc.
    recon_margin: float = 3.0
    # A traced loop whose endpoints are further apart than this is NOT closed,
    # which fails Definition 1 (I) exactly as a failed triangulation does.
    # (contour reconstruction only)
    closure_tol: float = 1.0e-6

    # IJRR 4.3 reconstruction (findings 39).
    #   "tdc"      the tangential Delaunay complex, built from points ON the
    #              wall only -- no grid is evaluated. Dimension-generic: a
    #              polyline at d = 2, a triangulated surface at d = 3.
    #   "contour"  legacy: evaluate the SVM on a grid and trace the zero
    #              level with matplotlib. Kept only to reproduce old numbers.
    reconstruction: str = "tdc"
    # Net spacing eta, as a multiple of net_d_min (so Algorithm 4's "shrink
    # facets" branch, which halves net_d_min, halves eta too).
    tdc_eta_factor: float = 1.0
    # Tangent step when growing the net, and the minimum spacing a new point
    # must keep, both in multiples of eta. Spacing ends up in [accept, ~2] eta.
    tdc_step: float = 1.0
    tdc_accept: float = 0.8
    # Uniform random points in the reconstruction window, projected onto the
    # wall to seed the net -- alongside Algorithm 2's wall points and the
    # training data. The window includes the virtual band, where there are no
    # invalid samples for Algorithm 2 to project, so these are what reach the
    # stretch of wall that closes outside the page.
    tdc_seed_points: int = 400
    # Neighbours per star; 0 = automatic (10 for a curve, 24 for a surface).
    tdc_k_neighbors: int = 0
    # Inconsistency removal: rounds of random point weights, each weight at
    # most tdc_weight_frac * the point's nearest-neighbour distance.
    tdc_fix_rounds: int = 20
    tdc_weight_frac: float = 0.3
    # Training points closer than this (in eta) to the wall are not used for
    # the sidedness check: a facet is a chord and may cut a sliver off them.
    tdc_side_tol: float = 1.0
    tdc_max_points: int = 20000

    # Penetration-depth mode. "ijrr_local_opt" is the paper's own method lifted
    # to Y+; "lipschitz" is the hardened variant (exact workspace PD plus a
    # Lipschitz transfer). Output is labelled either way.
    # Level-set projection for IJRR Eq. 2. "newton" uses the analytic
    # gradient directly; "slsqp" is the general solve, same answer but ~8 ms
    # of scipy overhead per call, and Algorithm 2 makes ~900 calls per pass.
    projection: str = "newton"
    newton_iters: int = 30
    newton_tol: float = 1.0e-10

    pd_mode: str = "ijrr_local_opt"
    pd_seed_attempts: int = 3
    # How the ijrr pd mode searches for a valid point closer than its seed
    # (findings 40). "march": walk from the facet centre toward the seed and
    # bisect at the first valid sample. "slsqp": the legacy solve, whose
    # constraint (classify, a +/-1 step) has no gradient -- it never
    # converged and cost 99.9% of run_proof.
    pd_search: str = "march"
    pd_bisect_iters: int = 12
    lipschitz_fd_step: float = 1.0e-5
    lipschitz_safety: float = 1.5
    lipschitz_fallback: float = 50.0


# --------------------------------------------------------------------------
# Stage 4 -- planner and flood fill
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class PlannerParams:
    """PRM in Y+ supplying samples to the proof side (IJRR runs both in parallel)."""

    n_samples: int = 4000
    k_neighbors: int = 12
    edge_resolution: float = 0.01
    rng_seed: int = 20261029
    # Where run_proof gets its five answers. "roadmap" is the default: a grid
    # is a uniform lattice, so its connectivity between two cell centres was
    # never checked, and `regions_under` -- the one question with no
    # adjudicator downstream -- becomes a footprint guess over a label array
    # instead of a lookup of the component a door anchor actually joined
    # (findings 38). "grid" keeps the original backend for the plotting
    # scripts and for speed comparisons.
    page_source: str = "roadmap"      # "roadmap" | "grid"
    # Samples per page for the roadmap backend. Separate from n_samples,
    # which the standalone PRM in prm.py uses.
    source_samples: int = 1500
    # Vectorised validity (batch.py, findings 44): N states through IK,
    # limits and collision in one numpy pass, tag-for-tag equal to the
    # scalar classify (tests/test_batch.py). False restores the scalar loops.
    batch_classify: bool = True


@dataclass(frozen=True)
class FloodParams:
    """Outer flood fill over pages and doors."""

    # Goal set: reach y = goal_y on any page at any theta.
    goal_y: float = 1.0
    goal_y_tol: float = 0.02
    start_y: float = 0.0
    start_y_tol: float = 0.02
    # Anchor the per-page labels on the seed region (flood) or on the goal
    # component across all pages (the strictly IJRR-shaped variant).
    anchor: str = "seed"   # "seed" | "goal"


@dataclass(frozen=True)
class CrossingParams:
    """Exact certification of a door's self-motion fiber.

    The old test asked "does ANY delta on the fiber work?" and answered yes
    whenever one sampled point was admissible. That is not the question. To
    change pages the robot must travel from the page's own IK spelling
    (delta = 0) to the partner page's spelling (delta = +/- pi) WITHOUT
    leaving the admissible set, so the question is whether ONE CONNECTED
    admissible interval contains both. Invariant 21 is the same defect seen
    from the other end: checking only the endpoint accepts a motion the joint
    cannot actually perform.

    The interval is available in closed form. Along the fiber the robot does
    not move at all (gimbal null motion), so collision is CONSTANT -- one
    query decides it -- and each joint is either constant or LINEAR in delta,
    so joint limits give intervals whose intersection is the answer. No
    sampling anywhere.
    """

    exact: bool = True
    # The closed form is valid only where the fiber is a true null motion.
    # Verified rather than assumed: if any link point moves more than this
    # between two fiber samples, fall back to the sampled sweep and say so.
    null_motion_tol_m: float = 1.0e-6
    # Require a connected admissible interval spanning 0 -> +/-pi (item #0).
    # Switching this off restores the old "any valid point" semantics.
    require_connected: bool = True
    # What may DISCHARGE a door (findings 43). "posture" (default): only facts
    # about the door POSTURE -- it is in collision, or a joint that does not
    # move along the fiber is out of range. "fiber_interval" (legacy): also
    # the delta-interval verdicts ("short of +/-pi", "delta = 0 not
    # admissible").
    #
    # The legacy verdicts are UNSOUND. At a q2 door the q1/q3 axes coincide,
    # so the IK's q1 there is atan2(0, 0) -- 0 on one page, -180 deg on the
    # other -- and the interval is measured from that artifact. A robot
    # arriving from direction d holds q1 = the elbow azimuth along d and can
    # cross STRAIGHT THROUGH (q2: + -> -, q1 and q3 continuous, no fiber
    # motion). On EXP 3's door, "short by 1.15 deg -> NOT crossable", 23 of 24
    # straight-through crossings pass verify_witness.
    discharge: str = "posture"


@dataclass(frozen=True)
class WitnessParams:
    """Constructive certificate for FEASIBLE: a verified end-to-end path.

    The wall is a non-constructive certificate for INFEASIBLE. Feasibility is
    an existential statement, so its witness is the motion itself -- IJRR
    Figure 2 reports a plan from the PLANNER, never from the proof side. Our
    flood fill reports feasibility from region labels instead, which composes
    unverified local guesses into a global claim; a route through a door the
    robot cannot reach, or cannot cross, then yields "feasible" for a task
    that is not. The path cannot be fooled that way: it is ONE object, and a
    broken leg means there is nothing to verify.

    Verification is deliberately cheap and independent of how the path was
    found: N x FK + N x limits + N x collision, plus junction and step bounds.
    No SVM, no topology, no resolution.
    """

    enable: bool = True
    # Densification. step_max also bounds the gap a collision sample must
    # cover, so it is what makes a discrete sample stand in for a curve.
    step_max: float = 0.02          # rad, max joint step between samples
    door_samples: int = 129         # samples along a door leg
    page_step: float = 0.01         # page-metric step along an in-page leg
    # Tolerances.
    tol_pos: float = 1.0e-6         # m, tool position error off the weld
    tol_rot: float = 1.0e-6         # rad, tool orientation error
    tol_join: float = 1.0e-9        # rad, junction gap between legs in Q
    # Scene.check().clearance SAMPLES each capsule axis, so it is an UPPER
    # bound on the true clearance. scene.py argues that bias is safe
    # "everywhere it is used" -- true for wall verification, where it can only
    # make a facet check fail. For a swept path segment the sign FLIPS: an
    # optimistic clearance would certify a segment that actually collides.
    # With 49 samples on a 0.33 m link the error reaches ~3.4 mm. A witness
    # path is a one-off, so it pays for the exact routine instead.
    exact_collision: bool = True
    # A straight-through door crossing switches spelling a hair either side
    # of the door point, as a FRACTION of the crossing half-length (findings
    # 43). The door point itself is never sampled: its IK spelling is an
    # atan2(0, 0) artifact. Not much smaller: within ~1e-9 of the
    # singularity the IK's orientation error (1.2e-6 rad) exceeds tol_rot.
    cross_hair: float = 1.0e-3
    # Roadmap used to produce in-page legs. Separate from PlannerParams so
    # path planning can be dense without changing the proof-side sampling.
    prm_samples: int = 1500
    prm_k: int = 12
    rng_seed: int = 20261108


# --------------------------------------------------------------------------
# Top-level bundle
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class BudgetParams:
    """Fail-fast budgets and escalation (findings 44).

    A wall attempt that is going nowhere is STOPPED instead of ground to its
    caps: Algorithm 1 asking for gamma beyond gamma_stop, `max_unclosed`
    reconstructions in a row that do not close, or more than wall_seconds of
    wall-clock. The whole run has run_seconds.

    A stop can only ever produce INCONCLUSIVE ("stopped by budget: ..."): the
    stopped region has no verified wall, so the certificate is missing a
    piece and INFEASIBLE is impossible; FEASIBLE needs a verified path and a
    stop never supplies one. A stopped region is NEVER dropped from the
    certificate. Escalation then re-runs the WHOLE proof with escalate_factor
    times the samples, a new seed and escalate_factor times the per-wall
    budgets, up to escalate_rungs runs and escalate_total_seconds; the first
    conclusive verdict wins, and each rung is an independent proof.

    enabled=False makes every hook a no-op (the pre-44 behaviour).
    """

    enabled: bool = True
    gamma_stop: float = 100.0         # Alg 1 needs a narrower kernel -> stop
    max_unclosed: int = 3             # consecutive non-closed reconstructions
    wall_seconds: float = 300.0       # one wall attempt (Alg 1-4 of one region)
    run_seconds: float = 1800.0       # one whole proof run
    escalate_rungs: int = 3
    escalate_factor: float = 3.0      # samples x3 per rung
    escalate_budget_factor: float = 2.0   # gamma_stop, wall_seconds x2 per rung
    escalate_seed_step: int = 1009
    escalate_total_seconds: float = 3600.0


@dataclass(frozen=True)
class Config:
    arm: ArmParams = field(default_factory=ArmParams)
    weld: WeldParams = field(default_factory=WeldParams)
    scene: SceneParams = field(default_factory=lambda: SceneParams.worked_illustration(True))
    pagemap: PageMapParams = field(default_factory=PageMapParams)
    sigma: SigmaParams = field(default_factory=SigmaParams)
    sigma_policy: SigmaPolicy = field(default_factory=SigmaPolicy)
    wall: WallParams = field(default_factory=WallParams)
    planner: PlannerParams = field(default_factory=PlannerParams)
    flood: FloodParams = field(default_factory=FloodParams)
    crossing: CrossingParams = field(default_factory=CrossingParams)
    witness: WitnessParams = field(default_factory=WitnessParams)
    budget: BudgetParams = field(default_factory=BudgetParams)


DEFAULT = Config()
