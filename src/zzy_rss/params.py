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
    gamma_warn: float = 40.0       # report, do not refuse, above this
    svm_C: float = 10.0            # findings 20: 1e6 pins gamma at 1.0

    # Page-coordinate scaling applied before the SVM. Without this a single
    # gamma is meaningless across the y and theta axes.
    scale_y: float = 1.0
    scale_theta: float = 1.0 / np.pi

    # (epsilon, eta)-net subsampling (IJRR Definition 2).
    net_keep_fraction: float = 0.5
    net_d_min: float = 0.02

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
    contour_level: float = 0.0

    # Virtual-boundary band thickness (IJRR Fig. 5). Lets a wall close against
    # the page edge instead of running off it.
    virtual_band: float = 0.03
    # Reconstruction window, in multiples of virtual_band. Must exceed 2 (the
    # ring's own offset) or the wall closes on the domain edge instead of
    # around the ring, and the traced contour is an open arc.
    recon_margin: float = 3.0
    # A traced loop whose endpoints are further apart than this is NOT closed,
    # which fails Definition 1 (I) exactly as a failed triangulation does.
    closure_tol: float = 1.0e-6

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


# --------------------------------------------------------------------------
# Top-level bundle
# --------------------------------------------------------------------------


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


DEFAULT = Config()
