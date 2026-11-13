"""The algebraic route in JOINT variables: general for any serial revolute chain.

algebraic.py wrote the door conditions in posture variables (the elbow),
which used the S-R-S structure. Here nothing does:

  unknowns  x = (c_1..c_n, s_1..s_n, u)        c_i = cos q_i, s_i = sin q_i
  equations c_i^2 + s_i^2 - 1 = 0                     n
            tool_position(c, s) - (p0 + u A) = 0       3   (the task: a weld slice)
            M[0,2], M[1,2], M[1,0] = 0, M = R_t^T R    3   (tool orientation)
            s_j = 0                                    1   (the door: label joint j is 0)
            a c_k + b s_k = 0                          1   (fixes the free spin of the
                                                            neighbour pair at the door)

for an n = 7 chain: 15 x 15. The forward kinematics come from krawczyk.SerialChain
(joint axes and offsets), i.e. from a URDF. Its total-degree Bezout number is
in the millions, so no start system is used. Instead (the method of
HomotopyContinuation.jl):

  1. MONODROMY. Pick random complex joint values q* with s_j = 0 and DEFINE a
     parameter instance P0 for which q* is a solution (target pose := FK(q*)).
     Walk the parameters around random loops P0 -> P1 -> P2 -> P0; tracking the
     known solutions permutes them and reveals new ones. Stop when several
     loops in a row add nothing: the solution set of the generic instance.
  2. PARAMETER HOMOTOPY. Track every generic solution from P0 to each real
     instance (a weld slice), through a random complex detour. Every isolated
     solution of the real instance is the limit of a tracked path (with
     probability one); some paths go to infinity or to spurious solutions,
     which are filtered.

Completeness of step 1 is probabilistic (the stopping rule); a "trace test"
would certify it. Nothing here uses a door formula, the arm's IK, or an L.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .krawczyk import SerialChain

_AX = {"x": 0, "y": 1, "z": 2}


_E = {   # rot(axis, c, s) = E0 + c Ec + s Es: every factor is LINEAR in (c, s)
    "z": (np.diag([0.0, 0.0, 1.0]), np.diag([1.0, 1.0, 0.0]),
          np.array([[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 0.0]])),
    "y": (np.diag([0.0, 1.0, 0.0]), np.diag([1.0, 0.0, 1.0]),
          np.array([[0.0, 0.0, 1.0], [0.0, 0.0, 0.0], [-1.0, 0.0, 0.0]])),
    "x": (np.diag([1.0, 0.0, 0.0]), np.diag([0.0, 1.0, 1.0]),
          np.array([[0.0, 0.0, 0.0], [0.0, 0.0, -1.0], [0.0, 1.0, 0.0]])),
}


def _rot(axis: str, c, s):
    E0, Ec, Es = _E[axis]
    return E0 + c * Ec + s * Es


def fk_cs(chain: SerialChain, c, s):
    """Tool position and rotation from (c, s) -- polynomial in them (one point)."""
    R = np.eye(3, dtype=complex)
    p = chain.base.astype(complex)
    for i, ax in enumerate(chain.axes):
        R = R @ _rot(ax, c[i], s[i])
        if i in chain.after:
            p = p + R @ chain.after[i]
    return p, R


def _ops(interval: bool):
    if interval:
        from .interval import Interval, imatmul
        return imatmul, (lambda a: Interval.of(a))
    return np.matmul, (lambda a: a)


def _fk_parts(chain: SerialChain, c, s, interval=False):
    """Batched FK and its EXACT derivatives in (c_i, s_i), for complex arrays
    (N, n) or for Intervals (then every quantity encloses the true one).

    FK is multilinear: R = R_0 ... R_{n-1}, R_i = E0 + c_i Ec + s_i Es, so
      dR/dc_i = pre_i Ec suf_{i+1},   dp/dc_i = pre_i Ec S_i,
    pre_i = R_0..R_{i-1}, suf_i = R_i..R_{n-1}, S_i = a_i + R_{i+1} S_{i+1}
    (a_i = the translation after joint i). Same with Es for s_i."""
    mm, lift = _ops(interval)
    n = len(chain.axes)
    N = c.shape[0]
    I3 = np.broadcast_to(np.eye(3), (N, 3, 3))
    Rs = [lift(_E[ax][0]) + c[:, i, None, None] * _E[ax][1] + s[:, i, None, None] * _E[ax][2]
          for i, ax in enumerate(chain.axes)]
    pre = [lift(I3)]
    for i in range(n):
        pre.append(mm(pre[-1], Rs[i]))
    suf = [None] * (n + 1)
    suf[n] = lift(I3)
    for i in range(n - 1, -1, -1):
        suf[i] = mm(Rs[i], suf[i + 1])
    a = [np.broadcast_to(chain.after.get(i, np.zeros(3)), (N, 3))[..., None] for i in range(n)]
    S = [None] * n
    S[n - 1] = lift(a[n - 1])
    for i in range(n - 2, -1, -1):
        S[i] = lift(a[i]) + mm(Rs[i + 1], S[i + 1])
    p = lift(np.broadcast_to(chain.base, (N, 3))[..., None]) + mm(pre[1], S[0])
    dR = [(mm(mm(pre[i], _E[ax][1]), suf[i + 1]), mm(mm(pre[i], _E[ax][2]), suf[i + 1]))
          for i, ax in enumerate(chain.axes)]
    dp = [(mm(mm(pre[i], _E[ax][1]), S[i]), mm(mm(pre[i], _E[ax][2]), S[i]))
          for i, ax in enumerate(chain.axes)]
    return p, pre[n], dR, dp


@dataclass
class DoorSystem:
    chain: SerialChain
    label: int                      # joint index whose sign labels the pages (s_j = 0)
    spin: int                       # joint index used to fix the null spin (a neighbour)

    @property
    def n(self) -> int:
        return len(self.chain.axes)

    @property
    def size(self) -> int:
        return 2 * self.n + 1

    # P = dict(p0 (3), A (3), Rt (3,3), ab (2), r0 ()), each optionally with a
    # leading batch axis. F is AFFINE in P, so along a straight parameter
    # segment dF/dt = F(x, Q) - F(x, P) exactly.

    def Fb(self, X, P):
        """Batched residual: X (N, 2n+1) -> (N, 2n+1)."""
        n = self.n
        c, s, u = X[:, :n], X[:, n:2 * n], X[:, 2 * n]
        p, R, _, _ = _fk_parts(self.chain, c, s)
        Rt = np.broadcast_to(P["Rt"], (len(X), 3, 3))
        M = np.swapaxes(Rt, -1, -2) @ R
        p0 = np.broadcast_to(P["p0"], (len(X), 3))
        A = np.broadcast_to(P["A"], (len(X), 3))
        ab = np.broadcast_to(P["ab"], (len(X), 2))
        r0 = np.broadcast_to(P.get("r0", 0.0), (len(X),))
        return np.concatenate([
            c * c + s * s - 1.0,
            p[..., 0] - p0 - u[:, None] * A,
            np.stack([M[:, 0, 2], M[:, 1, 2], M[:, 1, 0]], axis=1),
            (np.sum(X * self._nu(P, len(X)), axis=1)
             - np.broadcast_to(P.get("rd", 0.0), (len(X),)))[:, None],
            (ab[:, 0] * c[:, self.spin] + ab[:, 1] * s[:, self.spin] - r0)[:, None],
        ], axis=1)

    def Jb(self, X, P):
        """Batched EXACT Jacobian (N, 2n+1, 2n+1)."""
        n = self.n
        N = len(X)
        c, s = X[:, :n], X[:, n:2 * n]
        _, _, dR, dp = _fk_parts(self.chain, c, s)
        Rt = np.broadcast_to(P["Rt"], (N, 3, 3))
        RtT = np.swapaxes(Rt, -1, -2)
        A = np.broadcast_to(P["A"], (N, 3))
        ab = np.broadcast_to(P["ab"], (N, 2))
        J = np.zeros((N, 2 * n + 1, 2 * n + 1), dtype=complex)
        ar = np.arange(n)
        J[:, ar, ar] = 2 * c
        J[:, ar, n + ar] = 2 * s
        for i in range(n):
            for k, col in ((0, i), (1, n + i)):
                J[:, n:n + 3, col] = dp[i][k][..., 0]
                dM = RtT @ dR[i][k]
                J[:, n + 3, col] = dM[:, 0, 2]
                J[:, n + 4, col] = dM[:, 1, 2]
                J[:, n + 5, col] = dM[:, 1, 0]
        J[:, n:n + 3, 2 * n] = -A
        J[:, n + 6, :] = self._nu(P, N)
        J[:, n + 7, self.spin] = ab[:, 0]
        J[:, n + 7, n + self.spin] = ab[:, 1]
        return J

    def door_row(self) -> np.ndarray:
        """The door hyperplane s_label = 0 as a linear form on x."""
        e = np.zeros(self.size, dtype=complex)
        e[self.n + self.label] = 1.0
        return e

    def _nu(self, P, N):
        return np.broadcast_to(P["nu"] if "nu" in P else self.door_row(), (N, self.size))

    def F(self, x, P):
        return self.Fb(np.asarray(x, dtype=complex)[None], P)[0]

    def J(self, x, P):
        return self.Jb(np.asarray(x, dtype=complex)[None], P)[0]


# --------------------------------------------------------------------------
# batched path tracking
# --------------------------------------------------------------------------


def _bsolve(A, b):
    """Batched solve; rows whose matrix is singular get NaN (and ok=False)."""
    try:
        return np.linalg.solve(A, b[..., None])[..., 0], np.ones(len(A), dtype=bool)
    except np.linalg.LinAlgError:
        out = np.full(b.shape, np.nan, dtype=complex)
        ok = np.zeros(len(A), dtype=bool)
        for k in range(len(A)):
            try:
                out[k] = np.linalg.solve(A[k], b[k])
                ok[k] = True
            except np.linalg.LinAlgError:
                pass
        return out, ok


@dataclass
class TrackStats:
    paths: int = 0
    finished: int = 0
    diverged: int = 0
    failed: int = 0
    steps: int = 0


def track_batch(H, Hx, Ht, X0, dt0=0.05, dt_max=0.1, max_iter=20000, blowup=1e7,
                stats: TrackStats | None = None):
    """Track every row of X0 from t = 0 to t = 1 along H(x, t) = 0, all paths
    at once (each with its own t and step). H, Hx, Ht take (X (k, m), t (k,)).
    Returns (X1, ok): endpoints and which paths arrived (residual < 1e-8)."""
    X = np.array(X0, dtype=complex)
    N = len(X)
    t = np.zeros(N)
    dt = np.full(N, dt0)
    status = np.zeros(N, dtype=int)            # 0 active, 1 arrived, -1 failed, -2 diverged
    if N == 0:
        return X, np.zeros(0, dtype=bool)

    def xdot(x, tt):
        d, ok = _bsolve(Hx(x, tt), Ht(x, tt))
        return -d, ok

    for _ in range(max_iter):
        act = np.flatnonzero(status == 0)
        if act.size == 0:
            break
        x, tt = X[act], t[act]
        h = np.minimum(dt[act], 1.0 - tt)
        k1, o1 = xdot(x, tt)
        k2, o2 = xdot(x + 0.5 * h[:, None] * k1, tt + 0.5 * h)
        k3, o3 = xdot(x + 0.5 * h[:, None] * k2, tt + 0.5 * h)
        k4, o4 = xdot(x + h[:, None] * k3, tt + h)
        y = x + h[:, None] * (k1 + 2 * k2 + 2 * k3 + k4) / 6.0
        good = o1 & o2 & o3 & o4 & np.all(np.isfinite(y), axis=1)
        conv = np.zeros(len(act), dtype=bool)
        for _c in range(4):
            d, ok = _bsolve(Hx(y, tt + h), H(y, tt + h))
            good &= ok & np.all(np.isfinite(d), axis=1)
            d = np.where(good[:, None], d, 0.0)
            y = y - d
            conv |= good & (np.linalg.norm(d, axis=1) < 1e-10 * (1 + np.linalg.norm(y, axis=1)))
        acc = good & conv
        ia = act[acc]
        X[ia], t[ia] = y[acc], tt[acc] + h[acc]
        dt[ia] = np.minimum(1.5 * dt[ia], dt_max)
        ir = act[~acc]
        dt[ir] *= 0.5
        status[ir[dt[ir] < 1e-13]] = -1
        status[ia[t[ia] >= 1.0 - 1e-15]] = 1
        status[act[np.linalg.norm(X[act], axis=1) > blowup]] = -2
        if stats is not None:
            stats.steps += 1
    done = np.flatnonzero(status == 1)
    if done.size:
        one = np.ones(done.size)
        for _ in range(6):
            d, ok = _bsolve(Hx(X[done], one), H(X[done], one))
            X[done] = np.where(ok[:, None], X[done] - np.nan_to_num(d), X[done])
        res = np.linalg.norm(H(X[done], one), axis=1)
        status[done[~(res < 1e-8)]] = -1
    if stats is not None:
        stats.paths += N
        stats.finished += int(np.sum(status == 1))
        stats.diverged += int(np.sum(status == -2))
        stats.failed += int(np.sum(status == -1))
    return X, status == 1


# The door equation is the LINEAR form nu.x = rd; the real door is
# nu = door_row() (s_label), rd = 0. Generic (nu, rd) make it a generic
# hyperplane, which is what the witness set and the trace test need.
KEYS = ("p0", "A", "Rt", "ab", "r0", "nu", "rd")


def _lerp_b(P, Q, t):
    """Per-path parameters on the segment P -> Q (leading axis = paths)."""
    out = {}
    for k in P:
        a, b = np.asarray(P[k]), np.asarray(Q[k])
        tt = t.reshape((-1,) + (1,) * a.ndim)
        out[k] = (1 - tt) * a + tt * b
    return out


def _full(sys: DoorSystem, P: dict) -> dict:
    out = {k: np.asarray(P.get(k, 0.0), dtype=complex) for k in KEYS}
    if "nu" not in P:
        out["nu"] = sys.door_row()
    return out


def track_params(sys: DoorSystem, X0, P, Q, stats: TrackStats | None = None):
    """Parameter homotopy F(x, P) -> F(x, Q) on the straight segment, batched."""
    P, Q = _full(sys, P), _full(sys, Q)
    return track_batch(lambda X, t: sys.Fb(X, _lerp_b(P, Q, t)),
                       lambda X, t: sys.Jb(X, _lerp_b(P, Q, t)),
                       lambda X, t: sys.Fb(X, Q) - sys.Fb(X, P),
                       np.atleast_2d(X0), stats=stats)


def _rand_params(rng, m: int, scale=1.0):
    def cz(*shape):
        return scale * (rng.normal(size=shape) + 1j * rng.normal(size=shape))
    return dict(p0=cz(3), A=cz(3), Rt=np.eye(3) + 0.5 * cz(3, 3), ab=cz(2), r0=cz(),
                nu=cz(m) / np.sqrt(m), rd=cz())


def _dedupe(sols, tol=1e-6):
    out = []
    for x in sols:
        if all(np.linalg.norm(x - y) > tol * (1 + np.linalg.norm(x)) for y in out):
            out.append(x)
    return out


@dataclass
class MonodromyResult:
    base: dict               # a GENERIC instance: generic task and generic hyperplane nu.x = rd
    solutions: list          # its witness points
    loops: int
    history: list = field(default_factory=list)
    trace: "TraceResult | None" = None


def _loops(sys, sols, P0, rng, stable_loops, max_loops, loops0=0, hist=None,
           verbose=False, stats=None):
    calm, loops, hist = 0, loops0, list(hist or [])
    while calm < stable_loops and loops < loops0 + max_loops:
        loops += 1
        P1, P2 = _rand_params(rng, sys.size), _rand_params(rng, sys.size)
        X, ok = track_params(sys, np.array(sols), P0, P1, stats)
        X, ok2 = track_params(sys, X[ok], P1, P2, stats)
        X, ok3 = track_params(sys, X[ok2], P2, P0, stats)
        before = len(sols)
        sols = _dedupe(sols + list(X[ok3]))
        hist.append(len(sols))
        calm = calm + 1 if len(sols) == before else 0
        if verbose:
            print(f"[monodromy] loop {loops}: {len(sols)} solutions", flush=True)
    return sols, loops, hist


def monodromy_solve(sys: DoorSystem, rng=None, stable_loops=5, max_loops=60,
                    verbose=False, stats: TrackStats | None = None) -> MonodromyResult:
    """Witness points of a generic instance by monodromy (all known solutions
    tracked together around random loops P0 -> P1 -> P2 -> P0). The base is
    built AROUND a random point x*: every parameter that x* must satisfy is
    solved for. The stopping rule is a heuristic; monodromy_certified() runs
    until the trace test passes."""
    rng = np.random.default_rng(1) if rng is None else rng
    n, m = sys.n, sys.size
    z = rng.normal(size=n) + 1j * rng.normal(size=n)
    c, s = np.cos(z), np.sin(z)
    u = complex(rng.normal() + 1j * rng.normal())
    x0 = np.concatenate([c, s, [u]])
    T, R = fk_cs(sys.chain, c, s)
    P0 = _rand_params(rng, m)
    P0["p0"] = T - u * P0["A"]
    # Rt must make M = Rt^T R have M[0,2] = M[1,2] = M[1,0] = 0: take Rt = R D
    # with D a random complex matrix of that pattern (M = D^T R^T R = D^T).
    D = np.eye(3, dtype=complex) + 0.5 * (rng.normal(size=(3, 3)) + 1j * rng.normal(size=(3, 3)))
    D[2, 0] = D[2, 1] = D[0, 1] = 0.0          # (D^T)[0,2], (D^T)[1,2], (D^T)[1,0]
    P0["Rt"] = np.linalg.solve(R.T, D)           # Rt^T R = D^T  <=>  R^T Rt = D
    P0["r0"] = P0["ab"][0] * c[sys.spin] + P0["ab"][1] * s[sys.spin]
    P0["rd"] = np.dot(P0["nu"], x0)
    assert np.linalg.norm(sys.F(x0, P0)) < 1e-9, np.linalg.norm(sys.F(x0, P0))
    sols, loops, hist = _loops(sys, [x0], P0, rng, stable_loops, max_loops,
                               verbose=verbose, stats=stats)
    return MonodromyResult(base=P0, solutions=sols, loops=loops, history=hist)


# --------------------------------------------------------------------------
# trace test: is the witness set complete?
# --------------------------------------------------------------------------


@dataclass
class TraceResult:
    ok: bool                 # the witness set is complete
    residual: float          # worst relative nonlinearity of the trace over the coordinates
    paths_failed: int


def trace_test(sys: DoorSystem, mono: MonodromyResult, rng=None, tol=1e-7,
               stats: TrackStats | None = None) -> TraceResult:
    """Certify (with probability one) that mono.solutions is the WHOLE witness set.

    Drop the door equation and the solutions form a curve E (15 unknowns, 14
    equations). The base instance cuts E with a GENERIC hyperplane nu.x = rd.
    Translate it, nu.x = rd + s: the sum of the cut points is an affine
    function of s if and only if no point is missing (a proper subset of the
    witness set of a curve never has an affine trace). Checked coordinate by
    coordinate at s = 0, s1, s2.

    The hyperplane must be generic: the door hyperplane s_label = 0 itself
    also cuts E at infinity, its trace is not affine even when complete."""
    rng = np.random.default_rng(7) if rng is None else rng
    P = _full(sys, mono.base)
    X0 = np.array(mono.solutions)
    t0 = X0.sum(axis=0)
    slopes, failed = [], 0
    for _ in range(2):
        sg = 0.5 * complex(rng.normal() + 1j * rng.normal())
        Q = dict(P)
        Q["rd"] = P["rd"] + sg
        X, ok = track_params(sys, X0, P, Q, stats)
        failed += int(np.sum(~ok))
        slopes.append((X.sum(axis=0) - t0) / sg)
    d1, d2 = slopes
    scale = np.abs(d1) + np.abs(d2) + np.max(np.abs(X0), axis=0) * 1e-12 + 1e-300
    res = float(np.max(np.abs(d1 - d2) / scale))
    return TraceResult(ok=(failed == 0 and res < tol), residual=res, paths_failed=failed)


def monodromy_certified(sys: DoorSystem, rng=None, verbose=False, max_rounds=10,
                        stats: TrackStats | None = None) -> MonodromyResult:
    """Monodromy until the TRACE TEST passes, not until the heuristic stops."""
    rng = np.random.default_rng(1) if rng is None else rng
    mono = monodromy_solve(sys, rng=rng, verbose=verbose, stats=stats)
    for _ in range(max_rounds):
        tr = trace_test(sys, mono, stats=stats)
        mono.trace = tr
        if verbose:
            print(f"[trace] {len(mono.solutions)} witness points: residual {tr.residual:.1e}, "
                  f"{'COMPLETE' if tr.ok else 'incomplete'}", flush=True)
        if tr.ok:
            break
        sols, loops, hist = _loops(sys, mono.solutions, mono.base, rng, 5, 60, mono.loops,
                                   mono.history, verbose=verbose, stats=stats)
        mono = MonodromyResult(mono.base, sols, loops, hist)
    return mono


def solve_instance(sys: DoorSystem, mono: MonodromyResult, P_real: dict, rng=None,
                   stats: TrackStats | None = None) -> list:
    """Every finite solution of the real instance (the door hyperplane unless
    P_real says otherwise): track the witness points from the generic base
    through a random complex detour. Paths to points of E at infinity
    diverge and are dropped."""
    rng = np.random.default_rng(2) if rng is None else rng
    Pmid = _rand_params(rng, sys.size)
    X, ok = track_params(sys, np.array(mono.solutions), mono.base, Pmid, stats)
    X, ok = track_params(sys, X[ok], Pmid, P_real, stats)
    return _dedupe(list(X[ok]))


def task_instance(surf, v: float, ab=(0.37, 0.81)) -> dict:
    """The real door instance of a weld-patch slice v: unknown u, door
    hyperplane s_label = 0 (any fixed ab with a, b != 0 picks one spin root;
    the other is q_k + pi)."""
    return dict(p0=(surf.p0 + v * surf.dv).astype(complex), A=surf.du.astype(complex),
                Rt=surf.rot.astype(complex), ab=np.asarray(ab, dtype=complex),
                r0=np.complex128(0.0), rd=np.complex128(0.0))


def edge_instance(surf, edge: str, ab=(0.37, 0.81)) -> dict:
    """The door instance on a patch edge: where a door curve meets the
    boundary. 'u0' / 'u1': u fixed, the unknown (still called u) is v;
    'v0' / 'v1': the slices v = 0, 1."""
    if edge in ("v0", "v1"):
        return task_instance(surf, float(edge[1]), ab)
    p0 = surf.p0 + float(edge[1]) * surf.du
    return dict(p0=p0.astype(complex), A=surf.dv.astype(complex),
                Rt=surf.rot.astype(complex), ab=np.asarray(ab, dtype=complex),
                r0=np.complex128(0.0), rd=np.complex128(0.0))


def real_doors(sys: DoorSystem, sols, P: dict, imag_tol=1e-6, rot_tol=1e-6) -> list:
    """Door configurations among the endpoints of a real instance: real,
    u in (0, 1), the tool rotation really equal to R_t (drops the isolated
    half-turn solutions of the three orientation equations), q_label = 0 (not
    pi). Returns (u, q) pairs, sorted by u. One door POINT of the page has
    several joint spellings here (the other sign flips, and the two spin
    roots q_k, q_k + pi): group them by (u, arm angle) to compare with a chart."""
    n = sys.n
    Rt = np.asarray(P["Rt"]).real
    out = []
    for x in sols:
        if np.max(np.abs(x.imag)) > imag_tol:
            continue
        x = x.real
        c, s, u = x[:n], x[n:2 * n], float(x[2 * n])
        if not (0.0 < u < 1.0) or c[sys.label] < 0:
            continue
        _, R = fk_cs(sys.chain, c, s)
        if np.linalg.norm(Rt.T @ R.real - np.eye(3)) > rot_tol:
            continue
        q = np.arctan2(s, c)
        if any(abs(u - v) < 1e-7 and np.all(np.abs(np.angle(np.exp(1j * (q - p)))) < 1e-6)
               for v, p in out):
            continue
        out.append((u, q))
    return sorted(out, key=lambda t: t[0])


# --------------------------------------------------------------------------
# certify each real door point: Krawczyk on a tiny box (method 4 where it works)
# --------------------------------------------------------------------------


def _interval_FJ(sys: DoorSystem, Xb, P: dict):
    """Interval enclosures of F and J over boxes Xb (Interval, (N, m)), for a
    REAL instance P. Same formulas as Fb/Jb, in outward-rounded arithmetic."""
    from .interval import Interval, imatmul
    n, m = sys.n, sys.size
    N = Xb.shape[0]
    c, s, u = Xb[:, :n], Xb[:, n:2 * n], Xb[:, 2 * n]
    p, R, dR, dp = _fk_parts(sys.chain, c, s, interval=True)
    Pr = {k: np.real(v) for k, v in _full(sys, P).items()}
    RtT = np.broadcast_to(Pr["Rt"].T, (N, 3, 3))
    M = imatmul(RtT, R)
    F = Interval(np.zeros((N, m)))
    F[:, :n] = c * c + s * s - 1.0
    for r in range(3):
        F[:, n + r] = p[:, r, 0] - Pr["p0"][r] - u * Pr["A"][r]
    F[:, n + 3] = M[:, 0, 2]
    F[:, n + 4] = M[:, 1, 2]
    F[:, n + 5] = M[:, 1, 0]
    door = Interval(np.zeros(N))
    for k in np.flatnonzero(Pr["nu"]):
        door = door + Xb[:, k] * Pr["nu"][k]
    F[:, n + 6] = door - Pr["rd"]
    F[:, n + 7] = c[:, sys.spin] * Pr["ab"][0] + s[:, sys.spin] * Pr["ab"][1] - Pr["r0"]
    J = Interval(np.zeros((N, m, m)))
    for i in range(n):
        J[:, i, i] = c[:, i] * 2.0
        J[:, i, n + i] = s[:, i] * 2.0
        for k, col in ((0, i), (1, n + i)):
            J[:, n:n + 3, col] = dp[i][k][:, :, 0]
            dM = imatmul(RtT, dR[i][k])
            J[:, n + 3, col] = dM[:, 0, 2]
            J[:, n + 4, col] = dM[:, 1, 2]
            J[:, n + 5, col] = dM[:, 1, 0]
    J[:, n:n + 3, 2 * n] = Interval(np.broadcast_to(-Pr["A"], (N, 3)))
    J[:, n + 6, :] = Interval(np.broadcast_to(Pr["nu"], (N, m)))
    J[:, n + 7, sys.spin] = Interval(np.full(N, Pr["ab"][0]))
    J[:, n + 7, n + sys.spin] = Interval(np.full(N, Pr["ab"][1]))
    return F, J


@dataclass
class RootCertificate:
    certified: bool          # a UNIQUE exact root of the real system lies in the box
    center: np.ndarray
    radius: float            # box half-width (same in every coordinate)


def certify_root(sys: DoorSystem, x, P: dict, r0=1e-10, tries=12, polish=True) -> RootCertificate:
    """Krawczyk test on the box x +- r: K(X) = x - Y F(x) + (I - Y J(X))(X - x),
    Y = J(x)^-1. K(X) strictly inside X proves a unique real root in X
    (interval arithmetic with outward rounding). A Newton-polished root needs
    only a tiny box -- where Krawczyk works (method 4 failed on big cells)."""
    from .interval import Interval, imatmul, imatvec
    x = np.real(np.asarray(x)).astype(float)
    m = sys.size
    Pr = {k: np.real(v) for k, v in _full(sys, P).items()}
    for _ in range(4 if polish else 0):                    # polish in real arithmetic
        x = x - np.linalg.solve(np.real(sys.J(x, Pr)), np.real(sys.F(x, Pr)))
    Y = np.linalg.inv(np.real(sys.J(x, Pr)))
    r = r0 * (1.0 + np.max(np.abs(x)))
    for _ in range(tries):
        X = Interval.around(x[None], np.full((1, m), r))
        Fx, _ = _interval_FJ(sys, Interval(x[None]), Pr)
        _, JX = _interval_FJ(sys, X, Pr)
        YF = imatvec(Y[None], Fx)
        IYJ = Interval(np.eye(m)[None]) - imatmul(Y[None], JX)
        K = Interval(x[None]) - YF + imatvec(IYJ, X - x[None])
        if np.all(K.lo > X.lo) and np.all(K.hi < X.hi):
            return RootCertificate(True, x, float(r))
        r *= 10.0
    return RootCertificate(False, x, float(r))
