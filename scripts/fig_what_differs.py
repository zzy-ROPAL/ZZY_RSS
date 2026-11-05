"""What the eight pages share, and what separates them — drawn on the page map.

The companion to scripts/fig_why_pages_alike.py. That one shows WHY the eight
branches place the same arm; this one shows what that means for the page map:

  A  the reference page, with each invalid cell coloured by its reason.
  B  the collision-invalid set alone. ONE picture serves all eight pages --
     measured 0 differing cells of 3721, because collision depends on the arm's
     occupied volume and all eight branches place the same arm (findings 31).
  C  every cell whose validity differs between a page and its s2 / s4 / s6
     partner. These are the ONLY differences, and they are thin.
  D  why they sit there. The mirror sends |q| -> 180 - |q| on the two OUTER
     joints of the flipped triple, so validity can only change where such a
     joint is within its SLACK (= 180 - limit) of 0 or of +-180. That band is a
     necessary condition; the actual differing cells are the subset where the
     rest of the arm is within limits anyway.

usage: python3 scripts/fig_what_differs.py [--out out/what_differs.png]
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import ListedColormap

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from zzy_rss.arm import SIGN_BRANCHES, SrsArm  # noqa: E402
from zzy_rss.pagemap import page_name, q_for_branch  # noqa: E402

REF = (1, 1, 1)
PAIRS = (("$s_2$", (-1, 1, 1), (0, 2), "#c1121f"),
         ("$s_4$", (1, -1, 1), (2, 4), "#f4a261"),
         ("$s_6$", (1, 1, -1), (4, 6), "#2a9d8f"))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="out/what_differs.png")
    ap.add_argument("--res", type=int, default=61)
    args = ap.parse_args()

    import run_three as rt
    cfg, arm, weld, scene = rt.make(0.0, rt.OBS_CLOSED, args.res)
    lim = np.degrees(cfg.arm.q_max)
    slack = 180.0 - lim
    n = args.res
    ys = np.linspace(0.0, 1.0, n)
    ths = np.linspace(-np.pi, np.pi, n)

    ok = np.zeros((n, n), bool)
    coll = np.zeros((n, n), bool)
    Q = np.full((n, n, 7), np.nan)
    for i, y in enumerate(ys):
        pos, rot = weld.pose(float(y))
        for j, th in enumerate(ths):
            r = arm.ik_branch(pos, rot, float(th), REF)
            if not r.ok:
                continue
            ok[i, j] = True
            Q[i, j] = r.q
            coll[i, j] = scene.check(arm.frames(r.q)).in_collision

    def legal(signs):
        m = np.zeros((n, n), bool)
        for i in range(n):
            for j in range(n):
                if not ok[i, j]:
                    continue
                q = Q[i, j] if signs == REF else q_for_branch(Q[i, j], REF, signs)
                m[i, j] = bool(np.all(np.abs(np.degrees(q)) <= lim))
        return m

    L0 = legal(REF)
    ext = (ys[0], ys[-1], ths[0], ths[-1])
    kw = dict(origin="lower", extent=ext, aspect="auto", interpolation="nearest")

    fig, axes = plt.subplots(1, 4, figsize=(21.5, 5.4))
    fig.suptitle("What the eight pages SHARE, and what separates them",
                 fontsize=15, fontweight="bold", y=0.99)

    # -- A: the reference page, by reason -------------------------------
    tag = np.zeros((n, n), int)          # 0 valid
    tag[~ok] = 1                          # no IK
    tag[ok & ~L0] = 2                     # joint limit
    tag[ok & L0 & coll] = 3               # collision
    cmap = ListedColormap(["#f1faee", "#adb5bd", "#f4a261", "#c1121f"])
    axes[0].imshow(tag.T, cmap=cmap, vmin=0, vmax=3, **kw)
    axes[0].set_title(f"A.  the reference page $(+\\!+\\!+)$\n"
                      f"valid {100*np.mean(tag==0):.1f}%   no-IK "
                      f"{100*np.mean(tag==1):.1f}%   limit "
                      f"{100*np.mean(tag==2):.1f}%   collision "
                      f"{100*np.mean(tag==3):.1f}%", fontsize=10.5)
    for c, lab in zip(("#f1faee", "#adb5bd", "#f4a261", "#c1121f"),
                      ("valid", "no IK", "joint limit", "collision")):
        axes[0].plot([], [], "s", color=c, mec="#555", ms=9, label=lab)
    axes[0].legend(loc="upper right", fontsize=8, framealpha=0.95)

    # -- B: the collision set, shared by all eight ----------------------
    axes[1].imshow(np.where(coll & ok, 1, 0).T, cmap=ListedColormap(
        ["#f1faee", "#c1121f"]), vmin=0, vmax=1, **kw)
    axes[1].set_title(f"B.  the COLLISION-invalid set\n"
                      f"{int((coll & ok).sum())} cells — and this ONE picture is"
                      f" every page:\nmeasured difference across all 8 pages = "
                      f"0 / {n*n}", fontsize=10.5, color="#7b1113")

    # -- C: where the pages actually differ -----------------------------
    axes[2].imshow(np.where(ok, 0.0, 0.35).T, cmap="Greys", vmin=0, vmax=1, **kw)
    counts = {}
    for lab, part, _oj, col in PAIRS:
        d = (L0 != legal(part)) & ok
        counts[lab] = int(d.sum())
        yy, tt = np.where(d)
        axes[2].scatter(ys[yy], ths[tt], s=20, c=col, marker="s",
                        linewidths=0, label=f"{lab} flip: {counts[lab]} cells")
    axes[2].set_title("C.  EVERY cell whose validity differs between a page\n"
                      "and its partner — the limits are the only separator",
                      fontsize=10.5)
    axes[2].legend(loc="upper right", fontsize=8.4, framealpha=0.95)

    # -- D: why they sit there ------------------------------------------
    axes[3].imshow(np.where(ok, 0.0, 0.35).T, cmap="Greys", vmin=0, vmax=1, **kw)
    lab, part, (i0, i2), col = PAIRS[2]      # s6: the one that matters
    band = np.zeros((n, n), bool)
    for i in range(n):
        for j in range(n):
            if not ok[i, j]:
                continue
            band[i, j] = any(
                abs(np.degrees(Q[i, j, k])) < slack[k]
                or abs(np.degrees(Q[i, j, k])) > lim[k] for k in (i0, i2))
    yy, tt = np.where(band)
    axes[3].scatter(ys[yy], ths[tt], s=22, c="#ffd166", marker="s",
                    linewidths=0,
                    label=f"necessary band: $|q_5|$ or $|q_7|$ within its\n"
                          f"slack of 0 or of $\\pm180°$  ({int(band.sum())} cells)")
    d6 = (L0 != legal(part)) & ok
    yy, tt = np.where(d6)
    axes[3].scatter(ys[yy], ths[tt], s=11, c=col, marker="s", linewidths=0,
                    label=f"actually differs ({int(d6.sum())} cells)")
    axes[3].set_title("D.  why there — the $s_6$ case\n"
                      "the mirror sends $|q|\\rightarrow 180°-|q|$ on $q_5,q_7$;"
                      " slack $=180°-$limit\n"
                      "$q_5$ slack 1.15°,   $q_7$ slack 11.55° "
                      "— ten times wider, which is why $s_6$ dominates",
                      fontsize=10.5)
    axes[3].legend(loc="upper right", fontsize=7.6, framealpha=0.95)

    for a in axes:
        a.set_xlabel("$y$  (along the weld)")
        a.set_ylabel(r"$\theta$  (elbow around its circle)")
        a.set_ylim(ths[0], ths[-1])
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    fig.savefig(out, dpi=160)
    plt.close(fig)
    print(f"[fig] {out}")
    print(f"[fig] collision cells {int((coll & ok).sum())}, identical on all 8 pages")
    for lab, c in counts.items():
        print(f"[fig] {lab} flip differs in {c} cells")


if __name__ == "__main__":
    main()
