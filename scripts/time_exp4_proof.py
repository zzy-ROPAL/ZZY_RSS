"""EXP 4 proof timing, broken down and repeated.

'The infeasibility proof time' is ambiguous, so time all three things it
could mean, separately:
  BUILD   the eight pages + all 16 door fiber certifications. A prerequisite,
          not the proof.
  WALL    IJRR Algorithms 1-4 on the start region: train, sample the wall,
          reconstruct the closed polyline, verify every facet. This IS the
          infeasibility proof for the start page.
  VERDICT flood.run_proof -- the wall again, plus following every open door
          out of the region and discovering the goal on the partner page.
"""
import sys, time, statistics as st
from pathlib import Path
_R = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_R / "src"))
sys.path.insert(0, str(_R / "scripts"))
import numpy as np
import run_exp4 as E
from zzy_rss.flood import label_regions, run_proof

REPS = 5
for res in (41, 61, 81, 101):
    fbm = 0.5*np.hypot(1/(res-1), 2/(res-1))
    tb=[]; tw=[]; tv=[]; info={}
    for rep in range(REPS):
        t0=time.perf_counter()
        cfg,arm,weld,scene = E.make(res, fbm)
        pms,sig,no,nb = E.pages_and_sigma(cfg,arm,weld,scene)
        tb.append(time.perf_counter()-t0)
        pmA=pms[E.START_PAGE]; labA,regsA=label_regions(cfg,pmA)
        d=min([x for x in sig[E.START_PAGE]
               if 2 in x.vanishing and x.__dict__.get("_state")=="open"],
              key=lambda z: abs(z.anchor[0]-E.EXP4["door_y"]))
        i=int(np.argmin(np.abs(pmA.y_grid-d.anchor[0])))
        j=int(np.argmin(np.abs(pmA.theta_grid-d.anchor[1])))
        kA=int(labA[i,j])
        t0=time.perf_counter()
        poly,rep_,w,_,_,_,_ = E.prove(cfg,arm,weld,scene,pmA,labA,kA,
                                      sig[E.START_PAGE],iters=14,verbose=False)
        tw.append(time.perf_counter()-t0)
        t0=time.perf_counter()
        r = run_proof(cfg,arm,weld,scene,pms,sig,verbose=False,alg4_iters=14,
                      seeds=[(E.START_PAGE,kA)],n_train_max=450,n_invalid_max=150)
        tv.append(time.perf_counter()-t0)
        cont=E.containment_check(poly,pmA,labA,kA,regsA)
        info=dict(facets=f"{rep_.n_passed}/{rep_.n_facets}", closed=poly.closed,
                  gamma=w.gamma, ok=bool(rep_.ok and poly.closed and cont["ok"]),
                  proven=r.proven, via=r.goal_reached_via)
    f=lambda v: f"{st.median(v):6.2f} ({min(v):.2f}-{max(v):.2f})"
    print(f"res {res:3d} | build {f(tb)} | WALL {f(tw)} | verdict {f(tv)} | "
          f"{info['facets']:>9} closed={info['closed']} g={info['gamma']:.1f} "
          f"proof={info['ok']} via={info['via']}", flush=True)
