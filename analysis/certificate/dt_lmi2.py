import os as _os
_HERE = _os.path.dirname(_os.path.abspath(__file__))
import numpy as np, cvxpy as cp
exec(open(_os.path.join(_HERE, 'dt_lmi.py')).read().split('def common_P')[0])
def common_P(As,rho):
    n=As[0].shape[0]; P=cp.Variable((n,n),symmetric=True)
    cons=[P>>1e-3*np.eye(n), cp.trace(P)==n]+[A.T@P@A-rho**2*P<<-1e-7*np.eye(n) for A in As]
    pr=cp.Problem(cp.Minimize(0),cons)
    try: pr.solve(solver='SCS',eps=1e-9,max_iters=200000)
    except BaseException: return False,None
    if P.value is None: return False,None
    Pv=P.value; Pv=(Pv+Pv.T)/2
    ok=min(np.linalg.eigvalsh(Pv))>0 and all(max(np.linalg.eigvalsh(A.T@Pv@A-rho**2*Pv))<=0 for A in As)
    return ok,Pv
if __name__=='__main__':
  for kt in (200,600):
    for delay in (0,1,2):
        S=np.diag([1,1,1,1]+[1/kt]*delay)
        As=[S@closed_d(kt,Kp,Kd,kap,lk,delay)@np.linalg.inv(S) for Kp,Kd,(kap,lk) in V]
        srad=max(max(abs(np.linalg.eigvals(A))) for A in As)
        if srad>=1: print(f"k_tau={kt} delay={delay}: worst corner radius {srad:.3f} -> a corner is unstable, no certificate",flush=True); continue
        ok,_=common_P(As,0.99999)
        if not ok: print(f"k_tau={kt} delay={delay}: worst corner radius {srad:.3f}; no common quadratic P found",flush=True); continue
        lo,hi=srad,0.99999
        for _ in range(10):
            mid=(lo+hi)/2; ok,_=common_P(As,mid)
            if ok: hi=mid
            else: lo=mid
        print(f"k_tau={kt} delay={delay}: worst corner radius {srad:.3f}; common P found, rho={hi:.4f}, decay {-np.log(hi)/DT:.3f} 1/s",flush=True)
