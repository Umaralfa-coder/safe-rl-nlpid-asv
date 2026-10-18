import os as _os
_HERE = _os.path.dirname(_os.path.abspath(__file__))
import numpy as np, itertools, cvxpy as cp
import importlib.util,sys
spec=importlib.util.spec_from_file_location('d',_os.path.join(_HERE,'dt_check.py'))
src=open(_os.path.join(_HERE, 'dt_check.py')).read().split('for kt in (200,600):')[0]
exec(src)
def closed_d(kt,Kp,Kd,kap,lk,delay):
    # generalise to delay steps (0,1,2)
    a=AF; n=4+delay
    I=np.eye(n)
    raw=(I[0]-I[2])/DT; rf=a*I[3]+(1-a)*raw
    v=-kt*((Kp+lk)*I[0]+(Kd+kap)*rf)
    tau = v if delay==0 else I[4+delay-1]   # oldest stored command
    A=np.zeros((n,n)); A[0:2,:]=Ad@I[0:2,:]+Bd@tau[None,:]; A[2]=I[0]; A[3]=rf
    if delay>=1:
        A[4]=v
        for j in range(5,4+delay): A[j]=I[j-1]
    return A
def common_P(As,rho):
    n=As[0].shape[0]; P=cp.Variable((n,n),symmetric=True)
    cons=[P>>np.eye(n)]+[A.T@P@A-rho**2*P<<0 for A in As]
    pr=cp.Problem(cp.Minimize(0),cons); pr.solve(solver='CLARABEL')
    return pr.status in ('optimal','optimal_inaccurate'), P.value
for kt in (200,600):
    for delay in (0,1,2):
        As=[closed_d(kt,Kp,Kd,kap,lk,delay) for Kp,Kd,(kap,lk) in V]
        srad=max(max(abs(np.linalg.eigvals(A))) for A in As)
        lo,hi=srad,1.0; ok_hi,_=common_P(As,0.99999)
        if not ok_hi: print(f"k_tau={kt} delay={delay}: worst vertex radius {srad:.3f}; NO common P (switching not certified)"); continue
        for _ in range(14):
            mid=(lo+hi)/2; ok,_=common_P(As,mid)
            if ok: hi=mid
            else: lo=mid
        print(f"k_tau={kt} delay={delay}: worst vertex radius {srad:.3f}; common P with rho={hi:.4f} -> decay {-np.log(hi)/DT:.3f} 1/s")
