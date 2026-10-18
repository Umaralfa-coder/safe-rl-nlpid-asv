import os as _os
_HERE = _os.path.dirname(_os.path.abspath(__file__))
import numpy as np, itertools, sys
exec(open(_os.path.join(_HERE, 'dt_lmi2.py')).read().split("if __name__=='__main__':")[0])
kt=200; delay=1
for kdmax,kmax in ((10,2.8),(8,2.8),(8,1.0),(10,1.0)):
    VV=list(itertools.product([4.27,20.0],[2.5,kdmax],[(0,0),(kmax,0.01*kmax),(kmax,0.5*kmax)]))
    S=np.diag([1,1,1,1,1/kt])
    As=[S@closed_d(kt,Kp,Kd,kap,lk,delay)@np.linalg.inv(S) for Kp,Kd,(kap,lk) in VV]
    srad=max(max(abs(np.linalg.eigvals(A))) for A in As)
    ok,_=common_P(As,0.99999)
    print(f"delay=1, Kd<= {kdmax}, kappa<= {kmax}: worst radius {srad:.4f}, common P {'FOUND' if ok else 'not found'}",flush=True)
