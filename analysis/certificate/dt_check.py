import numpy as np, itertools
from scipy.linalg import expm
K,T,DT,AF=1.5e-3,0.68,0.05,0.5
def plant():
    Ac=np.array([[0,1],[0,-1/T]]); Bc=np.array([[0],[K/T]])
    M=np.zeros((3,3)); M[:2,:2]=Ac; M[:2,2:]=Bc; E=expm(M*DT)
    return E[:2,:2],E[:2,2:]
Ad,Bd=plant()
def closed(kt,Kp,Kd,kap,lk,delay=1,filt=True):
    a=AF if filt else 0.0
    # state: psi, r, psi_prev, rf_prev, [tau_prev]
    n=5 if delay else 4
    # express rf_k and v_k as row vectors of current state
    raw=np.zeros(n); raw[0]=1/DT; raw[2]=-1/DT
    rf=np.zeros(n); rf[3]=a; rf+= (1-a)*raw
    v=-kt*((Kp+lk)*np.eye(n)[0]+(Kd+kap)*rf)
    tau=np.eye(n)[4] if delay else v
    Acl=np.zeros((n,n))
    Acl[0:2,:]=Ad@np.eye(n)[0:2,:]+Bd@tau[None,:]
    Acl[2,:]=np.eye(n)[0]; Acl[3,:]=rf
    if delay: Acl[4,:]=v
    return Acl
def cont(kt,Kp,Kd,kap,lk):
    return np.array([[0,1],[-K*kt/T*(Kp+lk),-(1+K*kt*(Kd+kap))/T]])
V=list(itertools.product([4.27,20.0],[2.5,10.0],[(0,0),(2.8,0.028),(2.8,1.4)]))
def report(kt,delay,filt):
    rho=[];fr=[]
    for Kp,Kd,(kap,lk) in V:
        ev=np.linalg.eigvals(closed(kt,Kp,Kd,kap,lk,delay,filt)); i=np.argmax(abs(ev))
        rho.append(abs(ev[i])); fr.append(abs(np.angle(ev[i]))/(2*np.pi*DT))
    j=int(np.argmax(rho)); return max(rho), V[j], fr[j]
for kt in (200,600):
    for delay,filt in ((0,False),(0,True),(1,True)):
        r,v,f=report(kt,delay,filt)
        print(f"k_tau={kt} delay={delay} filter={filt}: worst spectral radius {r:.3f} at Kp,Kd,(kap,lk)={v}, mode {f:.2f} Hz")
    # continuous: max real part
    mr=max(max(np.linalg.eigvals(cont(kt,Kp,Kd,kap,lk)).real) for Kp,Kd,(kap,lk) in V)
    print(f"k_tau={kt} continuous: max Re(eig) {mr:.2f}")
print()
for name,Kp,Kd in (('hand-tuned',13.7,7.9),('learned',20.0,2.7)):
    for kt in (200,600):
        for delay in (0,1):
            ev=np.linalg.eigvals(closed(kt,Kp,Kd,0,0,delay,True)); i=np.argmax(abs(ev))
            print(f"{name:10s} k_tau={kt} delay={delay}: radius {abs(ev[i]):.3f}, mode {abs(np.angle(ev[i]))/(2*np.pi*DT):.2f} Hz")
