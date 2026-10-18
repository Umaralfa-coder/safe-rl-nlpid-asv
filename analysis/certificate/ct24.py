import numpy as np, itertools, cvxpy as cp
K,T,kt=1.5e-3,0.68,200.0
V=[]
for s in (0.4,1.0):
    for Kp,Kd,(kap,lk) in itertools.product([4.27,20.0],[2.5,10.0],[(0,0),(2.8,0.028),(2.8,1.4)]):
        g=s*K*kt/T
        V.append(np.array([[0,1],[-g*(Kp+lk),-(1/T+g*(Kd+kap))]]))
def feas(a):
    P=cp.Variable((2,2),symmetric=True)
    cons=[P>>1e-3*np.eye(2),P[0,0]==1]+[A.T@P+P@A+a*P<<0 for A in V]
    pr=cp.Problem(cp.Minimize(0),cons)
    try: pr.solve(solver='SCS',eps=1e-9,max_iters=100000)
    except BaseException: return None
    if P.value is None: return None
    Pv=(P.value+P.value.T)/2
    if min(np.linalg.eigvalsh(Pv))<=0: return None
    if max(max(np.linalg.eigvalsh(A.T@Pv+Pv@A+a*Pv)) for A in V)>1e-9: return None
    return Pv
lo,hi=0.0,3.0; best=None
for _ in range(25):
    m=(lo+hi)/2; Pv=feas(m)
    if Pv is not None: lo,best=m,Pv
    else: hi=m
print('alpha0 (24 vertices) =',round(lo,3)); print('P =',np.round(best,3))
# check paper P with 12 vertices
Pp=np.array([[1,.2],[.2,.22]])
a12=[max(np.linalg.eigvalsh(A.T@Pp+Pp@A+0.55*Pp)) for A in V[12:]]
a24=[max(np.linalg.eigvalsh(A.T@Pp+Pp@A+0.55*Pp)) for A in V[:12]]
print('paper P, s=1 vertices max eig',max(a12),' s=0.4 vertices max eig',max(a24))
Pi=np.linalg.inv(best)
c=min(np.radians(27)**2/(2*Pi[0,0]),1/(2*Pi[1,1]))
b0=0.524*abs(best[0,1])+1*abs(best[1,1])
print('c =',round(c,4),'beta0 =',round(b0,3),'wbar needed <',round(lo*c/b0,3))
print('|e|max deg',np.degrees(np.sqrt(2*c*Pi[0,0])),'|edot|max',np.sqrt(2*c*Pi[1,1]))
np.save('P24.npy',best)
