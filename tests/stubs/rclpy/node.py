import simclock
class _P:
    def __init__(s,v): s.value=v
class _Log:
    def __init__(s): s.msgs=[]
    def _l(s,m,**k): s.msgs.append(m)
    info=warn=error=debug=_l
class _D:
    def __init__(s,ns): s.nanoseconds=ns
class _T:
    def __init__(s,ns): s.ns=ns; s.nanoseconds=ns   # real rclpy Time has .nanoseconds
    def __sub__(s,o): return _D(s.ns-o.ns)
class _Clock:
    def now(s): return _T(int(simclock.t*1e9))
class _Pub:
    def __init__(s): s.last=None
    def publish(s,m): s.last=m
class Node:
    overrides={}
    def __init__(s,name): s._p={}; s._log=_Log(); s._pubs={}
    def declare_parameter(s,n,d): s._p[n]=Node.overrides.get(n,d)
    def get_parameter(s,n): return _P(s._p[n])
    def get_logger(s): return s._log
    def get_clock(s): return _Clock()
    def create_timer(s,*a): return None
    def create_subscription(s,*a): return None
    def create_publisher(s,t,topic,q): p=_Pub(); s._pubs[topic]=p; return p
    def destroy_node(s): pass
