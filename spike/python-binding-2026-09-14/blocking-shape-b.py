import asyncio, os, sys
import kremory_py_spike as K
DIM=16
class C:
    def __init__(self): self.n=0
async def main():
    c=C()
    async def e(t):
        c.n+=1; await asyncio.sleep(0)
        v=[0.0]*DIM
        for i,b in enumerate(t.encode()): v[i%DIM]+=b/255.0
        n=max(sum(x*x for x in v)**0.5,1e-6); return [x/n for x in v]
    p="/tmp/kremory-py-spike/blocking.db"
    for s in ("","-wal","-shm"):
        if os.path.exists(p+s): os.remove(p+s)
    print(f"pid={os.getpid()} calling round_trip_blocking (loop IS running on this thread)", flush=True)
    n = K.round_trip_blocking(e, DIM, p)   # blocking, no await
    print("returned calls=", n, flush=True)
asyncio.run(main())
