import asyncio, os, sys, time, sqlite3, shutil, traceback
import kremory_py_spike as K

DIM = 16

class Counter:
    def __init__(self): self.n = 0

def make_embedder(counter, dim=DIM, mode="ok"):
    async def embed(text: str):
        counter.n += 1
        await asyncio.sleep(0)                 # force a real suspension point
        if mode == "raise":
            raise ValueError("boom from python embedder")
        if mode == "wrongdim":
            return [0.1] * (dim + 3)
        v = [0.0] * dim
        for i, b in enumerate(text.encode()):
            v[i % dim] += b / 255.0
        norm = max(sum(x * x for x in v) ** 0.5, 1e-6)
        return [x / norm for x in v]
    return embed

def rowcount(db):
    """COUNT(*) is known-broken on kremory's vector-indexed tables; count rows."""
    try:
        c = sqlite3.connect(db)
        n = len(c.execute("SELECT recorded_at FROM entities").fetchall())
        m = len(c.execute("SELECT recorded_at FROM facts").fetchall())
        c.close()
        return n, m
    except Exception as e:
        return ("ERR:" + str(e), None)

async def one_run(i, workdir, mode="ok"):
    db = os.path.join(workdir, f"run{i}.db")
    for suf in ("", "-wal", "-shm"):
        p = db + suf
        if os.path.exists(p): os.remove(p)
    ctr = Counter()
    t0 = time.monotonic()
    err = None
    res = None
    try:
        res = await asyncio.wait_for(K.round_trip(make_embedder(ctr, mode=mode), DIM, db), timeout=60)
    except Exception as e:
        err = f"{type(e).__name__}: {e}"
    dt = time.monotonic() - t0
    return dict(i=i, mode=mode, secs=round(dt, 2), py_calls=ctr.n, err=err, res=res, rows=rowcount(db))

async def main():
    workdir = "/tmp/kremory-py-spike/dbs"
    os.makedirs(workdir, exist_ok=True)
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 3
    mode = sys.argv[2] if len(sys.argv) > 2 else "ok"

    # sanity: bridge constructs as Arc<dyn DynEmbeddingProvider>, Send+Sync asserted in Rust
    print("bridge_is_dyn_object:", K.bridge_is_dyn_object(make_embedder(Counter()), DIM), flush=True)

    fails = 0
    for i in range(n):
        r = await one_run(i, workdir, mode)
        ok = (r["err"] is None and r["py_calls"] >= 1 and r["res"] and r["res"]["rust_calls"] == r["py_calls"]
              and len(r["res"]["facts"]) > 0) if mode == "ok" else None
        print(f"run={i:02d} mode={mode} secs={r['secs']} py_calls={r['py_calls']} "
              f"rust_calls={(r['res'] or {}).get('rust_calls')} facts={len((r['res'] or {}).get('facts', []))} "
              f"rendered_len={(r['res'] or {}).get('rendered_len')} rows(entities,facts)={r['rows']} "
              f"err={r['err']} ok={ok}", flush=True)
        if mode == "ok" and not ok: fails += 1
    print(f"DONE n={n} mode={mode} fails={fails}", flush=True)
    return 1 if (mode == "ok" and fails) else 0

if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
