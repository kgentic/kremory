import json, sys
from pathlib import Path
sys.path.insert(0, '.')
import evidence_eval as ee

turns_by_sample = ee.load_turns(Path('data/locomo10.json'))

def probe(path, pool=50, win=10):
    rows = json.loads(Path(path).read_text())["results"]
    cur_r, orc_r, cur_rr, orc_rr, n = [], [], [], [], 0
    for row in rows:
        sample = row.get("sample_id") or row.get("sample") or ""
        turns = turns_by_sample.get(sample, {})
        ev = [turns[e] for e in ee.evidence_ids(row) if e in turns]
        if not ev:
            continue
        mems = [ee.norm(m) for m in (row.get("recalled_memories") or [])][:pool]
        if not mems:
            continue
        n += 1
        gains = [sum(1 for t in ev if ee.turn_in(t, h)) for h in mems]
        cg = gains[:win]
        cur_found = set()
        for i, g in enumerate(cg):
            if g:
                for t in ev:
                    if ee.turn_in(t, mems[i]):
                        cur_found.add(t)
        f = next((i + 1 for i, g in enumerate(cg) if g), None)
        cur_r.append(len(cur_found) / len(ev)); cur_rr.append(1.0 / f if f else 0.0)
        order = sorted(range(len(gains)), key=lambda i: -gains[i])[:win]
        orc_found = set()
        for i in order:
            if gains[i]:
                for t in ev:
                    if ee.turn_in(t, mems[i]):
                        orc_found.add(t)
        of = next((r + 1 for r, i in enumerate(order) if gains[i]), None)
        orc_r.append(len(orc_found) / len(ev)); orc_rr.append(1.0 / of if of else 0.0)
    m = lambda x: sum(x) / len(x) * 100 if x else 0.0
    print(f"{Path(path).name}  n={n}  pool={pool} -> window={win}")
    print(f"  recall@{win}:  current {m(cur_r):5.1f}%   ORACLE {m(orc_r):5.1f}%   headroom {m(orc_r)-m(cur_r):+5.1f}pt")
    print(f"  MRR:         current {m(cur_rr):5.1f}%   ORACLE {m(orc_rr):5.1f}%   headroom {m(orc_rr)-m(cur_rr):+5.1f}pt")

for p in sys.argv[1:]:
    probe(p)
