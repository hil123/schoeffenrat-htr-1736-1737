"""Fusionne les sorties des lots : results/lines/*.jsonl -> results/hits.csv + results/summary.md"""
import csv, glob, json, collections
rows = []
for p in sorted(glob.glob("results/lines/*.jsonl")):
    rows += [json.loads(l) for l in open(p, encoding="utf-8")]
rows.sort(key=lambda r: (r["vol"], r["vue"], r["side"], r["i"]))
with open("results/hits.csv", "w", newline="", encoding="utf-8") as f:
    w = csv.writer(f)
    w.writerow(["vol", "vue", "side", "i", "concordance", "hitsA", "hitsB", "A", "B", "crop"])
    for r in rows:
        la, lb = {h[0] for h in r.get("hitsA", [])}, {h[0] for h in r.get("hitsB", [])}
        if la or lb:
            w.writerow([r["vol"], r["vue"], r["side"], r["i"], "A+B" if la & lb else ("A" if la else "B"),
                        ";".join(sorted(la)), ";".join(sorted(lb)), r.get("A", ""), r.get("B", ""), r.get("crop", "")])
vues = collections.defaultdict(set); seg = collections.Counter()
for r in rows: vues[r["vol"]].add(r["vue"]); seg[r["seg"]] += 1
with open("results/summary.md", "w", encoding="utf-8") as f:
    f.write(f"# Résumé HTR\n\nLignes : {len(rows)}\n\n")
    for v in sorted(vues): f.write(f"- vol. {v} : {len(vues[v])} vues lues (min {min(vues[v])}, max {max(vues[v])})\n")
    f.write(f"\nSegmentation : {dict(seg)}\n")
print(open("results/summary.md").read())
