"""Kaggle (GPU) : HTR des Schöffenratsprotokolle 1736-1737 (ISG FFM H.02.25 Nr. 42-43).
Segmentation kraken (GPU, repli projection) ; lecture TrOCR A, B, H en fp16 ; mots-clés ; sorties par blocs.
Ordre : index, fenêtre déc. 1736 – avr. 1737, Innsbruck, puis le reste."""
import os, sys, json, time, csv, io, zipfile, subprocess, traceback
from concurrent.futures import ThreadPoolExecutor
import numpy as np, cv2, requests
from PIL import Image
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from keywords import hits

OUT = "/kaggle/working/results"
URL = "https://digitalisate.frankfurt.de/isg_ffm/h.02.25/{v}/isg_ffm_h.02.25_nr_{v}_{n:04d}.jpg"
LAST = {42: 308, 43: 323}
MODELS = {"A": "dh-unibe/trocr-kurrent-XVI-XVII", "B": "dh-unibe/trocr-kurrent", "H": "fgho/trocr-hanseXVII-kurrent"}
FALLBACK_PROC = "microsoft/trocr-base-handwritten"
EVAL = {(43, 31), (43, 38)}            # toutes les lignes sauvegardées (vérité terrain)
TOP_CUT, BOT_CUT = 0.14, 0.955
BLOCK = 40

def order():
    pri = [(42, n) for n in range(257, 309)] + [(43, n) for n in range(279, 324)] \
        + [(42, n) for n in range(240, 257)] + [(43, n) for n in range(1, 81)] + [(42, n) for n in range(97, 107)]
    seen, out = set(), []
    for x in pri + [(v, n) for v in (42, 43) for n in range(1, LAST[v] + 1)]:
        if x not in seen: seen.add(x); out.append(x)
    return out

def fetch(vn):
    v, n = vn
    for t in range(6):
        try:
            r = requests.get(URL.format(v=v, n=n), timeout=60)
            if r.status_code == 404: return vn, None
            r.raise_for_status()
            return vn, cv2.imdecode(np.frombuffer(r.content, np.uint8), cv2.IMREAD_GRAYSCALE)
        except Exception as e:
            time.sleep(3 * (t + 1))
    return vn, None

def gutter_x(g):
    H, W = g.shape
    band = g[int(.2 * H):int(.9 * H), int(.38 * W):int(.66 * W)].astype(np.float32)
    prof = cv2.blur(band.mean(0)[None, :], (31, 1))[0]
    return int(.38 * W) + int(np.argmin(prof))

_blla = None
def seg_kraken(img):
    global _blla
    from kraken import blla
    try: res = blla.segment(Image.fromarray(img), device="cuda:0")
    except TypeError: res = blla.segment(Image.fromarray(img))
    lines = res.lines if hasattr(res, "lines") else res["lines"]
    out = []
    for l in lines:
        poly = l.boundary if hasattr(l, "boundary") else l.get("boundary")
        if not poly: continue
        p = np.array(poly); x0, y0 = p.min(0); x1, y1 = p.max(0)
        out.append([int(x0), int(y0), int(x1), int(y1)])
    return out

def seg_projection(g, gx):
    H, W = g.shape; out = []
    _, bw = cv2.threshold(g, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    for xa, xb in ((int(.03 * W), gx - 30), (gx + 30, int(.97 * W))):
        sub = bw[int(TOP_CUT * H):int(BOT_CUT * H), xa:xb]
        prof = cv2.blur((sub > 0).sum(1).astype(np.float32)[None, :], (1, 9))[0]
        thr = max(3.0, .08 * prof.max()); on, y = False, 0
        for i, val in enumerate(prof):
            if val > thr and not on: on, y = True, i
            elif val <= thr and on:
                on = False
                if i - y > 12:
                    cols = np.where(sub[y:i].sum(0) > 0)[0]
                    if len(cols): out.append([xa + int(cols[0]), y + int(TOP_CUT * H), xa + int(cols[-1]), i + int(TOP_CUT * H)])
    return out

def load_models():
    import torch
    from transformers import TrOCRProcessor, VisionEncoderDecoderModel
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    ms = {}
    for k, mid in MODELS.items():
        try: proc = TrOCRProcessor.from_pretrained(mid)
        except Exception as e:
            print(f"processor {mid} absent ({e}) -> {FALLBACK_PROC}", flush=True); proc = TrOCRProcessor.from_pretrained(FALLBACK_PROC)
        m = VisionEncoderDecoderModel.from_pretrained(mid).eval().to(dev)
        if dev == "cuda": m = m.half()
        ms[k] = (proc, m)
    print("modèles chargés sur", dev, flush=True)
    return ms, dev

def read(ms, dev, crops, bs=64):
    import torch
    res = {}
    for k, (proc, m) in ms.items():
        out = []
        with torch.inference_mode():
            for i in range(0, len(crops), bs):
                pv = proc(images=crops[i:i + bs], return_tensors="pt").pixel_values.to(dev)
                if dev == "cuda": pv = pv.half()
                ids = m.generate(pv, max_new_tokens=96, num_beams=1)
                out += proc.batch_decode(ids, skip_special_tokens=True)
        res[k] = out
    return res

def main():
    os.makedirs(f"{OUT}/crops", exist_ok=True)
    todo = order(); print(f"{len(todo)} vues à traiter", flush=True)
    ms, dev = load_models()
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    fl = open(f"{OUT}/lines.jsonl", "a", encoding="utf-8")
    t0 = time.time(); done = 0
    with ThreadPoolExecutor(12) as pool:
        for b in range(0, len(todo), BLOCK):
            block = todo[b:b + BLOCK]
            imgs = dict(pool.map(fetch, block))
            rows, crops = [], []
            for vn in block:
                g = imgs.get(vn)
                if g is None: print(f"{vn}: image absente", flush=True); continue
                H, W = g.shape; gx = gutter_x(g); method = "kraken"
                try: boxes = seg_kraken(clahe.apply(g))
                except Exception:
                    traceback.print_exc(); method = "projection"; boxes = seg_projection(g, gx)
                boxes = [x for x in boxes if TOP_CUT * H < (x[1] + x[3]) / 2 < BOT_CUT * H and x[2] - x[0] > 25 and x[3] - x[1] > 10]
                boxes.sort(key=lambda x: ((x[0] + x[2]) / 2 >= gx, x[1], x[0]))
                for i, (x0, y0, x1, y1) in enumerate(boxes):
                    c = g[max(0, y0 - 6):min(H, y1 + 6), max(0, x0 - 6):min(W, x1 + 6)]
                    crops.append(Image.fromarray(c).convert("RGB"))
                    rows.append({"vol": vn[0], "vue": vn[1], "side": "R" if (x0 + x1) / 2 >= gx else "L", "i": i, "bbox": [x0, y0, x1, y1], "seg": method})
            texts = read(ms, dev, crops) if crops else {}
            for j, (r, c) in enumerate(zip(rows, crops)):
                for k in MODELS: r[k] = texts[k][j]
                r["hits"] = {k: hits(r[k]) for k in MODELS}
                if any(r["hits"].values()) or (r["vol"], r["vue"]) in EVAL:
                    name = f"{r['vol']}_{r['vue']:04d}_{r['side']}_{r['i']:03d}.jpg"
                    c.resize((c.width * 2, c.height * 2), Image.LANCZOS).save(f"{OUT}/crops/{name}", quality=88)
                    r["crop"] = f"crops/{name}"
                fl.write(json.dumps(r, ensure_ascii=False) + "\n")
            fl.flush(); done += len(block)
            print(f"bloc {b // BLOCK + 1}: {done}/{len(todo)} vues, {len(rows)} lignes, {time.time() - t0:.0f}s", flush=True)
    fl.close(); finish()

def finish():
    rows = [json.loads(l) for l in open(f"{OUT}/lines.jsonl", encoding="utf-8")]
    with open(f"{OUT}/hits.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f); w.writerow(["vol", "vue", "side", "i", "labels", "concordance", *MODELS, "crop"])
        for r in rows:
            labs = {k: {h[0] for h in r["hits"][k]} for k in MODELS}
            allx = set().union(*labs.values())
            if allx:
                conc = "+".join(k for k in MODELS if labs[k])
                w.writerow([r["vol"], r["vue"], r["side"], r["i"], ";".join(sorted(allx)), conc, *[r[k] for k in MODELS], r.get("crop", "")])
    with zipfile.ZipFile("/kaggle/working/results.zip", "w", zipfile.ZIP_DEFLATED) as z:
        for d, _, fs in os.walk(OUT):
            for fn in fs: p = os.path.join(d, fn); z.write(p, os.path.relpath(p, "/kaggle/working"))
    print(f"FINI : {len(rows)} lignes ; /kaggle/working/results.zip ({os.path.getsize('/kaggle/working/results.zip') / 1e6:.1f} Mo)", flush=True)
    push()

def push():
    try:
        from kaggle_secrets import UserSecretsClient
        tok = UserSecretsClient().get_secret("GITHUB_TOKEN")
    except Exception:
        print("pas de secret GITHUB_TOKEN : télécharger results.zip à la main", flush=True); return
    repo = f"https://x-access-token:{tok}@github.com/hil123/schoeffenrat-htr-1736-1737.git"
    cmds = f"""set -e; cd /kaggle/working; rm -rf repo; git clone -q --depth 1 {repo} repo; cd repo;
git checkout -q -b kaggle-results; mkdir -p kaggle; cp -r ../results/* kaggle/;
git -c user.name=htr-bot -c user.email=htr-bot@users.noreply.github.com add kaggle;
git -c user.name=htr-bot -c user.email=htr-bot@users.noreply.github.com commit -q -m 'kaggle results';
git push -q -f origin kaggle-results"""
    r = subprocess.run(["bash", "-c", cmds], capture_output=True, text=True)
    print("push GitHub :", "OK (branche kaggle-results)" if r.returncode == 0 else r.stderr.replace(tok, "***")[-500:], flush=True)

if __name__ == "__main__":
    main()
