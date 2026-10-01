"""HTR d'un lot de vues ISG FFM H.02.25 : segmentation kraken (repli : profil de projection),
lecture TrOCR A (dh-unibe/trocr-kurrent-XVI-XVII) et B (dh-unibe/trocr-kurrent), détection de mots-clés."""
import argparse, io, json, os, sys, time, traceback
import numpy as np, cv2, requests
from PIL import Image
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from keywords import hits

URL = "https://digitalisate.frankfurt.de/isg_ffm/h.02.25/{v}/isg_ffm_h.02.25_nr_{v}_{n:04d}.jpg"
MODELS = {"A": "dh-unibe/trocr-kurrent-XVI-XVII", "B": "dh-unibe/trocr-kurrent"}
FALLBACK_PROC = "microsoft/trocr-base-handwritten"
TOP_CUT, BOT_CUT = 0.14, 0.955   # bandeau des compteurs / pied de vue

def fetch(v, n):
    for t in range(6):
        try:
            r = requests.get(URL.format(v=v, n=n), timeout=60)
            if r.status_code == 404: return None
            r.raise_for_status()
            return cv2.imdecode(np.frombuffer(r.content, np.uint8), cv2.IMREAD_GRAYSCALE)
        except Exception as e:
            print(f"fetch {v}/{n} try {t}: {e}", flush=True); time.sleep(5 * (t + 1))
    raise RuntimeError(f"fetch failed {v}/{n}")

def gutter_x(g):
    H, W = g.shape
    band = g[int(.2 * H):int(.9 * H), int(.38 * W):int(.66 * W)].astype(np.float32)
    prof = cv2.blur(band.mean(0)[None, :], (31, 1))[0]
    return int(.38 * W) + int(np.argmin(prof))

def seg_kraken(img_clahe):
    from kraken import blla
    res = blla.segment(Image.fromarray(img_clahe))
    lines = res.lines if hasattr(res, "lines") else res["lines"]
    out = []
    for l in lines:
        poly = l.boundary if hasattr(l, "boundary") else l.get("boundary")
        if not poly: continue
        p = np.array(poly)
        x0, y0 = p.min(0); x1, y1 = p.max(0)
        out.append([int(x0), int(y0), int(x1), int(y1)])
    return out

def seg_projection(g, gx):
    H, W = g.shape
    out = []
    _, bw = cv2.threshold(g, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    for xa, xb in ((int(.03 * W), gx - 30), (gx + 30, int(.97 * W))):
        sub = bw[int(TOP_CUT * H):int(BOT_CUT * H), xa:xb]
        prof = cv2.blur((sub > 0).sum(1).astype(np.float32)[None, :], (1, 9))[0]
        thr = max(3.0, .08 * prof.max())
        on, y = False, 0
        for i, val in enumerate(prof):
            if val > thr and not on: on, y = True, i
            elif val <= thr and on:
                on = False
                if i - y > 12:
                    cols = np.where(sub[y:i].sum(0) > 0)[0]
                    if len(cols): out.append([xa + int(cols[0]), y + int(TOP_CUT * H), xa + int(cols[-1]), i + int(TOP_CUT * H)])
    return out

def load(mid):
    from transformers import TrOCRProcessor, VisionEncoderDecoderModel
    try: proc = TrOCRProcessor.from_pretrained(mid)
    except Exception as e:
        print(f"processor {mid} absent ({e}); fallback {FALLBACK_PROC}", flush=True)
        proc = TrOCRProcessor.from_pretrained(FALLBACK_PROC)
    model = VisionEncoderDecoderModel.from_pretrained(mid).eval()
    return proc, model

def read_all(crops, mid, bs=16):
    import torch
    proc, model = load(mid)
    out = []
    with torch.inference_mode():
        for i in range(0, len(crops), bs):
            pv = proc(images=crops[i:i + bs], return_tensors="pt").pixel_values
            ids = model.generate(pv, max_new_tokens=96, num_beams=1)
            out += proc.batch_decode(ids, skip_special_tokens=True)
            if i % (bs * 10) == 0: print(f"  {mid}: {i + len(ids)}/{len(crops)}", flush=True)
    del model
    return out

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--vol", type=int, required=True)
    ap.add_argument("--vues", required=True, help="ex. 1-24 ou 31,38")
    ap.add_argument("--out", default="out")
    a = ap.parse_args()
    vues = []
    for part in a.vues.split(","):
        if "-" in part: s, e = map(int, part.split("-")); vues += list(range(s, e + 1))
        else: vues.append(int(part))
    os.makedirs(f"{a.out}/crops", exist_ok=True)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    rows, crops = [], []
    for n in vues:
        g = fetch(a.vol, n)
        if g is None: print(f"vue {n}: 404", flush=True); continue
        H, W = g.shape; gx = gutter_x(g)
        method = "kraken"
        try: boxes = seg_kraken(clahe.apply(g))
        except Exception:
            traceback.print_exc(); method = "projection"; boxes = seg_projection(g, gx)
        boxes = [b for b in boxes if TOP_CUT * H < (b[1] + b[3]) / 2 < BOT_CUT * H and b[2] - b[0] > 25 and b[3] - b[1] > 10]
        boxes.sort(key=lambda b: ((b[0] + b[2]) / 2 >= gx, b[1], b[0]))
        print(f"vue {n}: {len(boxes)} lignes ({method}), gouttière x={gx}", flush=True)
        for i, (x0, y0, x1, y1) in enumerate(boxes):
            pad = 6
            c = g[max(0, y0 - pad):min(H, y1 + pad), max(0, x0 - pad):min(W, x1 + pad)]
            crops.append(Image.fromarray(c).convert("RGB"))
            rows.append({"vol": a.vol, "vue": n, "side": "R" if (x0 + x1) / 2 >= gx else "L", "i": i,
                         "bbox": [x0, y0, x1, y1], "seg": method})
    for key, mid in MODELS.items():
        texts = read_all(crops, mid)
        for r, t in zip(rows, texts): r[key] = t
    for r, c in zip(rows, crops):
        r["hitsA"] = hits(r.get("A", "")); r["hitsB"] = hits(r.get("B", ""))
        if r["hitsA"] or r["hitsB"]:
            name = f"{r['vol']}_{r['vue']:04d}_{r['side']}_{r['i']:03d}.jpg"
            c.resize((c.width * 2, c.height * 2), Image.LANCZOS).save(f"{a.out}/crops/{name}", quality=85)
            r["crop"] = f"crops/{name}"
    tag = f"{a.vol}_{a.vues.replace(',', '_')}"
    with open(f"{a.out}/lines_{tag}.jsonl", "w", encoding="utf-8") as f:
        for r in rows: f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"OK {len(rows)} lignes, {sum(1 for r in rows if 'crop' in r)} avec mot-clé", flush=True)

if __name__ == "__main__":
    main()
