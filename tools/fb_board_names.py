"""Player names on the broadcast scoreboard of each FBBench clip (one OCR'd frame: the first one fb_scoreboard read a
board on), to tell which clips come from the same match (tools/fb_score_chain.py).

Output: outputs/fb/board_names.json  {uid: [upper row name, lower row name]}
usage: python tools/fb_board_names.py   (env WORKERS=8)"""
import glob, json, os, re, sys
from multiprocessing import Pool

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(ROOT, "tools"))
import fb_scoreboard as SB  # noqa: E402

OUT = os.path.join(ROOT, "outputs", "fb", "board_names.json")


def names(img):
    """Upper / lower name of the two player rows (fb_scoreboard.read_board's row pairing), upper-cased letters."""
    if SB._eng is None:
        SB.read_board(img[:8, :8])  # builds the engine
    h, w = img.shape[:2]
    c = SB.CROP
    r = SB._eng(img[int(c[1] * h):int(c[3] * h), int(c[0] * w):int(c[2] * w)])
    if r.txts is None:
        return None
    toks = []
    for box, t, sc in zip(r.boxes, r.txts, r.scores):
        if sc >= 0.6 and re.search(r"[A-Za-z]{2,}", t):
            ys = [p[1] for p in box]
            toks.append(((min(ys) + max(ys)) / 2, max(ys) - min(ys), re.sub(r"[^A-Z]", "", t.upper())))
    toks.sort()
    for a, b in zip(toks, toks[1:]):
        if 0.6 * a[1] < b[0] - a[0] < 3.0 * a[1] and a[2] and b[2]:
            return [a[2], b[2]]
    return None


def run(uid):
    import cv2
    cv2.setNumThreads(1)
    p = os.path.join(ROOT, "outputs", "fb", "scoreboard", uid + ".json")
    rd = [r for r in json.load(open(p))["reads"] if r[1] is not None]
    if not rd:
        return uid, None
    cap = cv2.VideoCapture(os.path.join(SB.CLIPS, uid + ".mp4"))
    for f in [rd[0][0], rd[len(rd) // 2][0]]:
        cap.set(cv2.CAP_PROP_POS_FRAMES, f)
        ok, im = cap.read()
        if ok:
            n = names(im)
            if n:
                return uid, n
    return uid, None


if __name__ == "__main__":
    uids = sorted(os.path.basename(p)[:-5] for p in glob.glob(os.path.join(ROOT, "outputs", "fb", "scoreboard", "*.json")))
    out = {}
    with Pool(int(os.environ.get("WORKERS", "8"))) as pool:
        for u, n in pool.imap_unordered(run, uids):
            if n:
                out[u] = n
    json.dump(out, open(OUT, "w"), indent=0)
    print(len(out), "/", len(uids), "clips with names ->", OUT)
