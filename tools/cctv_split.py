"""Train/test split of the CCTV rallies by match, and the test rallies' 3D ground truth.

A session's matches are the runs of rallies between breaks of more than MATCH_GAP (the 10-15 min between matches);
every 5th match (index 4, 9, ...) is held out. A test rally's ground truth is its two-view triangulated shuttle
(outputs/cctv/tri, tools/cctv_tri.py), written next to its observations as <uid>_gt.json in the DroneData format
(shuttle [[frame, x, y, z], ...]; no player or hit labels), so train_rallynet's evaluation reads it unchanged.

Output: outputs/cctv/split.json {train: [uid], test: [uid], matches: {session: [[first uid, last uid, n], ...]}}
usage: python tools/cctv_split.py"""
import glob, json, os

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
MATCH_GAP = 300.0  # s between the end of one rally and the start of the next
TEST_EVERY = 5


def main():
    split, matches = {"train": [], "test": []}, {}
    for sess in sorted(glob.glob(os.path.join(ROOT, "data", "cctv", "*", "rallies", "index.json"))):
        name = os.path.basename(os.path.dirname(os.path.dirname(sess)))
        R = []
        for m in glob.glob(os.path.join(os.path.dirname(sess), "*", "meta.json")):
            j = json.load(open(m))
            R.append((j["tv_start_s"], j["tv_start_s"] + j["len_s"], os.path.basename(os.path.dirname(m))))
        R.sort()
        blocks, cur = [], [R[0]]
        for prev, r in zip(R, R[1:]):
            if r[0] - prev[1] > MATCH_GAP:
                blocks.append(cur)
                cur = []
            cur.append(r)
        blocks.append(cur)
        matches[name] = [[b[0][2], b[-1][2], len(b)] for b in blocks]
        for i, b in enumerate(blocks):
            split["test" if i % TEST_EVERY == TEST_EVERY - 1 else "train"] += [r[2] for r in b]
    split["matches"] = matches
    json.dump(split, open(os.path.join(ROOT, "outputs", "cctv", "split.json"), "w"), indent=1)
    n = 0
    for uid in split["test"]:
        tp = os.path.join(ROOT, "outputs", "cctv", "tri", uid + ".json")
        obs = os.path.join(ROOT, "outputs", "rally_obs", "cctv", uid + ".npz")
        if not (os.path.exists(tp) and os.path.exists(obs)):
            continue
        sh = json.load(open(tp))["shuttle"]
        if len(sh) < 10:
            continue
        json.dump({"shuttle": sh, "players": {}, "hits": []}, open(obs[:-4] + "_gt.json", "w"))
        n += 1
    print({k: len(v) for k, v in split.items()}, "test rallies with ground truth:", n,
          {s: len(m) for s, m in matches.items()}, "matches")


if __name__ == "__main__":
    main()
