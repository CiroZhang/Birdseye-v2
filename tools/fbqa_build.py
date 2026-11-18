"""A training set of multiple-choice questions in the formats of the FineBadminton benchmark (FBBench), built from the
FineBadminton-20K training annotations. FBBench has no training split: a QA model (Qwen) is fine-tuned on these and
tested on FBBench. Only matches outside data/fb20k/benchmark_source_videos.txt are used (asserted), so nothing here
comes from the matches FBBench was built from.

Each FBBench task was read off its 200-250 questions (data/fbbench/finebadmintonbenchmark/) and is mimicked here:
question templates paraphrased from the real question texts, 4 options with FBBench's "A. " prefixes and option
wording, distractors drawn the way FBBench's appear to be, answer letter uniform. Per task:
  action_classification  type of the asked side's last hit; options "Name: definition", 3 other types uniformly
  action_prediction      type of the opponent's response to the asked side's last hit (clip ends before it)
  action_localization    "which hit number is the <type> (and top or bottom)": options "k, side" (70%) or "k", the
                         distractors other hit numbers of the clip with their true hitter; the type occurs once
  hitting_localization   the asked side's court area at its last hit (own facing frame); FBBench's fixed option sets:
                         a corner -> the 4 corners, mid front/mid back/mid court -> their own 3 companions
  hitting_prediction     the opponent's area at the next hit (where the last hit sends the shuttle), same option sets
  moving_recognization   the asked side's areas over its hits in the clip ("a->b->c", 3-7 hits); distractors are
                         uniformly random 9-zone sequences of the same length (as FBBench's look)
  hitcount               hits by top / bottom / both in the rally; distractors answer +-2/4/6 (non-negative)
  select_detail_count    times one stroke type occurs in the rally (>= 1); distractors nearby counts; balanced
                         over counts 1 / 2 / 3 / 4 / 5+ (natural counts are 57% ones: a value prior would get 0.8)
  hitting_comment        the annotated comment on the asked side's last hit; distractors comments on other hits of
                         a different type in other rallies, top/bottom swapped so all name the asked side
  pointer_recog          the rally's score_reason or lose_reason (question asks which); distractors the same field
                         of other rallies
  round_comment          the rally's total_eva; distractors other rallies' total_eva
  roundcount             skipped: needs multi-rally clips (see SKIPPED)
Clips: FBBench cuts a window of a rally for the "last hit" tasks (FBBench clip lengths per task in DUR). Every
question names its FineBadminton-20K rally clip (uid) and the window to show in meta.clip = [start, end] frames of
that clip video (data/fb20k/clips/<uid>.mp4, the label frame numbers; seconds = frame / meta.fps); the window ends
just before the opponent's next contact, or halfway to it for the prediction tasks. Answer classes are balanced per
task (round robin over stroke types / zones), at most MAX_PER_RALLY questions per rally and task, CAP per task.
Distractor texts come from the same split. Split by match: VAL_FRAC of the matches to val (seed SEED).

Inputs: data/fb20k/clips/<match>_<k>.json (tools/fb20k_prep.py), data/fb20k/benchmark_source_videos.txt,
        data/fbbench/finebadmintonbenchmark/*.json (only for the reference baselines in the summary)
Output: outputs/fbqa/train.jsonl, outputs/fbqa/val.jsonl   one question per line:
            {uid, task, question, options ["A. ..", ..], answer "A".., meta {clip, fps, ...}}
        outputs/fbqa/summary.json   per task counts, answer letters, trivial baselines (most common letter, longest
            option, answer prior: how often each option text was right in train) on train+val (letter, longest), on
            val and on FBBench (fitted on train), skipped tasks
usage: python tools/fbqa_build.py"""
import collections, glob, json, os, random, re

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
FB20 = os.path.join(ROOT, "data", "fb20k")
BENCH = os.path.join(ROOT, "data", "fbbench", "finebadmintonbenchmark")
OUT = os.path.join(ROOT, "outputs", "fbqa")
SEED, VAL_FRAC, CAP, MAX_PER_RALLY = 0, 0.1, 600, 2
LETTERS = "ABCD"
LEAD_S = 0.4  # s of picture before a window's first contact

# FBBench task name (file name without benchmark_ and suffixes) -> its file
FILES = {"action_classification": "benchmark_action_classification.json",
         "action_localization": "benchmark_action_localization.json",
         "action_prediction": "benchmark_action_prediction.json", "hitcount": "benchmark_hitcount.json",
         "hitting_comment": "benchmark_hitting_comment2_updated.json",
         "hitting_localization": "benchmark_hitting_localization_new.json",
         "hitting_prediction": "benchmark_hitting_prediction.json",
         "moving_recognization": "benchmark_moving_recognization.json",
         "pointer_recog": "benchmark_pointer_recog_updated.json",
         "round_comment": "benchmark_round_comment_updated.json", "roundcount": "benchmark_roundcount.json",
         "select_detail_count": "benchmark_select_detail_count_questions.json"}
SKIPPED = {"roundcount": "needs multi-rally clips (FBBench: 4 s - 17 min of broadcast, 6-22 rallies, asks how many "
                         "rallies). FineBadminton-20K annotates only some rallies of each match (e.g. 3 of match 0015) "
                         "and has no score, so a span of consecutive annotated rallies cannot be shown to hold no "
                         "unannotated rally in between; it would also need new multi-rally clips cut from the "
                         "match videos, while every question here names one rally clip."}

# FBBench's stroke names and the definitions its options and questions carry (FineBadminton-20K says "kill")
DEFS = {"Clear": "A clear is a shot hit deep into the opponent's backcourt with a high trajectory.",
        "Smash": "A smash is a powerful overhead shot hit steeply downwards into the opponent's court.",
        "Drop shot": "A drop shot is typically hit from the midcourt or backcourt, aiming to land softly and close to "
                     "the net in the opponent's forecourt.",
        "Push shot": "A push shot is played from the forecourt or midcourt, directing the shuttle with a pushing "
                     "motion, often flat or slightly upwards.",
        "Net shot": "A net shot is played close to the net, usually dropping tightly over the net into the "
                    "opponent's forecourt.",
        "Net kill": "A net kill is an aggressive shot hit downwards forcefully from close to the net when the "
                    "shuttle is above net height.",
        "Net lift": "A net lift is an underhand shot played from the forecourt, lifting the shuttle high and deep "
                    "into the opponent's backcourt, often used defensively.",
        "Block": "A block is a defensive shot played at the net, using the opponent's pace to return the shuttle, "
                 "often dropping it short.",
        "Cross-court net shot": "A cross-court net shot involves hitting the shuttle diagonally across the net from "
                                "one side of the forecourt to the other, using wrist action.",
        "Drive": "A drive is a fast, flat shot hit horizontally across the net, usually from the midcourt."}
TYPE = {"clear": "Clear", "kill": "Smash", "drop shot": "Drop shot", "push shot": "Push shot", "net shot": "Net shot",
        "net kill": "Net kill", "net lift": "Net lift", "block": "Block", "cross-court net shot": "Cross-court net shot",
        "drive": "Drive"}  # serve is never asked or offered in FBBench
ZONES = ["left front court", "mid front court", "right front court", "left mid court", "mid court", "right mid court",
         "left back court", "mid back court", "right back court"]
CORNERS = ["left front court", "right front court", "left back court", "right back court"]
# hitting localization / prediction: FBBench's options are fixed by the answer (never left / right mid court)
ZONE_SET = {**{c: CORNERS for c in CORNERS},
            "mid front court": ["mid front court", "mid back court", "left back court", "right back court"],
            "mid back court": ["mid back court", "mid front court", "left front court", "right front court"],
            "mid court": ["mid court", "mid back court", "left back court", "right back court"]}
# FBBench clip lengths (s, 10th-90th percentile) of the tasks that show part of a rally
DUR = {"action_classification": (2.9, 4.8), "action_prediction": (2.8, 7.9), "hitting_comment": (3.5, 5.7),
       "hitting_localization": (3.2, 6.3), "hitting_prediction": (1.4, 3.9)}
LOC_N = {4: .30, 5: .25, 6: .20, 7: .12, 8: .08, 9: .05}  # hits in an action localization clip
MOVE_N = {3: .40, 4: .25, 5: .15, 6: .10, 7: .10}  # the asked side's hits in a moving recognition clip
LOC_WITH_SIDE = 0.7  # FBBench: 141 of 200 localization questions also ask top or bottom

# question templates paraphrased from FBBench ({S} the asked side, {O} the opponent)
Q = {
    "action_classification": [
        "Determine what type of shot the player (at the {S}) used for their last hit. Choose from the four options",
        "Which type of shot best describes the player's (at the {S}) last hit? Choose from the four options provided",
        "Select the type of shot that the player at the {S} most likely used for their last hit. Make your choice "
        "based on the four explanations provided",
        "Choose from four options: What type of shot did the player (at the {S}) most likely use for their last hit?",
        "Determine which type of shot the player at the {S} most likely used for their last hit. Refer to the four "
        "explanations below for your choice",
        "Regarding the type of shot, which one did the player at the {S} most likely use for their last hit? Choose "
        "from the four options",
        "In the video clip, which type of shot is most likely used by the player at the {S} for their last hit? "
        "Choose the most probable one from the four explanations provided",
        "What type of shot was most likely used by the player (at the {S}) during their last hit? Select based on "
        "the descriptions given",
        "Determine which type of shot the player (at the {S}) most likely used during their last hit. Choose from "
        "the four options",
        "Analyze the action of the player's (at the {S}) last hit and determine which type of shot they most likely "
        "used. Choose from the four options"],
    "action_prediction": [
        "After the final stroke by the player at the {S}, what is the most likely response from the opponent at the "
        "{O} in the video? Select the best explanation from the four options",
        "Based on the four explanations, determine the most likely return from the opponent at the {O} after the "
        "final hit by the player at the {S}",
        "What is the most likely response from the opponent at the {O} after the final hit by the player at the "
        "{S}? Pick the best option from the four",
        "In the video clip, after the player at the {S} of the screen hits last, what is the most likely response "
        "from the opponent at the {O}? Choose the best option from the four explanations",
        "After the player at the {S} makes their final hit, predict the most probable return from the opponent at "
        "the {O}. Choose from the four explanations",
        "From the four explanations, select the most probable way the opponent at the {O} would return after the "
        "player at the {S} hits last",
        "Choose the most likely return from the opponent at the {O} from the four options after the final hit by "
        "the player at the {S}",
        "What is the most likely outcome of the return from the opponent at the {O} after the player at the {S} "
        "makes their final hit? Choose from the four options",
        "Analyze the most probable return from the opponent at the {O} after the player at the {S} hits last. "
        "Choose the best answer from the four explanations",
        "Infer the most probable return from the opponent at the {O} after the player at the {S} hits last. Select "
        "the best explanation from the four options"],
    "hitting_localization": [
        "Determine the area of their court (with their facing direction as front) where the player (at the {S}) "
        "executed their last hit. Choose the closest option based on the explanations",
        "Infer the area of their court (with their facing direction as front) where the player at the {S} hit from "
        "during their last hit. Choose the closest option from the four provided",
        "What area of their court (with their facing direction as front) was the player at the {S} most likely "
        "hitting from during their last hit? Please select the closest option based on the descriptions",
        "Which area of their court (facing direction as front) did the player (at the {S}) hit from during their "
        "last hit? (Choose the most appropriate option)",
        "Which area of their court (facing direction as front) did the player at the {S} hit from during their last "
        "hit? Please select the closest option",
        "Choose the closest option: Which area of their court (facing direction as front) did the player at the {S} "
        "hit from during their last hit?",
        "Which area of their own court (with their facing direction as front) did the player at the {S} execute "
        "their last hit from? Please select the closest answer",
        "Analyze the player (at the {S})'s last hit— which area of their court (with their facing direction as "
        "front) did they hit from? Choose the closest option",
        "In the video clip, which area of their own court (with their facing direction as front) did the player at "
        "the {S} hit from during their last hit? Please choose the most appropriate option based on the explanations",
        "Select the closest option: Which area of their court (facing direction as front) did the player at the {S} "
        "hit from during their last hit?"],
    "hitting_prediction": [
        "Infer the most likely landing area on the opponent's court (with the opponent's facing direction as front) "
        "after the player (at the {S}) hits last. Choose from the options",
        "Analyze where the ball will most likely land on the opponent's court (with the opponent's facing direction "
        "as front) after the player (at the {S}) hits last. Choose from the options",
        "Choose from four options: After the player (at the {S}) hits last, which area on the opponent's court "
        "(with the opponent's facing direction as front) will the ball land most likely",
        "Determine the most probable landing area on the opponent's court (with the opponent's facing direction as "
        "front) after the player (at the {S}) hits last. Choose from the options",
        "In the video clip, after the player at the {S} hits last, which area of the opponent's court (with the "
        "opponent's facing direction as front) will the ball most likely land? Choose from the four options",
        "What area on the opponent's court (with the opponent's facing direction as front) is the ball most likely "
        "to land in after the player (at the {S}) hits last? Select the answer",
        "Which area of the opponent's court (with the opponent's facing direction as front) is most likely to be "
        "the landing spot after the player (at the {S}) hits last?",
        "Select the area on the opponent's court (with the opponent's facing direction as front) where the ball is "
        "most likely to land after the player (at the {S}) hits last",
        "Which area on the opponent's court (with the opponent's facing direction as front) is most likely to be "
        "hit by the ball after the player (at the {S}) hits last?",
        "Predict which area of the opponent's court (with the opponent's facing direction as front) the ball will "
        "land in after the player (at the {S}) hits last. Choose from the options"],
    "moving_recognization": [
        "In this badminton clip, which option best matches the movement direction of the player (at the {S}) "
        "relative to their facing direction?",
        "Which option best aligns with the player's (at the {S}) movement relative to their facing direction "
        "during the hit?",
        "Which option best describes the actual movement path of the {S} player relative to their facing "
        "direction during the hit?",
        "Based on the video, infer which option describes the {S} player's movement relative to their facing "
        "direction during the hit.",
        "What option describes the movement direction of the player (at the {S}) relative to their facing "
        "direction during the hit?",
        "Which option corresponds to the player's (at the {S}) movement relative to their facing direction during "
        "the hit?",
        "Select the correct option that describes the {S} player's movement relative to their facing direction "
        "during the hit.",
        "Analyze the player's (at the {S}) movement path relative to their facing direction— which option best "
        "describes it?",
        "Which option most closely matches the actual path of the {S} player relative to their facing direction "
        "during the hit?",
        "Determine which option matches the player's (at the {S}) movement relative to their facing direction "
        "during the hit."],
    "hitting_comment": [
        "For the last shot by the player at the {S} of the screen, which of the following evaluations is the most "
        "appropriate?",
        "For the last shot made by the player at the {S} of the screen, which of the following evaluations is most "
        "appropriate?",
        "For the last shot by the player at the {S} of the screen, which of the following evaluations is the most "
        "accurate?",
        "Regarding the last shot by the player at the {S} of the screen, which of the following evaluations is most "
        "appropriate?",
        "For the last shot by the player at the {S} of the screen, which evaluation is the most appropriate?",
        "Which of the following evaluations is most appropriate for the last shot made by the player at the {S} of "
        "the screen?",
        "Please select the most appropriate evaluation of the last shot by the player at the {S} of the screen based "
        "on the following description:",
        "For the last shot by the player at the {S} of the screen, which of the following evaluations best matches "
        "the actual situation?",
        "In a badminton singles match, regarding the last shot made by the player at the {S} of the screen, which "
        "of the following evaluations is the most accurate?",
        "Which of the following most accurately evaluates the last shot by the player at the {S} of the screen?",
        "For the last shot of the {S} player in this rally, which of the following evaluations best matches the "
        "actual performance?",
        "In the last shot made by the player at the {S} of the match footage, which of the following evaluations is "
        "the most accurate?",
        "For the last shot by the player at the {S} of the screen in the clip, which of the following evaluations "
        "most accurately describes the characteristics of the shot?",
        "Based on the last shot of the player at the {S} in the video, select the most appropriate evaluation.",
        "Please evaluate the last shot of the player at the {S} of the screen based on the match clip and choose the "
        "most appropriate description.",
        "Which of the following evaluations best describes the performance of the player at the {S} of the screen "
        "during the last shot?"],
    "pointer_recog": {
        "score": ["Please analyze the main reason for scoring in this rally.",
                  "Please analyze the main reasons for scoring in this rally",
                  "For this rally, please explain the reason for scoring?",
                  "Please analyze the key factors of scoring in this rally?",
                  "Please explain the reason for scoring in this rally?",
                  "Please analyze the main reason for scoring in this round.",
                  "Please explain the main reason for scoring in this rally",
                  "What was the key factor that led to scoring in this rally?",
                  "Please analyze the main reason why the scoring side scored in this rally."],
        "lose": ["Please analyze the main reason for losing the point in this rally?",
                 "Please analyze the main reason for losing this rally?",
                 "Please analyze the main reason for losing the point in this rally.",
                 "Please identify the main reason for losing the point in this rally?",
                 "Please analyze the main reasons for losing the point in this rally.",
                 "What was the main reason for losing the point in this rally?",
                 "For this rally, please explain the reason for losing the point?",
                 "Please analyze the main reasons for losing this round?",
                 "Analyze the key reasons leading to the loss of this rally."]},
    "round_comment": [
        "Which evaluation is most appropriate for this rally?", "Which evaluation is most appropriate for this round?",
        "Which of the following evaluations best describes the overall performance of this rally?",
        "Which evaluation is the most appropriate for this rally?",
        "Which of the following best evaluates the overall performance of this rally?",
        "Which of the following options most accurately evaluates the overall performance of this rally?",
        "For this rally, which evaluation is the most appropriate?",
        "In this rally, which of the following evaluations most accurately describes the overall performance?",
        "Which option best comprehensively reflects the overall performance of this rally?",
        "Which of the following is the most suitable overall evaluation of this rally?"],
    "hitcount": {  # {W}: the top player / the bottom player / both players
        "any": ["How many strokes did {W} hit altogether in this badminton rally?",
                "Calculate the total number of strokes hit by {W} in this rally.",
                "What is the total number of strokes {W} hit in this badminton rally?",
                "Tally the total number of strokes {W} hit in this badminton rally.",
                "In this badminton rally, how many strokes did {W} hit in total?",
                "Find the total number of strokes {W} hit during this badminton rally.",
                "Please calculate the total number of hits {W} made during this rally.",
                "Determine the combined number of hits {W} made in this badminton rally."],
        "both": ["How many strokes did the two players hit combined in this badminton rally?",
                 "Count the total number of strokes each player hit in this badminton rally."]},
}
LOC_Q = "In the video segment, which hit number corresponds to {T}: {D}"
SELECT_Q = "In the provided video segment, how many times was the '{T}' performed? {D}"


def bench_sources():
    return {l.strip()[:4] for l in open(os.path.join(FB20, "benchmark_source_videos.txt")) if l.strip()}


def ascii_text(t, min_len=40):
    return isinstance(t, str) and len(t.strip()) >= min_len and all(ord(c) < 128 for c in t)


def load_rallies():
    rallies = []
    for p in sorted(glob.glob(os.path.join(FB20, "clips", "*.json"))):
        c = json.load(open(p, encoding="utf8"))
        hits = c["hits"]
        if not hits:
            continue
        uid = os.path.basename(p)[:-5]
        for h in hits:
            h["stroke"] = TYPE.get(h["type"])  # None: serve or unlabelled
            h["zone"] = h["area"] if h["area"] in ZONES else None
        rallies.append({"uid": uid, "match": uid.split("_")[0], "fps": c["fps"], "hits": hits,
                        "eval": c.get("evaluation", {}), "n_frames": hits[-1]["end"] + 10 + 1})  # prep's PAD = 10
    return rallies


# ---- clip windows (frames of the rally clip video) ----

def win_start(r, j):
    """Window start: LEAD_S before hit j's contact, after the previous contact."""
    h = r["hits"]
    s = h[j]["frame"] - int(round(LEAD_S * r["fps"]))
    return max(0 if j == 0 else h[j - 1]["frame"] + 1, s, 0)


def win_end(r, k, predict=False):
    """Window end after hit k: just before the next contact (halfway to it when the response is to be predicted)."""
    h = r["hits"]
    if k + 1 < len(h):
        nxt = h[k + 1]["frame"]
        return h[k]["frame"] + max(1, (nxt - h[k]["frame"]) // 2) if predict else nxt - 1
    return min(r["n_frames"] - 1, h[k]["end"] + 10)


def last_hit_window(r, k, task, rng, predict=False):
    """[start, end] ending at hit k with a FBBench-like length: the hit start closest to a target length."""
    end = win_end(r, k, predict)
    target = rng.uniform(*DUR[task]) * r["fps"]
    j = min(range(k + 1), key=lambda i: abs(end - win_start(r, i) - target))
    return [win_start(r, j), end], j


def full_window(r):
    return [0, r["n_frames"] - 1]


OTHER = {"top": "bottom", "bottom": "top"}


def swap_sides(text):
    """top <-> bottom (and upper <-> lower), case kept."""
    sw = {"top": "bottom", "bottom": "top", "upper": "lower", "lower": "upper"}

    def f(m):
        w = m.group(0)
        o = sw[w.lower()]
        return o.upper() if w.isupper() else o.capitalize() if w[0].isupper() else o
    return re.sub(r"\b(top|bottom|upper|lower)\b", f, text, flags=re.I)


def stroke_opt(name):
    return f"{name}: {DEFS[name]}"


# ---- candidates per task: dict(uid, cls (answer class, for balancing), answer, distractors | None, question, meta);
#      text distractors are filled in later from the split's pools ----

def cand_action_classification(r, rng):
    out = []
    for k, h in enumerate(r["hits"]):
        if not h["stroke"]:
            continue
        clip, j = last_hit_window(r, k, "action_classification", rng)
        others = rng.sample([n for n in DEFS if n != h["stroke"]], 3)
        out.append(dict(cls=h["stroke"], answer=stroke_opt(h["stroke"]), distractors=[stroke_opt(n) for n in others],
                        question=rng.choice(Q["action_classification"]).format(S=h["hitter"]),
                        meta=dict(clip=clip, hit=k, side=h["hitter"], hits_in_clip=k - j + 1)))
    return out


def cand_action_prediction(r, rng):
    out = []
    hs = r["hits"]
    for k in range(len(hs) - 1):
        resp = hs[k + 1]
        if not resp["stroke"]:
            continue
        clip, j = last_hit_window(r, k, "action_prediction", rng, predict=True)
        others = rng.sample([n for n in DEFS if n != resp["stroke"]], 3)
        s = hs[k]["hitter"]
        out.append(dict(cls=resp["stroke"], answer=stroke_opt(resp["stroke"]),
                        distractors=[stroke_opt(n) for n in others],
                        question=rng.choice(Q["action_prediction"]).format(S=s, O=OTHER[s]),
                        meta=dict(clip=clip, hit=k, side=s, last_type=hs[k]["type"], hits_in_clip=k - j + 1)))
    return out


def cand_action_localization(r, rng):
    out = []
    hs = r["hits"]
    n = rng.choices(list(LOC_N), weights=list(LOC_N.values()))[0]
    if len(hs) < 4:
        return out
    n = min(n, len(hs))
    j = rng.randrange(len(hs) - n + 1)
    win = hs[j:j + n]
    counts = collections.Counter(h["stroke"] for h in win)
    with_side = rng.random() < LOC_WITH_SIDE
    for i, h in enumerate(win):
        if not h["stroke"] or counts[h["stroke"]] != 1:  # asked type occurs once in the clip
            continue
        lab = (lambda q: f"{q + 1}, {win[q]['hitter']}") if with_side else (lambda q: f"{q + 1}")
        others = rng.sample([q for q in range(n) if q != i], 3)
        q = LOC_Q.format(T=h["stroke"], D=DEFS[h["stroke"]])
        q += ", and is the hitter at the top or bottom of the screen?" if with_side else "?"
        out.append(dict(cls=h["stroke"], answer=lab(i), distractors=[lab(o) for o in others], question=q,
                        meta=dict(clip=[win_start(r, j), win_end(r, j + n - 1)], hit=j + i, hits_in_clip=n,
                                  with_side=with_side)))
    return out


def cand_hitting_localization(r, rng):
    out = []
    for k, h in enumerate(r["hits"]):
        if h["type"] == "serve" or h["zone"] not in ZONE_SET:
            continue
        clip, j = last_hit_window(r, k, "hitting_localization", rng)
        out.append(dict(cls=h["zone"], answer=h["zone"], distractors=[z for z in ZONE_SET[h["zone"]] if z != h["zone"]],
                        question=rng.choice(Q["hitting_localization"]).format(S=h["hitter"]),
                        meta=dict(clip=clip, hit=k, side=h["hitter"], hits_in_clip=k - j + 1)))
    return out


def cand_hitting_prediction(r, rng):
    out = []
    hs = r["hits"]
    for k in range(len(hs) - 1):
        z = hs[k + 1]["zone"]
        if z not in ZONE_SET:
            continue
        clip, j = last_hit_window(r, k, "hitting_prediction", rng, predict=True)
        out.append(dict(cls=z, answer=z, distractors=[o for o in ZONE_SET[z] if o != z],
                        question=rng.choice(Q["hitting_prediction"]).format(S=hs[k]["hitter"]),
                        meta=dict(clip=clip, hit=k, side=hs[k]["hitter"], hits_in_clip=k - j + 1)))
    return out


def cand_moving_recognization(r, rng):
    out = []
    hs = r["hits"]
    for side in ("top", "bottom"):
        idx = [k for k, h in enumerate(hs) if h["hitter"] == side]
        L = rng.choices(list(MOVE_N), weights=list(MOVE_N.values()))[0]
        if len(idx) < 3:
            continue
        L = min(L, len(idx))
        a = rng.randrange(len(idx) - L + 1)
        ks = idx[a:a + L]
        seq = [hs[k]["zone"] for k in ks]
        if None in seq:
            continue
        ans = "->".join(seq)
        dis = set()
        while len(dis) < 3:
            d = "->".join(rng.choice(ZONES) for _ in seq)
            if d != ans:
                dis.add(d)
        out.append(dict(cls=L, answer=ans, distractors=sorted(dis),
                        question=rng.choice(Q["moving_recognization"]).format(S=side),
                        meta=dict(clip=[win_start(r, ks[0]), win_end(r, ks[-1])], hits=ks, side=side, n=L)))
    return out


def cand_hitcount(r, rng):
    hs = r["hits"]
    who = rng.choices(["both", "top", "bottom"], weights=[2, 1, 1])[0]  # FBBench: ~half ask both players
    n = len(hs) if who == "both" else sum(h["hitter"] == who for h in hs)
    if who == "both" and rng.random() < 0.3:
        q = rng.choice(Q["hitcount"]["both"])
    else:
        q = rng.choice(Q["hitcount"]["any"]).format(W="both players" if who == "both" else f"the {who} player")
    offs = rng.sample([d for d in (-6, -4, -2, 2, 4, 6) if n + d >= 0], 3)
    return [dict(cls=who, answer=str(n), distractors=[str(n + d) for d in offs], question=q,
                 meta=dict(clip=full_window(r), side=who, count=n))]


def cand_select_detail_count(r, rng):
    out = []
    counts = collections.Counter(h["stroke"] for h in r["hits"] if h["stroke"])
    for t, n in counts.items():
        pool = [v for v in range(max(0, n - 4), n + 6) if v != n]
        w = [1.0 if abs(v - n) <= 3 else 0.5 for v in pool]
        dis = set()
        while len(dis) < 3:
            dis.add(rng.choices(pool, weights=w)[0])
        out.append(dict(cls=min(n, 5), answer=str(n), distractors=[str(v) for v in dis],
                        question=SELECT_Q.format(T=t, D=DEFS[t]),
                        meta=dict(clip=full_window(r), hit_type=t, count=n)))
    return out


def cand_hitting_comment(r, rng):
    out = []
    for k, h in enumerate(r["hits"]):
        if not h["stroke"] or not ascii_text(h["comment"], 60):
            continue
        clip, j = last_hit_window(r, k, "hitting_comment", rng)
        out.append(dict(cls=h["stroke"], answer=h["comment"].strip(), distractors=None,
                        question=rng.choice(Q["hitting_comment"]).format(S=h["hitter"]),
                        meta=dict(clip=clip, hit=k, side=h["hitter"], hit_type=h["stroke"], hits_in_clip=k - j + 1,
                                  point=h["point"])))
    return out


def winner(r):
    last = r["hits"][-1]
    if "score the point" in last["point"]:
        return last["hitter"]
    if "lose the point" in last["point"]:
        return OTHER[last["hitter"]]
    return None


def cand_pointer_recog(r, rng):
    kind = rng.choice(["score", "lose"])
    t = r["eval"].get(kind + "_reason", "")
    if not ascii_text(t):
        return []
    return [dict(cls=kind, answer=t.strip(), distractors=None, question=rng.choice(Q["pointer_recog"][kind]),
                 meta=dict(clip=full_window(r), kind=kind, winner=winner(r)))]


def cand_round_comment(r, rng):
    t = r["eval"].get("total_eva", "")
    if not ascii_text(t):
        return []
    return [dict(cls="rally", answer=t.strip(), distractors=None, question=rng.choice(Q["round_comment"]),
                 meta=dict(clip=full_window(r), winner=winner(r)))]


GEN = {"action_classification": cand_action_classification, "action_localization": cand_action_localization,
       "action_prediction": cand_action_prediction, "hitcount": cand_hitcount,
       "hitting_comment": cand_hitting_comment, "hitting_localization": cand_hitting_localization,
       "hitting_prediction": cand_hitting_prediction, "moving_recognization": cand_moving_recognization,
       "pointer_recog": cand_pointer_recog, "round_comment": cand_round_comment,
       "select_detail_count": cand_select_detail_count}
BALANCE = {"action_classification", "action_prediction", "action_localization", "hitting_localization",
           "hitting_prediction", "select_detail_count", "hitting_comment"}  # round robin over answer classes (cls)


def text_pools(task, rallies, cands, val):
    """(split, kind) -> [(uid, text, stroke type, side)]: every hit comment / rally text that could be an answer."""
    pools = collections.defaultdict(list)
    if task == "hitting_comment":
        for c in cands:
            pools[(c["split"], None)].append((c["uid"], c["answer"], c["cls"], c["meta"]["side"]))
    elif task in ("pointer_recog", "round_comment"):
        for r in rallies:
            for kind, key in ((("score", "score_reason"), ("lose", "lose_reason")) if task == "pointer_recog"
                              else ((None, "total_eva"),)):
                t = r["eval"].get(key, "")
                if ascii_text(t):
                    pools[("val" if r["match"] in val else "train", kind)].append((r["uid"], t.strip(), None, None))
    return pools


def text_distractors(task, c, pools, rng):
    """Three texts of other rallies in the same split. Hitting comments: of another stroke type, top / bottom swapped
    so they name the asked side, as FBBench's do."""
    pool = pools[(c["split"], c["meta"].get("kind"))]
    out, seen = [], {c["answer"]}
    while len(out) < 3:
        uid, t, typ, side = rng.choice(pool)
        if uid == c["uid"] or (task == "hitting_comment" and typ == c["cls"]):
            continue
        if task == "hitting_comment" and side != c["meta"]["side"]:
            t = swap_sides(t)
        if t not in seen:
            seen.add(t)
            out.append(t)
    return out


def sample(cands, rng, balance):
    """At most MAX_PER_RALLY per rally, then up to CAP: round robin over answer classes (balance) or at random."""
    rng.shuffle(cands)
    per = collections.Counter()
    kept = []
    for c in cands:
        if per[c["uid"]] < MAX_PER_RALLY:
            per[c["uid"]] += 1
            kept.append(c)
    if not balance:
        return kept[:CAP]
    by = collections.defaultdict(list)
    for c in kept:
        by[c["cls"]].append(c)
    out, i = [], 0
    while len(out) < CAP and any(i < len(v) for v in by.values()):
        out += [v[i] for v in by.values() if i < len(v)][:CAP - len(out)]
        i += 1
    return out


def finish(task, cs, rng):
    """Options in random order with the answer letter balanced over the task."""
    letters = [LETTERS[i % 4] for i in range(len(cs))]
    rng.shuffle(letters)
    rows = []
    for c, L in zip(cs, letters):
        opts = list(c["distractors"])
        rng.shuffle(opts)
        i = LETTERS.index(L)
        opts.insert(i, c["answer"])
        meta = dict(c["meta"], fps=c["fps"], clip_s=[round(f / c["fps"], 2) for f in c["meta"]["clip"]])
        rows.append({"uid": c["uid"], "task": task, "question": c["question"],
                     "options": [f"{LETTERS[q]}. {o}" for q, o in enumerate(opts)], "answer": L, "meta": meta})
    return rows


# ---- trivial baselines ----

def opt_text(o):
    return o[3:]


def baselines(train, test, rng):
    """Accuracy of: the most common answer letter in train; the longest option; the option whose text was most
    often right when offered in train (Laplace, ties at random)."""
    if not test:
        return {}
    letter = collections.Counter(x["answer"] for x in train).most_common(1)[0][0]
    shown, right = collections.Counter(), collections.Counter()
    for x in train:
        for o in x["options"]:
            shown[opt_text(o)] += 1
        right[opt_text(x["options"][LETTERS.index(x["answer"])])] += 1
    acc = collections.Counter()
    for x in test:
        a = LETTERS.index(x["answer"])
        acc["letter"] += a == LETTERS.index(letter)
        lens = [len(o) for o in x["options"]]
        acc["longest"] += a == lens.index(max(lens))
        pr = [(right[opt_text(o)] + 1) / (shown[opt_text(o)] + 4) + rng.random() * 1e-6 for o in x["options"]]
        acc["answer_prior"] += a == pr.index(max(pr))
    return {k: round(acc[k] / len(test), 3) for k in ("letter", "longest", "answer_prior")}


def load_bench(task):
    """FBBench items in our row format (English), for reference baselines."""
    d = json.load(open(os.path.join(BENCH, FILES[task]), encoding="utf8"))
    return [{"options": it["options_en"], "answer": it["answer"]} for it in d]


def main():
    rng = random.Random(SEED)
    rallies = load_rallies()
    matches = sorted({r["match"] for r in rallies})
    overlap = set(matches) & bench_sources()
    assert not overlap, f"benchmark source matches in the training clips: {sorted(overlap)}"
    val = set(random.Random(SEED).sample(matches, max(1, round(VAL_FRAC * len(matches)))))
    print(f"{len(rallies)} rallies, {len(matches)} matches; val matches {sorted(val)}")

    out = {"train": [], "val": []}
    summary = {"seed": SEED, "cap_per_task": CAP, "val_matches": sorted(val), "n_rallies": len(rallies),
               "excluded_benchmark_sources": sorted(bench_sources()), "tasks": {}, "skipped": SKIPPED}
    for task, gen in GEN.items():
        trng = random.Random(f"{SEED}:{task}")
        cands = []
        for r in rallies:
            for c in gen(r, trng):
                c.update(uid=r["uid"], fps=r["fps"], split="val" if r["match"] in val else "train")
                cands.append(c)
        pools = text_pools(task, rallies, cands, val)
        chosen = sample(cands, trng, task in BALANCE)
        for c in chosen:
            if c["distractors"] is None:
                c["distractors"] = text_distractors(task, c, pools, trng)
        rows = {s: finish(task, [c for c in chosen if c["split"] == s], trng) for s in ("train", "val")}
        for s in rows:
            out[s] += rows[s]
        allr = rows["train"] + rows["val"]
        brng = random.Random(SEED)
        summary["tasks"][task] = {
            "fbbench_file": FILES[task], "train": len(rows["train"]), "val": len(rows["val"]),
            "candidates": len(cands),
            "answer_letters": dict(sorted(collections.Counter(x["answer"] for x in allr).items())),
            "answer_classes": dict(collections.Counter(str(c["cls"]) for c in chosen).most_common()),
            "baselines_all": {k: v for k, v in baselines(allr, allr, brng).items() if k != "answer_prior"},
            "baselines_val": baselines(rows["train"], rows["val"], brng),
            "baselines_fbbench": baselines(rows["train"], load_bench(task), brng)}
    os.makedirs(OUT, exist_ok=True)
    for s in out:
        with open(os.path.join(OUT, s + ".jsonl"), "w", encoding="utf8") as fh:
            for x in out[s]:
                fh.write(json.dumps(x, ensure_ascii=False) + "\n")
    summary["total"] = {s: len(out[s]) for s in out}
    json.dump(summary, open(os.path.join(OUT, "summary.json"), "w", encoding="utf8"), indent=1, ensure_ascii=False)

    print(f"{'task':22s} {'train':>5s} {'val':>4s}  letters A/B/C/D  all: letter longest | val: letter longest "
          f"prior | FBBench: letter longest prior")
    for t, v in summary["tasks"].items():
        L = "/".join(str(v["answer_letters"].get(c, 0)) for c in LETTERS)
        ba, bv, bb = v["baselines_all"], v["baselines_val"], v["baselines_fbbench"]
        print(f"{t:22s} {v['train']:5d} {v['val']:4d}  {L:15s}  {ba['letter']:.2f} {ba['longest']:.2f}     | "
              f"{bv.get('letter', 0):.2f} {bv.get('longest', 0):.2f} {bv.get('answer_prior', 0):.2f}     | "
              f"{bb['letter']:.2f} {bb['longest']:.2f} {bb['answer_prior']:.2f}")
    print("total", summary["total"], "| skipped:", ", ".join(SKIPPED))


if __name__ == "__main__":
    main()
