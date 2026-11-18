"""Shared bits for the rally QA model: splits, prompt format, answer parsing and scoring."""
import json, os, random, re

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
QA2 = os.path.join(ROOT, "outputs", "qa2")
MODEL = "Qwen/Qwen3-4B-Instruct-2507"

# split by match (no rally of a test match is ever seen in training)
TRAIN = ["BD12", "BD13", "BD14", "BD15", "BD16", "BD17", "BD18", "BD19", "BD26", "BD27", "BD28"]
VAL = ["BD20"]
TEST = ["BD21", "BD22", "BD23", "BD29", "BD30"]

SYSTEM = ("You answer questions about one badminton doubles rally. You get an automatic 3D reconstruction made "
          "from a single camera behind the near baseline; it can miss strokes, split one stroke into several, or "
          "be off in depth. Answer each numbered question on its own line as '<number>: <answer>', where the "
          "answer is a number, or one word from the choices the question gives. No explanations.")


def load(matches):
    rows = []
    for m in matches:
        p = os.path.join(QA2, m + ".jsonl")
        if os.path.exists(p):
            rows += [json.loads(l) for l in open(p)]
    return rows


def fmt_answer(q, a):
    if q["kind"] in ("float", "rel"):
        return f"{a:.1f}" if q["kind"] == "float" else f"{a:.0f}"
    return str(a)


def thin_scene(text):
    """Keep every other sample of the shuttle track (10 -> 5 Hz) and the players (5 -> 2.5 Hz) and say so;
    halves the prompt so training fits in GPU memory."""
    lines = text.split("\n")
    for i, l in enumerate(lines):
        if l.startswith("  ") and " | " in l:
            lines[i] = "  " + " | ".join(l.strip().split(" | ")[::2])
        lines[i] = lines[i].replace("10 Hz (", "5 Hz (").replace("5 Hz (t: near", "2.5 Hz (t: near")
    return "\n".join(lines)


def make_prompt(row, qs):
    body = thin_scene(row["scene_text"]) +"\n\nQuestions:\n" + "\n".join(f"{i + 1}. {q['q']}" for i, q in enumerate(qs))
    return [{"role": "system", "content": SYSTEM}, {"role": "user", "content": body}]


def make_target(qs):
    return "\n".join(f"{i + 1}: {fmt_answer(q, q['answer'])}" for i, q in enumerate(qs))


def sample_questions(row, rng, max_q=24):
    qs = list(row["questions"])
    rng.shuffle(qs)
    return qs[:max_q]


def parse(text, n):
    out = [None] * n
    for line in text.splitlines():
        m = re.match(r"\s*(\d+)\s*[:.)]\s*(.+?)\s*$", line)
        if m and 1 <= int(m.group(1)) <= n:
            out[int(m.group(1)) - 1] = m.group(2)
    return out


def correct(q, pred):
    if pred is None:
        return False
    try:
        if q["kind"] == "choice":
            return str(pred).strip().lower().strip(".'\"") == q["answer"]
        v = float(re.findall(r"-?\d+(?:\.\d+)?", str(pred))[0])
        if q["kind"] == "int":
            return int(round(v)) == q["answer"]
        if q["kind"] == "float":
            return abs(v - q["answer"]) <= q["tol"]
        return abs(v - q["answer"]) <= q["tol"] * abs(q["answer"])  # rel
    except (TypeError, ValueError, IndexError):
        return False
