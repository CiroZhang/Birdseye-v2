"""Who won an FBBench rally, from the broadcast scoreboard of the NEXT rally's clip.

FBBench clips end before the board updates, but the benchmark holds many rallies of the same match: the clip whose
board shows exactly one more point (nearby uid) is the next rally, and the row that went up won. Rows are names,
not court sides; the side of each row comes from the serve rule -- the server stands in the right service court
when their score is even -- read off our reconstruction (server = first hitter, court = sign of server + receiver
lateral positions at the rally start, tools/fbbench_answer.own_frame), voted over the clips of the same game.

Uses other benchmark clips' VIDEO (never labels): transductive, so reported separately.
winner(uid) -> ("top" | "bottom" | None, info).   Inputs: outputs/fb/scoreboard/*.json (tools/fb_scoreboard.py),
outputs/fbbench_serve_dump_all.json (server side, lat, recv_lat per clip)."""
import collections, glob, json, os

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
FB = os.path.join(ROOT, "outputs", "fb")
OTHER = {"top": "bottom", "bottom": "top"}
WINDOW = 60  # uids: clips of one match lie close together
MIN_COURT = 0.3  # m: |server + receiver lateral| below this is no court reading
_B, _S, _N = None, None, None


def same_match(u, v):
    """Both boards name the same two players (tools/fb_board_names.py); unknown names never match."""
    global _N
    if _N is None:
        p = os.path.join(FB, "board_names.json")
        _N = json.load(open(p)) if os.path.exists(p) else {}
    return u in _N and _N.get(u) == _N.get(v)


def num(u):
    return int(u.split("_")[1])


def boards():
    """uid -> (upper row, lower row) points at the clip start (the most common of the first readings)."""
    global _B
    if _B is None:
        _B = {}
        for p in glob.glob(os.path.join(FB, "scoreboard", "*.json")):
            rd = [r for r in json.load(open(p))["reads"] if r[1] is not None]
            if rd:
                _B[os.path.basename(p)[:-5]] = collections.Counter((a, b) for _, a, b in rd[:5]).most_common(1)[0][0]
    return _B


def serves():
    global _S
    if _S is None:
        p = os.path.join(ROOT, "outputs", "fbbench_serve_dump_all.json")
        _S = json.load(open(p)) if os.path.exists(p) else {}
    return _S


def same_game(u, v):
    """v's board can come before / after u's in one game (nearby uid, scores move the right way)."""
    B = boards()
    if v not in B or abs(num(v) - num(u)) > WINDOW or not same_match(u, v):
        return False
    (a, b), (c, d) = B[u], B[v]
    if num(v) > num(u):
        return c >= a and d >= b and c + d - a - b <= 2 * WINDOW
    return c <= a and d <= b


def next_row(u):
    """'upper' / 'lower': the row that scored, from the nearest later clip one point on; None if not found or
    both rows are possible."""
    B = boards()
    if u not in B:
        return None, None
    a, b = B[u]
    cand = sorted((abs(num(v) - num(u)), v) for v in B if v != u and abs(num(v) - num(u)) <= WINDOW
                  and B[v] in ((a + 1, b), (a, b + 1)) and same_match(u, v))
    if not cand or len({B[v] for _, v in cand}) > 1:  # none, or both rows +1 somewhere: ambiguous
        return None, None
    v = cand[0][1]
    return ("upper" if B[v] == (a + 1, b) else "lower"), v


def upper_side(u):
    """Side (top/bottom) of the board's upper row in u's game, voted by the serve rule over the game's clips."""
    B, S = boards(), serves()
    votes = collections.Counter()
    for w in [u] + [v for v in B if v != u and same_game(u, v)]:
        s = S.get(w)
        if s is None or s.get("recv_lat") is None:
            continue
        a, b = B[w]
        court = s["lat"] + s["recv_lat"]
        if a % 2 == b % 2 or abs(court) < MIN_COURT:
            continue
        server_row = "upper" if (a % 2 == 0) == (court > 0) else "lower"
        votes[s["server"] if server_row == "upper" else OTHER[s["server"]]] += 1
    if not votes:
        return None, votes
    top = votes.most_common(1)[0]
    return (top[0] if top[1] > sum(votes.values()) - top[1] else None), votes


def winner(u):
    row, v = next_row(u)
    if row is None:
        return None, {"why": "no next rally on the board"}
    up, votes = upper_side(u)
    if up is None:
        return None, {"why": "row sides unknown", "next": v, "votes": dict(votes)}
    side = up if row == "upper" else OTHER[up]
    return side, {"next": v, "row": row, "votes": dict(votes)}
