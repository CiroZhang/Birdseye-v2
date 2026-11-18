import sys, json, time, requests
from gdown.download_folder import _parse_embedded_folder_view
sess = requests.Session()
out = []
def walk(fid, path, depth):
    for attempt in range(3):
        try:
            name, children = _parse_embedded_folder_view(sess=sess, folder_id=fid, verify=True); break
        except Exception as e:
            time.sleep(2); children = None
    if children is None:
        out.append({"path": path, "id": fid, "error": True}); return
    for cid, cname, ctype in children:
        p = f"{path}/{cname}"
        if ctype == "application/vnd.google-apps.folder":
            if depth < int(sys.argv[2]): walk(cid, p, depth + 1)
            else: out.append({"path": p, "id": cid, "folder": True})
        else:
            out.append({"path": p, "id": cid, "type": ctype})
walk(sys.argv[1], "", 0)
json.dump(out, open(sys.argv[3], "w"), indent=1)
print(len(out)); [print(o) for o in out[:30]]
