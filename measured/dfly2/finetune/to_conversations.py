# Convert generated V4.1 replies to AngelSpec conversations and check that AngelSpec's dsv41
# template rebuilds the exact token ids V4.1 produced (prompt + reply).
# usage: to_conversations.py <replies.jsonl> <train.jsonl> [check_n]
import json, sys
sys.path[:0] = ["/data/nvme3/dfly2/AngelSpec"]
from angelspec.utils.processing import load_tokenizer
from angelspec.data.template import TEMPLATE_REGISTRY
from angelspec.data.parse import create_parser
src, dst = sys.argv[1], sys.argv[2]
check_n = int(sys.argv[3]) if len(sys.argv) > 3 else 200
M = "/data/nvme3/lab/teachers/deepseek-v4.1-flash-1017"
tok = load_tokenizer(M, trust_remote_code=True)
parser = create_parser(tok, TEMPLATE_REGISTRY.get("dsv41"))
n = kept = trunc = same = checked = cuts = 0
mism_cut = 0
first_diff = None
with open(dst, "w") as out:
    for line in open(src):
        r = json.loads(line); n += 1
        if not r.get("token_ids"):
            trunc += 1; continue
        # split V4.1's own decoded reply at </think> so every space and newline is kept
        reply = tok.decode(r["token_ids"], skip_special_tokens=False)
        cut = r["finish_reason"] != "stop"
        reply = reply.removesuffix("<｜end▁of▁sentence｜>")
        reasoning, sep, content = reply.partition("</think>")
        if not sep:
            reasoning, content = reply, ""
        cuts += cut
        conv = [{"role": "user", "content": r["prompt"]},
                {"role": "assistant", "reasoning_content": reasoning, "content": content}]
        if checked < check_n:
            ids, mask = parser.parse(conv, max_length=1 << 20)
            ids = [int(x) for x in ids.flatten().tolist()]
            want = r["prompt_token_ids"] + r["token_ids"]
            checked += 1
            if cut:
                want = want + [ids[-2], ids[-1]] if len(ids) == len(want) + 2 else want
            if ids == want:
                same += 1
            elif cut:
                mism_cut += 1
            elif first_diff is None:
                k = next((i for i, (a, b) in enumerate(zip(ids, want)) if a != b), min(len(ids), len(want)))
                first_diff = (r["id"], k, len(ids), len(want), tok.decode(ids[max(0, k-5):k+5]), tok.decode(want[max(0, k-5):k+5]))
        out.write(json.dumps({"id": r["id"], "conversations": conv}, ensure_ascii=False) + "\n"); kept += 1
print(f"replies {n}, kept {kept} ({cuts} cut at the token cap, closed by the template), dropped empty {trunc}")
print(f"exact token match {same} of {checked} (mismatches in cut replies {mism_cut})")
if first_diff: print("first mismatch:", first_diff)
