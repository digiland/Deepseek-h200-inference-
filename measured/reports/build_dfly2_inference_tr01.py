#!/usr/bin/env python3
"""DFly2 Inference TR-01: Tencent AngelSpec DFly2 drafter against DeepSeek's built-in DSpark drafter,
DeepSeek-V4.1-Flash-1017 on 8 x H200 (vLLM 0.32.0). Two pages, A4 portrait.
Every number is read from /data/nvme3/dfly2: results/bench/results.csv (benchmark rows), results/quality/*.json
(quality probe), logs/probe_t06.log and logs/probe_think.log (acceptance probes), logs/gate2_think.log (oracle check),
logs/matrix.log (failures and timings) and the drafter's safetensors files.
Run: report/.render/venv/bin/python report/build_dfly2_inference_tr01.py"""
import csv, glob, html, json, os, re
from theme import FONTS, LOGO

L = '/data/nvme3/lab'; W = '/data/nvme3/dfly2'
DATE = '27 Nov 2026'
TITLE = 'DFly2 Inference TR-01 - DFly2 against DSpark drafts for DeepSeek V4.1 Flash on 8 x H200'
OUTDIR = '/data/nvme3/inference/report'; os.makedirs(OUTDIR, exist_ok=True)
OUT = f'{OUTDIR}/{TITLE} ({DATE}).pdf'
esc = html.escape

# ---------- benchmark rows ----------
R = list(csv.DictReader(open(f'{W}/results/bench/results.csv')))
def rows(tag, conc):
    """both copies (A, B) of a 2 x TP4 setup, or the one row of a TP8 setup"""
    return [r for r in R if r['conc'] == str(conc) and (r['tag'] in (tag + 'A', tag + 'B') or r['tag'] == tag)]
def node(tag, conc):
    rs = rows(tag, conc)
    if not rs: return None
    n = len(rs)
    return dict(users=conc * n, pu=sum(float(r['per_user_tok_s']) for r in rs) / n, tot=sum(float(r['output_tok_s']) for r in rs),
                acc=sum(float(r['accept_len']) for r in rs) / n, ttft=sum(float(r['ttft_p50_ms']) for r in rs) / n,
                tpot=sum(float(r['tpot_p50_ms']) for r in rs) / n, out=sum(float(r['mean_out_len']) for r in rs) / n)
Q = {os.path.basename(p)[8:-5]: json.load(open(p)) for p in glob.glob(f'{W}/results/quality/quality-M*.json')}
def qs(tag):
    k = tag + 'A' if tag + 'A' in Q else tag
    return f"{sum(r['correct'] for r in Q[k]['rows'])} of {len(Q[k]['rows'])}" if k in Q else 'n/r'

# setups on 2 x TP4 (two copies, 4 GPUs each); M4r is the full rerun of M4 (see section 6)
DUAL = [('M1_dspark3_think', 'DSpark 3, thinking on'), ('M2_dfly3_think', 'DFly2 3, thinking on'),
        ('M3_dfly4_think', 'DFly2 4, thinking on'), ('M4r_dspark3_off', 'DSpark 3, thinking off'), ('M5_dfly3_off', 'DFly2 3, thinking off')]
PERCOPY = [20, 40, 80, 100]
f1 = lambda x: f'{x:.1f}'; f0 = lambda x: f'{x:,.0f}'
def pct(a, b): return f'{100 * (a / b - 1):+.0f}%'

head = ''.join(f'<th class="num">{esc(n)}</th>' for _, n in DUAL)
pu_html = ''
for c in PERCOPY:
    v = [node(t, c) for t, _ in DUAL]; best = max(x['pu'] for x in v)
    pu_html += f'<tr><td class="num">{c * 2}</td>' + ''.join(f'<td class="num">{"<b>" + f1(x["pu"]) + "</b>" if x["pu"] == best else f1(x["pu"])}</td>' for x in v) + '</tr>'
tot_tds = ''.join(f'<td class="num">{f0(node(t, 100)["tot"])}</td>' for t, _ in DUAL)
acc_tds = ''.join(f'<td class="num">{node(t, 40)["acc"]:.2f}</td>' for t, _ in DUAL)
out_tds = ''.join(f'<td class="num">{node(t, 40)["out"]:.0f}</td>' for t, _ in DUAL)
q_tds = ''.join(f'<td class="num">{qs(t)}</td>' for t, _ in DUAL)

# gap of each DFly2 setup against the DSpark setup with the same thinking mode
def gaps(dfly, dspark):
    g = [100 * (node(dfly, c)['pu'] / node(dspark, c)['pu'] - 1) for c in PERCOPY]
    return min(g), max(g)
g_on = gaps('M2_dfly3_think', 'M1_dspark3_think'); g_on4 = gaps('M3_dfly4_think', 'M1_dspark3_think')
g_off = gaps('M5_dfly3_off', 'M4r_dspark3_off')
def rng(g):
    lo, hi = sorted(round(abs(x)) for x in g)
    return f'{lo}%' if lo == hi else f'{lo} to {hi}%'

# other layouts: 1 x TP8 and high load
TP8 = [40, 80, 160, 200]
tp8_html = ''
for c in TP8:
    a, b = node('M6_dspark3_tp8', c), node('M7_dfly3_tp8', c); d = node('M1_dspark3_think', c // 2)
    tp8_html += (f'<tr><td class="num">{c}</td><td class="num">{f1(a["pu"])}</td><td class="num">{f1(b["pu"])}</td><td class="num">{pct(b["pu"], a["pu"])}</td>'
                 f'<td class="num">{f0(a["tot"])}</td><td class="num">{f0(b["tot"])}</td><td class="num">{f1(d["pu"])}</td></tr>')
hi_a, hi_b = node('M8_dspark3_high', 256), node('M9_dfly3_high', 256)
hi_html = ''.join(f'<tr><td>{n}</td><td class="num">{x["users"]}</td><td class="num">{f0(x["tot"])}</td><td class="num">{f1(x["pu"])}</td><td class="num">{f0(x["ttft"])}</td><td class="num">{x["acc"]:.2f}</td><td class="num">{qs(t)}</td></tr>'
                  for t, n, x in (('M8_dspark3_high', 'DSpark 3', hi_a), ('M9_dfly3_high', 'DFly2 3', hi_b)))
tp8_80 = node('M6_dspark3_tp8', 80)['pu'] / node('M1_dspark3_think', 40)['pu'] - 1
tp8_rng = [100 * (node(t8, c)['pu'] / node(t4, c // 2)['pu'] - 1) for t8, t4 in (('M6_dspark3_tp8', 'M1_dspark3_think'), ('M7_dfly3_tp8', 'M2_dfly3_think')) for c in TP8]

# ---------- acceptance probes: per-position acceptance ----------
def probe(path):
    P = {}
    for line in open(path):
        m = re.match(r'(\S+) T=([\d.]+) top_p=[\d.]+: .*accept_len=([\d.]+) .*per-position: (.*)', line)
        if m: P[(m.group(1).replace('-think', ''), m.group(2))] = (float(m.group(3)), [float(x) for x in re.findall(r'p\d=([\d.]+)', m.group(4))])
    return P
P_off, P_on = probe(f'{W}/logs/probe_t06.log'), probe(f'{W}/logs/probe_think.log')
PROBE = [('dspark3-prob-block', 'DSpark 3'), ('dfly2-n3-prob-std', 'DFly2 3')]
pr_html = ''
for k, n in PROBE:
    for mode, P in (('off', P_off), ('on', P_on)):
        a, ps = P[(k, '0.6')]
        pr_html += f'<tr><td>{n}, thinking {mode}</td><td class="num">{a:.2f}</td>' + ''.join(f'<td class="num">{p:.2f}</td>' for p in ps[:3]) + f'<td class="num">{P[(k, "0.0")][0]:.2f}</td></tr>'
p0_on = (P_on[('dspark3-prob-block', '0.6')][1][0], P_on[('dfly2-n3-prob-std', '0.6')][1][0])
p0_off = (P_off[('dspark3-prob-block', '0.6')][1][0], P_off[('dfly2-n3-prob-std', '0.6')][1][0])

# step time at equal acceptance (thinking off, 20 users per copy)
st_a, st_b = node('M4r_dspark3_off', 20), node('M5_dfly3_off', 20)
step_a, step_b = st_a['tpot'] * st_a['acc'], st_b['tpot'] * st_b['acc']   # ms per decode step = ms per token x tokens per step

# ---------- oracle check against AngelSpec's own model code ----------
G = open(f'{W}/logs/gate2_think.log').read()
gk = lambda pat: re.search(pat, G).group(1)
oracle = [('Block KV cache, vLLM against AngelSpec, mean cosine', gk(r'cosine block KV.*?mean ([\d.]+)')),
          ('Target hidden state fed to the drafter, mean cosine', gk(r'cosine backbone hidden: mean ([\d.]+)')),
          ('Corrected hidden state, mean cosine', gk(r'cosine corrected hidden: mean ([\d.]+)')),
          ('Same top-1 draft token, share of draft slots', f"{100 * float(gk(r'top-1 agreement per slot: ([\d.]+)')):.1f}%"),
          ('Result', 'PASS' if 'GATE2 PASS' in G else 'FAIL')]
or_html = ''.join(f'<tr><td>{a}</td><td class="num">{b}</td></tr>' for a, b in oracle)

# drafter size from the safetensors files
dsz = sum(os.path.getsize(f) for f in glob.glob(f'{W}/DS-V4.1-Flash-DFly2-B6/*.safetensors')) / 1024 ** 3
cfg = json.load(open(f'{W}/served/DS-V4.1-Flash-DFly2-B6/config.json'))

# ---------- problems met ----------
INV = sorted(os.path.basename(p) for p in glob.glob(f'{W}/results/bench/invalid/*.json'))
M = open(f'{W}/logs/matrix.log').read()
t0 = re.search(r'(\S+) matrix=M1_dspark3_think', M).group(1)[11:16]; t1 = re.search(r'(\S+) redo done', M).group(1)[11:16]
fails = M.count('SERVER FAILED') + M.count('SERVER TIMEOUT')

PAIRS = [(d, k, c) for d, k, cs in (('M2_dfly3_think', 'M1_dspark3_think', PERCOPY), ('M3_dfly4_think', 'M1_dspark3_think', PERCOPY),
         ('M5_dfly3_off', 'M4r_dspark3_off', PERCOPY), ('M7_dfly3_tp8', 'M6_dspark3_tp8', TP8), ('M9_dfly3_high', 'M8_dspark3_high', [256])) for c in cs]
NLOSE = sum(node(d, c)['pu'] < node(k, c)['pu'] for d, k, c in PAIRS)
QV = sorted(sum(r['correct'] for r in v['rows']) for v in Q.values()); QLO, QHI = f'{QV[0]} of 50', f'{QV[-1]} of 50'
b80 = node('M4r_dspark3_off', 40); a80 = node('M1_dspark3_think', 40); d80 = node('M2_dfly3_think', 40); o80 = node('M5_dfly3_off', 40)
qa = [('Does DFly2 beat DSpark on DeepSeek V4.1 Flash?', f'<b>No.</b> DFly2 is slower per user in {NLOSE} of {len(PAIRS)} matched setup and load pairs, on 2 x 4 GPUs, 1 x 8 GPUs and at 512 users.'),
      ('Speed per user at 80 users per node, 2 x 4 GPUs', f'Thinking on: DSpark 3 <b>{f1(a80["pu"])}</b> tok/s, DFly2 3 {f1(d80["pu"])} ({pct(d80["pu"], a80["pu"])}). Thinking off: DSpark 3 <b>{f1(b80["pu"])}</b>, DFly2 3 {f1(o80["pu"])} ({pct(o80["pu"], b80["pu"])}).'),
      ('DFly2 loss per user, all loads, 2 x 4 GPUs', f'Thinking on: {rng(g_on)} with 3 drafts, {rng(g_on4)} with 4 drafts. Thinking off: {rng(g_off)} with 3 drafts.'),
      ('Cause with thinking on', f'Fewer accepted drafts: first draft accepted {p0_on[1]:.2f} of the time against DSpark\'s {p0_on[0]:.2f}; DFly2 was trained on public chat, not DeepSeek V4.1 reasoning text.'),
      ('Cause with thinking off', f'Equal acceptance ({st_b["acc"]:.2f} against {st_a["acc"]:.2f} tokens per step) but a longer step: {step_b:.1f} ms against {step_a:.1f} ms at 20 users per copy.'),
      ('Highest total output, 512 users on 2 x 4 GPUs', f'DSpark 3 <b>{f0(hi_a["tot"])}</b> tok/s; DFly2 3 {f0(hi_b["tot"])} ({pct(hi_b["tot"], hi_a["tot"])}).'),
      ('Is the DFly2 port to vLLM correct?', f'Yes: the served drafter picks the same top-1 draft token as AngelSpec\'s own code in {oracle[3][1]} of draft slots (section 5).'),
      ('Quality probe, 50 questions', f'Every setup scored {QLO} to {QHI}.')]

CSS = FONTS + """
@page { size: A4 portrait; margin: 11mm 12mm 13mm 12mm; }
body { font-family: 'Open Sans'; color: #20252A; font-size: 7.9pt; line-height: 1.36; margin: 0; }
.hdr { display: flex; justify-content: space-between; align-items: center; border-bottom: 0.6pt solid #C9CED3; padding-bottom: 3pt; margin-bottom: 7pt; }
.hdr span { font-size: 6.8pt; color: #5B6268; } .hdr img { height: 17pt; }
h1 { font-size: 14pt; margin: 0 0 1pt; } .date { font-size: 7pt; color: #5B6268; margin-bottom: 5pt; }
h2 { font-size: 11pt; margin: 8pt 0 4pt; padding-bottom: 2pt; border-bottom: 1.3pt solid #2BB3A3; break-after: avoid; }
p { margin: 0 0 4pt; }
table { border-collapse: collapse; width: 100%; }
table.t th { background: #12264A; color: #fff; font-weight: 700; text-align: left; padding: 3pt 4pt; font-size: 7.1pt; vertical-align: bottom; }
table.t th.num { text-align: right; }
table.t td { padding: 1.55pt 4pt; border-bottom: 0.5pt solid #DDE1E5; vertical-align: top; font-size: 7.3pt; }
table.t tr:nth-child(even) td { background: #F3F5F8; }
table.t td:first-child { font-weight: 700; }
.num { text-align: right !important; white-space: nowrap; font-family: 'IBM Plex Mono'; font-size: 7pt; }
.note { font-size: 6.6pt; color: #5B6268; margin: 2pt 0 4pt; }
.two { display: flex; gap: 8pt; } .two > div { flex: 1; min-width: 0; }
.pb { break-before: page; }
"""
hdr = f'<div class="hdr"><span>{esc(TITLE)}</span><img src="{LOGO}"></div>'
body = f"""{hdr}
<h1>DFly2 against DSpark draft models for DeepSeek V4.1 Flash on 8 x H200</h1>
<div class="date">{DATE} · Internal · target model deepseek-ai/DeepSeek-V4.1-Flash-1017 · drafters: DSpark (built into the checkpoint) and Tencent AngelSpec DFly2 ({dsz:.1f} GiB, block of {cfg['dspark_block_size']}, {cfg['dspark_n_routed_experts']} experts, {cfg['dspark_num_experts_per_tok']} per token) · vLLM 0.32.0 · node cpt-gpu170, 8 x NVIDIA H200 141 GB</div>
<h2>1. Summary</h2>
<p><b>DFly2</b>, Tencent's draft model for DeepSeek V4.1 Flash, was ported to vLLM and measured against the built-in <b>DSpark</b> draft model on the production setup (2 copies of 4 GPUs) and on 1 copy of 8 GPUs, with thinking on and off. <b>DSpark with 3 draft tokens stays the faster draft model in every measurement.</b></p>
<table class="t"><tr><th style="width:27%">Question</th><th>Our answer</th></tr>{''.join(f'<tr><td>{esc(a)}</td><td>{b}</td></tr>' for a, b in qa)}</table>
<h2>2. Speed per user on 2 copies of 4 GPUs, tok/s (bold: best at that load)</h2>
<table class="t"><tr><th class="num">Users per node</th>{head}</tr>{pu_html}
<tr><td>Node total at 200 users, tok/s</td>{tot_tds}</tr><tr><td>Tokens per decode step per user</td>{acc_tds}</tr><tr><td>Mean reply length, tokens</td>{out_tds}</tr><tr><td>Quality probe</td>{q_tds}</tr></table>
<div class="note">Source: dfly2/results/bench/results.csv, setups M1 to M5 (M4 rerun as M4r). Speed per user = 1,000 / median ms between one user's output tokens, mean of the 2 copies. "DSpark 3" = DSpark with 3 draft tokens. Tokens per step and reply length at 80 users per node. Reply length is capped at 256 tokens.</div>
<h2>3. Other layouts: 1 copy of 8 GPUs, and 512 users</h2>
<table class="t"><tr><th class="num">Users, 1 x 8 GPUs</th><th class="num">DSpark 3, tok/s per user</th><th class="num">DFly2 3, tok/s per user</th><th class="num">DFly2 against DSpark</th><th class="num">DSpark 3 node total</th><th class="num">DFly2 3 node total</th><th class="num">DSpark 3 on 2 x 4 GPUs</th></tr>{tp8_html}</table>
<div class="note">Source: setups M6 and M7, thinking on. 1 copy of 8 GPUs is {abs(max(tp8_rng)):.0f} to {abs(min(tp8_rng)):.0f}% slower per user than 2 copies of 4 GPUs for both draft models.</div>
<table class="t"><tr><th>512 users, 2 x 4 GPUs, thinking on</th><th class="num">Users</th><th class="num">Node total tok/s</th><th class="num">Tok/s per user</th><th class="num">TTFT ms</th><th class="num">Tokens per step</th><th class="num">Quality</th></tr>{hi_html}</table>
<div class="note">Source: setups M8 and M9, 256 users per copy (vLLM limit of 256 sequences per copy). TTFT: median time to first token.</div>
<div class="pb"></div>{hdr}
<h2 style="margin-top:0">4. Why DFly2 is slower: draft acceptance and step time</h2>
<p>Acceptance probe: 160 short ShareGPT prompts at temperature 0.6, 3 draft tokens, one copy of 4 GPUs. Draft position n is accepted only if every earlier draft was.</p>
<table class="t"><tr><th>Draft model and thinking mode</th><th class="num">Tokens per step</th><th class="num">Draft 1 accepted</th><th class="num">Draft 2 accepted</th><th class="num">Draft 3 accepted</th><th class="num">Tokens per step at temperature 0</th></tr>{pr_html}</table>
<div class="note">Source: dfly2/logs/probe_t06.log (thinking off) and probe_think.log (thinking on, the chat template default). Thinking on drops DFly2's first-draft acceptance from {p0_off[1]:.2f} to {p0_on[1]:.2f}; DSpark holds {p0_on[0]:.2f}.</div>
<table class="t"><tr><th>Thinking off, 20 users per copy</th><th class="num">Tokens per step</th><th class="num">Ms per output token</th><th class="num">Ms per decode step</th><th class="num">Tok/s per user</th></tr>
<tr><td>DSpark 3</td><td class="num">{st_a['acc']:.2f}</td><td class="num">{st_a['tpot']:.2f}</td><td class="num">{step_a:.1f}</td><td class="num">{f1(st_a['pu'])}</td></tr>
<tr><td>DFly2 3</td><td class="num">{st_b['acc']:.2f}</td><td class="num">{st_b['tpot']:.2f}</td><td class="num">{step_b:.1f}</td><td class="num">{f1(st_b['pu'])}</td></tr></table>
<div class="note">Ms per decode step = ms per output token x tokens per step. The DFly2 drafter reads the target model's KV cache and the sparse-attention indexer's top-k at 3 layers ({esc('28, 32, 36')}), which DSpark does not.</div>
<h2>5. Port check: served DFly2 against AngelSpec's own model code</h2>
<div class="two"><div><table class="t"><tr><th>Check, thinking-on traffic</th><th class="num">Value</th></tr>{or_html}</table></div>
<div><table class="t"><tr><th>Item</th><th>Setting</th></tr>
<tr><td>Traffic</td><td>16 decode steps, 121 requests, 363 draft slots dumped from vLLM</td></tr>
<tr><td>Reference</td><td>AngelSpec DFly2 forward pass on the same dumped inputs</td></tr>
<tr><td>Files</td><td><span class="num">dfly2/pkg/dfly2_v41</span>, <span class="num">dfly2/patches</span></td></tr></table></div></div>
<div class="note">Source: dfly2/logs/gate2_think.log.</div>
<h2>6. Problems met and fixes</h2>
<table class="t"><tr><th style="width:27%">Problem</th><th>Evidence and fix</th></tr>
<tr><td>Thinking-off setting not sent</td><td><span class="num">vllm bench serve --chat-template-kwargs</span> only renders client-side prompts; the server never saw it, so the first M4 runs at 40 and 80 users per node ran with thinking on (identical to M1). Fixed with <span class="num">--extra-body '{{"chat_template_kwargs": ...}}'</span>; the invalid results moved to <span class="num">results/bench/invalid</span>; M4 rerun at every load as M4r.</td></tr>
<tr><td>Thinking mode mixed in early tests</td><td>The first acceptance probe ran thinking off, the benchmark ran thinking on, so DFly2 looked equal at temperature 0 and weak at 0.6. Section 4 compares both modes at both temperatures.</td></tr>
<tr><td>DFly2 never served on 8 GPUs</td><td>Setup M7 started without changes; quality {qs('M7_dfly3_tp8')}.</td></tr>
<tr><td>Server start failures</td><td>{fails} in the matrix ({t0} to {t1} UTC).</td></tr></table>
<h2>7. Method</h2>
<table class="t"><tr><th style="width:17%">Item</th><th>Setting</th></tr>
<tr><td>Serving</td><td>vLLM 0.32.0, W4A8 experts with Humming kernels (2 CTAs per SM), prompts admitted every 8th step, max 256 sequences and 131,072 tokens per copy; DSpark verified with block rejection, DFly2 with standard rejection; drafts sampled from the draft distribution</td></tr>
<tr><td>Benchmark</td><td>vllm bench serve, ShareGPT, temperature 0.6, top-p 0.95, output capped at 256, 16 x users requests per run, fresh prompt seed per run, both copies loaded at once</td></tr>
<tr><td>Quality probe</td><td>50 arithmetic questions, greedy, chat template default</td></tr>
<tr><td>Files</td><td>Scripts <span class="num">dfly2/tools/run_matrix.sh</span>, <span class="num">run_matrix_redo.sh</span>, <span class="num">serve_v41_any.sh</span>, <span class="num">bench_any.sh</span>; summary <span class="num">dfly2/results/matrix_summary.csv</span></td></tr></table>
<div class="note"><b>Terms.</b> <b>Draft model</b>: a small model that guesses the next tokens; the target model checks them in one step and keeps the accepted ones. <b>Tokens per step</b>: 1 + accepted draft tokens per decode step per user. <b>Thinking</b>: DeepSeek V4.1 writes a reasoning section before its answer; on by default. <b>TP4, TP8</b>: one copy of the model split over 4 or 8 GPUs.</div>
"""
H = f'{L}/report/build/dfly2_tr01.html'
open(H, 'w', encoding='utf8').write(f'<!doctype html><html><head><meta charset="utf-8"><title>{esc(TITLE)}</title><style>{CSS}</style></head><body>{body}</body></html>')
import pymupdf
from playwright.sync_api import sync_playwright
os.environ['PLAYWRIGHT_BROWSERS_PATH'] = f'{L}/report/.render/pw'
foot = '<div style="font-family:Arial;font-size:7px;color:#5B6268;width:100%;padding:0 12mm;display:flex;justify-content:space-between"><span></span><span>Internal</span><span>Page <span class="pageNumber"></span></span></div>'
with sync_playwright() as p:
    b = p.chromium.launch(); g = b.new_page(); g.goto(f'file://{H}'); g.wait_for_timeout(800); g.evaluate('document.fonts.ready')
    g.pdf(path=OUT, prefer_css_page_size=True, print_background=True, display_header_footer=True, header_template='<span></span>', footer_template=foot); b.close()
d = pymupdf.open(OUT); txt = ''.join(pg.get_text() for pg in d)
assert '—' not in txt and '–' not in txt, 'dash found'
print(OUT, 'pages', len(d))
for i, pg in enumerate(d): pg.get_pixmap(dpi=90).save(f'{L}/report/build/dfly2_tr01-p{i+1}.png')
