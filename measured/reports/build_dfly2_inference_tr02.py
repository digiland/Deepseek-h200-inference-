#!/usr/bin/env python3
"""DFly2 Inference TR-02: Tencent AngelSpec DFly2 drafter fine-tuned on DeepSeek-V4.1-Flash-1017's own thinking-mode
replies, against the released DFly2 and V4.1's built-in DSpark drafter, on held-out ShareGPT prompts, 8 x H200.
Two pages, A4 portrait. Every number is read from /data/nvme3/dfly2: results/bench/results.csv (tags E1 to E4),
results/quality/quality-E*.json, ft/logs/eval.log (acceptance probes), ft/logs/gen.log, ft/logs/convert_check.log,
ft/logs/roundtrip_check.log, ft/logs/train_ft_v1.log, ft/data/*, ft/runs/ft_v1 resolved config and the drafter files.
Run: report/.render/venv/bin/python report/build_dfly2_inference_tr02.py"""
import csv, glob, html, json, os, re, struct
from theme import FONTS, LOGO

L = '/data/nvme3/lab'; W = '/data/nvme3/dfly2'; F = f'{W}/ft'
DATE = '27 Nov 2026'
TITLE = 'DFly2 Inference TR-02 - DFly2 fine-tuned on DeepSeek V4.1 thinking replies, against DSpark on 8 x H200'
OUTDIR = '/data/nvme3/inference/report'; os.makedirs(OUTDIR, exist_ok=True)
OUT = f'{OUTDIR}/{TITLE} ({DATE}).pdf'
REPO = 'github.com/digiland/Deepseek-h200-inference-, branch measured-runs'
esc = html.escape
f1 = lambda x: f'{x:.1f}'; f0 = lambda x: f'{x:,.0f}'
def pct(a, b): return f'{100 * (a / b - 1):+.1f}%'

# ---------- benchmark rows (2 copies x TP4, held-out prompts, thinking on) ----------
R = list(csv.DictReader(open(f'{W}/results/bench/results.csv')))
def node(tag, conc):
    rs = [r for r in R if r['conc'] == str(conc) and r['tag'] in (tag + 'A', tag + 'B')]
    n = len(rs)
    return dict(pu=sum(float(r['per_user_tok_s']) for r in rs) / n, tot=sum(float(r['output_tok_s']) for r in rs),
                acc=sum(float(r['accept_len']) for r in rs) / n, tpot=sum(float(r['tpot_p50_ms']) for r in rs) / n,
                ttft=sum(float(r['ttft_p50_ms']) for r in rs) / n)
SET = [('E1_dspark3_ho', 'DSpark 3 (production)'), ('E2_dfly3rel_ho', 'DFly2 3, released'),
       ('E3_dfly3ft_ho', 'DFly2 3, fine-tuned'), ('E4_dfly4ft_ho', 'DFly2 4, fine-tuned')]
PERCOPY = [20, 40, 80, 100]
Q = {t: json.load(open(f'{W}/results/quality/quality-{t}A.json')) for t, _ in SET}
qs = lambda t: f"{sum(r['correct'] for r in Q[t]['rows'])} of {len(Q[t]['rows'])}"

head = ''.join(f'<th class="num">{esc(n)}</th>' for _, n in SET)
pu_html = ''
for c in PERCOPY:
    v = [node(t, c) for t, _ in SET]; best = max(x['pu'] for x in v)
    pu_html += f'<tr><td class="num">{2 * c}</td>' + ''.join(f'<td class="num">{"<b>" + f1(x["pu"]) + "</b>" if x["pu"] == best else f1(x["pu"])}</td>' for x in v) + '</tr>'
tot_tds = ''.join(f'<td class="num">{f0(node(t, 100)["tot"])}</td>' for t, _ in SET)
acc_tds = ''.join(f'<td class="num">{node(t, 40)["acc"]:.2f}</td>' for t, _ in SET)
q_tds = ''.join(f'<td class="num">{qs(t)}</td>' for t, _ in SET)
cmp_html = ''
for c in PERCOPY:
    e1, e2, e3, e4 = (node(t, c) for t, _ in SET)
    cmp_html += (f'<tr><td class="num">{2 * c}</td><td class="num">{pct(e3["pu"], e2["pu"])}</td><td class="num">{pct(e3["pu"], e1["pu"])}</td>'
                 f'<td class="num">{pct(e4["pu"], e3["pu"])}</td><td class="num">{pct(e2["pu"], e1["pu"])}</td></tr>')
def span(a, b):
    g = sorted(100 * (node(a, c)['pu'] / node(b, c)['pu'] - 1) for c in PERCOPY)
    return g[0], g[-1]
ft_vs_rel = span('E3_dfly3ft_ho', 'E2_dfly3rel_ho'); ft_vs_ds = span('E3_dfly3ft_ho', 'E1_dspark3_ho')
rel_vs_ds = span('E2_dfly3rel_ho', 'E1_dspark3_ho'); n4_vs_n3 = span('E4_dfly4ft_ho', 'E3_dfly3ft_ho')
rg = lambda g: f'{abs(g[1]):.1f} to {abs(g[0]):.1f}%' if g[1] < 0 else f'{g[0]:.1f} to {g[1]:.1f}%'
NLOSE = sum(node('E3_dfly3ft_ho', c)['pu'] < node('E1_dspark3_ho', c)['pu'] for c in PERCOPY)
# ms per decode step = ms per output token x tokens per step, 20 users per copy
step = {t: node(t, 20)['tpot'] * node(t, 20)['acc'] for t, _ in SET}

# ---------- acceptance probes ----------
P = {}
for line in open(f'{F}/logs/eval.log'):
    m = re.match(r'(E\d_\S+) T=0.6 .*accept_len=([\d.]+) .*per-position: (.*)', line)
    if m: P[m.group(1)] = (float(m.group(2)), [float(x) for x in re.findall(r'p\d=([\d.]+)', m.group(3))])
pr_html = ''.join(f'<tr><td>{esc(n)}</td><td class="num">{P[t][0]:.3f}</td>' + ''.join(f'<td class="num">{P[t][1][i]:.3f}</td>' if i < len(P[t][1]) else '<td class="num">n/r</td>' for i in range(4)) + '</tr>' for t, n in SET)
gap_closed = (P['E3_dfly3ft_ho'][0] - P['E2_dfly3rel_ho'][0]) / (P['E1_dspark3_ho'][0] - P['E2_dfly3rel_ho'][0])

# ---------- data ----------
held = len(json.load(open(f'{F}/data/sharegpt_heldout.json')))
nprompt = sum(1 for _ in open(f'{F}/data/train_prompts.jsonl'))
G = open(f'{F}/logs/gen.log').read()
gm = re.search(r'done ok (\d+) err (\d+) reply tokens (\d+) in (\d+) s', G)
g_ok, g_err, g_tok, g_s = (int(x) for x in gm.groups())
lens, cut = [], 0
for line in open(f'{F}/data/replies_v1.jsonl'):
    r = json.loads(line); lens.append(len(r['prompt_token_ids']) + len(r['token_ids'])); cut += r['finish_reason'] != 'stop'
lens.sort(); n_rep = len(lens)
C = open(f'{F}/logs/convert_check.log').read()
cm = re.search(r'exact token match (\d+) of (\d+) \(mismatches in cut replies (\d+)\)', C)
ex_ok, ex_n, ex_cut = (int(x) for x in cm.groups())
complete = n_rep - cut; ex_complete = complete - (ex_n - ex_ok - ex_cut)
RT = re.search(r'weights compared (\d+): identical values (\d+), different (\d+)', open(f'{F}/logs/roundtrip_check.log').read()).groups()

# ---------- drafter initialisation: release names against training names ----------
def header_keys(path):
    with open(path, 'rb') as f:
        n = struct.unpack('<Q', f.read(8))[0]; h = json.loads(f.read(n))
    h.pop('__metadata__', None); return set(h)
rel_keys = set(json.load(open(f'{W}/DS-V4.1-Flash-DFly2-B6/model.safetensors.index.json'))['weight_map'])
train_keys = header_keys(f'{F}/init/release_train_layout/model.safetensors')
shipped = json.load(open(f'{W}/AngelSpec/angelspec/config/dfly2_dsv41_flash_draft_config.json'))
ftcfg = json.load(open(f'{F}/init/draft_config_ft.json'))
cfg_diff = [k for k in ftcfg if ftcfg.get(k) != shipped.get(k)]

# ---------- training ----------
T = open(f'{F}/logs/train_ft_v1.log', errors='replace').read()
tm = re.search(r'Training completed: (\d+) steps in ([\d.]+)s', T)
steps, secs = int(tm.group(1)), float(tm.group(2))
init = re.search(r'draft init from \S+: loaded (\d+) tensors \((\d+) left at init, (\d+) ignored\)', T).groups()
met = {}
for m in re.finditer(r'\| *(\d+)/\d+ \[[^\]]*, loss=([\d.]+), acc=([\d.]+), acc_len=([\d.]+)', T.replace('\r', '\n')):
    met.setdefault(int(m.group(1)), tuple(float(x) for x in m.groups()[1:]))
WIN = 52
win_html = ''
for i in range(0, steps, WIN):
    v = [met[s] for s in range(i + 1, i + WIN + 1) if s in met]
    if i // WIN in (0, 1, 3, 5, 6):
        win_html += f'<tr><td>{i + 1} to {i + len(v)}</td>' + ''.join(f'<td class="num">{sum(x[j] for x in v) / len(v):.3f}</td>' for j in range(3)) + '</tr>'
RC = open(f'{F}/runs/ft_v1/dfly2_dsv41_flash/config.yaml').read()
rc = lambda k: re.search(rf'^\s*{k}: (.+)$', RC, re.M).group(1).strip()
dsz = sum(os.path.getsize(f) for f in glob.glob(f'{W}/DS-V4.1-Flash-DFly2-B6/*.safetensors')) / 1024 ** 3

e1_80, e2_80, e3_80 = node('E1_dspark3_ho', 40), node('E2_dfly3rel_ho', 40), node('E3_dfly3ft_ho', 40)
qv = sorted(sum(r['correct'] for r in Q[t]['rows']) for t, _ in SET)
qa = [('Does fine-tuning on V4.1 thinking replies make DFly2 faster?', f'<b>Yes.</b> The fine-tuned DFly2 with 3 drafts is {rg(ft_vs_rel)} faster per user than the released DFly2 at 40 to 200 users per node.'),
      ('Does the fine-tuned DFly2 beat DSpark?', f'<b>No.</b> It is {rg(ft_vs_ds)} slower per user than DSpark with 3 drafts at {NLOSE} of {len(PERCOPY)} loads (released DFly2: {rg(rel_vs_ds)} slower).'),
      ('Speed per user at 80 users per node', f'DSpark 3 <b>{f1(e1_80["pu"])}</b> tok/s; DFly2 3 fine-tuned {f1(e3_80["pu"])} ({pct(e3_80["pu"], e1_80["pu"])}); released {f1(e2_80["pu"])} ({pct(e2_80["pu"], e1_80["pu"])}).'),
      ('Draft acceptance after fine-tuning', f'{P["E3_dfly3ft_ho"][0]:.2f} tokens per step against {P["E2_dfly3rel_ho"][0]:.2f} released and {P["E1_dspark3_ho"][0]:.2f} for DSpark: fine-tuning closed {100 * gap_closed:.0f}% of the acceptance gap. First draft accepted {P["E3_dfly3ft_ho"][1][0]:.2f} (released {P["E2_dfly3rel_ho"][1][0]:.2f}, DSpark {P["E1_dspark3_ho"][1][0]:.2f}).'),
      ('Why the fine-tuned DFly2 is still slower', f'A longer decode step at near-equal acceptance: {step["E3_dfly3ft_ho"]:.1f} ms against {step["E1_dspark3_ho"]:.1f} ms for DSpark at 40 users per node.'),
      ('4 drafts instead of 3', f'{P["E4_dfly4ft_ho"][0]:.2f} tokens per step but {rg(n4_vs_n3)} slower per user than 3 drafts.'),
      ('Training data and run', f'{g_ok:,} V4.1 thinking-on replies ({g_tok / 1e6:.1f} million reply tokens) to ShareGPT prompts; {steps} steps, 2 epochs, {secs / 3600:.1f} h on 8 GPUs, starting from the released weights.'),
      ('Quality probe, 50 questions', f'Every setup scored {qv[0]} of 50 to {qv[-1]} of 50.')]

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
<h1>DFly2 fine-tuned on DeepSeek V4.1 thinking replies, against DSpark on 8 x H200</h1>
<div class="date">{DATE} · Internal · target model deepseek-ai/DeepSeek-V4.1-Flash-1017 · drafters: DSpark (built into the checkpoint), Tencent AngelSpec DFly2 as released ({dsz:.1f} GiB) and DFly2 fine-tuned here · vLLM 0.32.0 · node cpt-gpu170, 8 x NVIDIA H200 141 GB</div>
<h2>1. Summary</h2>
<p>The <b>DFly2</b> draft model was fine-tuned on {g_ok:,} replies that <b>DeepSeek V4.1 Flash</b> itself wrote with thinking on, then served and measured against the released DFly2 and the built-in <b>DSpark</b> draft model on ShareGPT prompts held out from training (2 copies of 4 GPUs, thinking on). <b>Fine-tuning made DFly2 {rg(ft_vs_rel)} faster per user; DSpark with 3 drafts stays {rg(ft_vs_ds)} faster.</b></p>
<table class="t"><tr><th style="width:27%">Question</th><th>Our answer</th></tr>{''.join(f'<tr><td>{esc(a)}</td><td>{b}</td></tr>' for a, b in qa)}</table>
<h2>2. Speed per user on held-out prompts, 2 copies of 4 GPUs, tok/s (bold: best at that load)</h2>
<table class="t"><tr><th class="num">Users per node</th>{head}</tr>{pu_html}
<tr><td>Node total at 200 users, tok/s</td>{tot_tds}</tr><tr><td>Tokens per decode step per user</td>{acc_tds}</tr><tr><td>Quality probe</td>{q_tds}</tr></table>
<div class="note">Source: dfly2/results/bench/results.csv, setups E1 to E4. Speed per user = 1,000 / median ms between one user's output tokens, mean of the 2 copies. "DFly2 3" = DFly2 with 3 draft tokens. Tokens per step at 80 users per node.</div>
<table class="t"><tr><th class="num">Users per node</th><th class="num">Fine-tuned against released (DFly2 3)</th><th class="num">Fine-tuned DFly2 3 against DSpark 3</th><th class="num">Fine-tuned DFly2 4 against DFly2 3</th><th class="num">Released DFly2 3 against DSpark 3</th></tr>{cmp_html}</table>
<div class="note">Change in speed per user; negative is slower.</div>
<h2>3. Draft acceptance on held-out prompts</h2>
<table class="t"><tr><th>Draft model</th><th class="num">Tokens per step</th><th class="num">Draft 1 accepted</th><th class="num">Draft 2 accepted</th><th class="num">Draft 3 accepted</th><th class="num">Draft 4 accepted</th></tr>{pr_html}</table>
<div class="note">Source: dfly2/ft/logs/eval.log. 160 held-out ShareGPT prompts, temperature 0.6, thinking on, one copy of 4 GPUs. Draft position n is accepted only if every earlier draft was. Ms per decode step at 40 users per node (ms per output token x tokens per step): {', '.join(f'{esc(n)} {step[t]:.1f}' for t, n in SET)}.</div>
<div class="pb"></div>{hdr}
<h2 style="margin-top:0">4. Fine-tuning data and run</h2>
<div class="two"><div><table class="t"><tr><th>Training data</th><th class="num">Value</th></tr>
<tr><td>ShareGPT conversations held out for benchmarks</td><td class="num">{held:,}</td></tr>
<tr><td>Unique training prompts available</td><td class="num">{nprompt:,}</td></tr>
<tr><td>V4.1 replies generated, errors</td><td class="num">{g_ok:,}, {g_err}</td></tr>
<tr><td>Reply tokens, generation time</td><td class="num">{g_tok / 1e6:.1f} M, {g_s / 60:.0f} min</td></tr>
<tr><td>Replies cut at 8,192 tokens</td><td class="num">{cut:,} ({100 * cut / n_rep:.1f}%)</td></tr>
<tr><td>Tokens per conversation, median, p95</td><td class="num">{lens[n_rep // 2]:,}, {lens[int(n_rep * .95)]:,}</td></tr>
<tr><td>Template rebuilds V4.1's exact tokens, complete replies</td><td class="num">{ex_complete:,} of {complete:,}</td></tr>
<tr><td>Same, cut replies (template adds closing tokens)</td><td class="num">{cut - ex_cut:,} of {cut:,}</td></tr></table></div>
<div><table class="t"><tr><th>Training run</th><th class="num">Value</th></tr>
<tr><td>Starting weights, tensors loaded</td><td class="num">{init[0]} ({init[1]} at init, {init[2]} ignored)</td></tr>
<tr><td>GPUs: V4.1 inference, DFly2 training</td><td class="num">4 (TP4), 4 (FSDP)</td></tr>
<tr><td>Steps, epochs, time</td><td class="num">{steps}, {rc('num_epochs')}, {secs / 3600:.1f} h</td></tr>
<tr><td>Rows per step, tokens per row</td><td class="num">{4 * int(rc('draft_accumulation_steps'))}, {int(rc('max_seq_length')):,}</td></tr>
<tr><td>Optimizer, learning rate</td><td class="num">{rc('optimizer_type')}, {float(rc('learning_rate')):.0e} to {float(rc('min_lr')):.0e}</td></tr>
<tr><td>Loss</td><td class="num">{rc('dflash_loss_objective')}, block {rc('dflash_block_size')}</td></tr>
<tr><td>Export to serving layout, weights identical in value (round trip)</td><td class="num">{RT[1]} of {RT[0]}</td></tr></table></div></div>
<table class="t" style="margin-top:4pt"><tr><th>Training steps</th><th class="num">Loss</th><th class="num">Draft 1 accepted (training data)</th><th class="num">Accepted length (trainer, block {rc('dflash_block_size')})</th></tr>{win_html}</table>
<div class="note">Source: dfly2/ft/logs (gen.log, convert_check.log, roundtrip_check.log, train_ft_v1.log), resolved config runs/ft_v1/dfly2_dsv41_flash/config.yaml. Averages over {WIN}-step windows of single-batch values; epoch 2 repeats epoch 1's conversations.</div>
<h2>5. Problems met and fixes</h2>
<table class="t"><tr><th style="width:27%">Problem</th><th>Evidence and fix</th></tr>
<tr><td>Released weights not loaded by the trainer</td><td>The release uses V4.1 serving names; {len(rel_keys & train_keys)} of {len(rel_keys):,} release tensor names match the trainer's, and the shipped draft config lacks {len(cfg_diff)} settings the release was trained with ({esc(', '.join(cfg_diff))}). Fixed with a converter (FP4 experts dequantized and stacked) and a matching draft config: {init[0]} tensors loaded, {init[2]} ignored.</td></tr>
<tr><td>AngelSpec's vLLM patches not applied</td><td>The patches load through a <span class="num">vllm.general_plugins</span> entry point that exists only for an installed package; without them V4.1's attention input was not captured. Fixed by registering the entry point with a package metadata folder on the Python path.</td></tr>
<tr><td>Hidden-state dump to disk fails</td><td>AngelSpec's frozen-asset exporter expects Hugging Face names (<span class="num">self_attn.kv_a_proj_with_mqa</span>); V4.1 uses <span class="num">attn.wkv</span>. Trained online instead: inference and training on 4 GPUs each.</td></tr>
<tr><td>Thinking text rebuilt with wrong spaces</td><td>vLLM's reasoning parser strips the space before <span class="num">&lt;/think&gt;</span>. Reasoning and answer are now cut from V4.1's own decoded tokens.</td></tr>
<tr><td>Smaller fixes</td><td>Image input switched off for text-only capture; Ray temp path shortened (socket path limit 107 bytes); stand-in <span class="num">wandb</span> module with <span class="num">run = None</span>.</td></tr></table>
<h2>6. Method</h2>
<table class="t"><tr><th style="width:17%">Item</th><th>Setting</th></tr>
<tr><td>Prompt split</td><td>ShareGPT V3 conversation id md5 mod 100 below 15 held out for benchmarks; first user turns of the rest (20 to 6,000 characters, unique) for training</td></tr>
<tr><td>Replies</td><td>Production servers (2 copies of 4 GPUs, DSpark 3), thinking on, temperature 0.6, top-p 0.95, up to 8,192 tokens</td></tr>
<tr><td>Serving</td><td>vLLM 0.32.0, W4A8 experts with Humming kernels, prompts admitted every 8th step, 256 sequences per copy; DSpark with block rejection, DFly2 with standard rejection; fine-tuned experts requantized to MXFP4 like the release</td></tr>
<tr><td>Benchmark</td><td>vllm bench serve on held-out ShareGPT, temperature 0.6, top-p 0.95, output capped at 256, 16 x users requests per run, seeds 71000 and 72000, both copies loaded at once</td></tr>
<tr><td>Files</td><td>Scripts and runbook in {esc(REPO)} under <span class="num">measured/dfly2/finetune</span> (REPLICATE.md); node copies in <span class="num">dfly2/ft</span></td></tr></table>
<div class="note"><b>Terms.</b> <b>Draft model</b>: a small model that guesses the next tokens; the target model checks them in one step and keeps the accepted ones. <b>Tokens per step</b>: 1 + accepted draft tokens per decode step per user. <b>Thinking</b>: DeepSeek V4.1 writes a reasoning section before its answer; on by default. <b>Held out</b>: never used for training prompts. <b>TP4</b>: one copy of the model split over 4 GPUs.</div>
"""
H = f'{L}/report/build/dfly2_tr02.html'
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
for i, pg in enumerate(d): pg.get_pixmap(dpi=90).save(f'{L}/report/build/dfly2_tr02-p{i+1}.png')
