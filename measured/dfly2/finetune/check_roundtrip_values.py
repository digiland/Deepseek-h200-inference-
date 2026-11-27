# Value check: dequantized experts and plain tensors of release vs release->train->release export.
import sys, torch
sys.path[:0] = ["/data/nvme3/dfly2/ft", "/data/nvme3/dfly2/AngelSpec"]
from train_to_release import load_dir
from angelspec.models.target.block_quant import dequant_fp4_packed
a, b = load_dir(sys.argv[1]), load_dir(sys.argv[2])
same = diff = 0; worst = 0.0
for k, v in a.items():
    if k.endswith(".scale"): continue
    if k.endswith(".weight") and v.dtype == torch.int8:
        s = k[:-len("weight")] + "scale"
        x, y = dequant_fp4_packed(v, a[s].float()), dequant_fp4_packed(b[k], b[s].float())
    else:
        x, y = v.float(), b[k].float()
    if torch.equal(x, y): same += 1
    else: diff += 1; worst = max(worst, (x - y).abs().max().item())
print(f"weights compared {same + diff}: identical values {same}, different {diff}, largest difference {worst}")
