# Byte check: release -> training layout -> release must reproduce the release tensors.
import sys, os, torch
sys.path[:0] = ["/data/nvme3/dfly2/ft"]
from train_to_release import load_dir
a, b = load_dir(sys.argv[1]), load_dir(sys.argv[2])
same = diff = 0; bad = []
for k, v in a.items():
    w = b.get(k)
    if w is not None and w.dtype == v.dtype and w.shape == v.shape and torch.equal(w.view(torch.uint8) if w.element_size()==1 else w, v.view(torch.uint8) if v.element_size()==1 else v):
        same += 1
    else:
        diff += 1; bad.append(k)
print(f"release tensors {len(a)}, identical {same}, different {diff}, extra in export {len(set(b)-set(a))}")
print("first differences:", bad[:5])
