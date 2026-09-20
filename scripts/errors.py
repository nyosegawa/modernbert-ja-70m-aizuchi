#!/usr/bin/env python3
"""学習済み best.pt を検証データにかけ、混同行列と間違えた例を出す。
使い方: .venv/bin/python scripts/errors.py out/run1 [--prefix 0.5]"""
import sys, json, pathlib, argparse, collections
import torch
sys.path.insert(0, str(pathlib.Path(__file__).parent))
from train import *  # noqa

ap = argparse.ArgumentParser(); ap.add_argument('run'); ap.add_argument('--prefix', type=float, default=None); ap.add_argument('--data', default=str(ROOT/'data'/'all.jsonl'))
args = ap.parse_args()
device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
tok = AutoTokenizer.from_pretrained(MODEL_NAME)
rows = [r for r in load_rows(pathlib.Path(args.data)) if r['src'] in VAL_SCENES]
ds = AizuchiDataset(rows, tok, train=False, prefix_ratio=args.prefix)
dl = DataLoader(ds, batch_size=128, collate_fn=make_collate(tok))
model = TwoHead(MODEL_NAME, len(CLASSES)).to(device)
model.load_state_dict(torch.load(pathlib.Path(args.run)/'best.pt', map_location=device)); model.eval()
preds, probs = [], []
with torch.no_grad():
    for b in dl:
        lg, _ = model(b['input_ids'].to(device), b['attention_mask'].to(device))
        p = torch.softmax(lg, -1).cpu(); preds += p.argmax(-1).tolist(); probs += p.max(-1).values.tolist()
res = evaluate(model, dl, device); print('macro_f1', round(res['macro_f1'],3)); print_confusion(res['confusion'])
pairs = collections.Counter()
errs = []
for r, item, p, pr in zip(rows, ds, preds, probs):
    if item['cls'] == -100: continue
    if p != r['cls']:
        pairs[(CLASSES[r['cls']], CLASSES[p])] += 1
        errs.append((CLASSES[r['cls']], CLASSES[p], round(pr,2), item['text'], r['prev'][:25]))
print('\ntop confusions:', pairs.most_common(12))
print('\nerrors (gold -> pred, conf, text, prev):')
for e in sorted(errs, key=lambda e: (e[0], e[1])): print('  ', *e)
