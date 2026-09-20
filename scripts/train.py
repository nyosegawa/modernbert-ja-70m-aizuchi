#!/usr/bin/env python3
"""相槌分類器の学習(ModernBERT-ja-70m、2ヘッド)。

入力: 「直前のアシスタント発話 [SEP] ユーザーの部分認識」。句読点と空白は落とす。
出力: cls(13クラス)と complete(言い切っているか)。

学習時の水増し:
- prev を確率 PREV_DROP で落とす(prev が無くても判断できるように)
- 確率 PREFIX_P でトークン境界の前方部分に切る。切った行は complete=False。
  切った長さが元の CLS_MIN_RATIO 未満か CLS_MIN_CHARS 未満なら cls の損失は掛けない
  (「明日の」に work を教えない)

検証は場面単位で外す(VAL_SCENES)。同じ場面の中は言い回しが似るので、ランダム分割だと高く出すぎる。

使い方:
  .venv/bin/python scripts/train.py [--epochs 5] [--lr 5e-5] [--seed 0] [--out out/run1]
"""
from __future__ import annotations

import argparse, json, math, os, random, re, sys, time, unicodedata, collections, pathlib
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModel, get_linear_schedule_with_warmup
from sklearn.metrics import f1_score, confusion_matrix, roc_auc_score

ROOT = pathlib.Path(__file__).resolve().parents[1]
CLASSES = ['none', 'flow', 'ack', 'work', 'think', 'check', 'understand', 'agree', 'empathy', 'cheer', 'surprise', 'correct', 'hold']
C2I = {c: i for i, c in enumerate(CLASSES)}
MODEL_NAME = 'sbintuitions/modernbert-ja-70m'
VAL_SCENES = ('hobby', 'travel', 'shopping')
MAX_LEN = 64
PREV_DROP = 0.3
PREFIX_P = 0.6
CLS_MIN_RATIO = 0.4
CLS_MIN_CHARS = 4

PUNCT = re.compile(r'[\s、。,.?？!！…・「」『』()（）]')


def normalize(text: str) -> str:
    return PUNCT.sub('', unicodedata.normalize('NFKC', text))


def load_rows(path: pathlib.Path) -> list[dict]:
    rows = []
    for line in path.read_text().splitlines():
        if not line.strip():
            continue
        r = json.loads(line)
        rows.append({
            'text': normalize(r['text']),
            'prev': normalize(r['prev']) if r.get('prev') else '',
            'cls': C2I[r['cls']],
            'complete': bool(r['complete']),
            'src': r.get('src', ''),
        })
    return [r for r in rows if r['text']]


class AizuchiDataset(Dataset):
    """train=True なら水増しをかける。eval 用には prefix_ratio で固定の切り出しを指定できる"""

    def __init__(self, rows: list[dict], tok, train: bool, prefix_ratio: float | None = None, rng_seed: int = 0):
        self.rows, self.tok, self.train, self.prefix_ratio = rows, tok, train, prefix_ratio
        self.rng = random.Random(rng_seed)
        # トークン境界で切るために、text 単独のトークン列を持つ
        self.text_tokens = [tok(r['text'], add_special_tokens=False)['input_ids'] for r in rows]

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, i: int):
        r = self.rows[i]
        ids = self.text_tokens[i]
        prev = r['prev']
        cls_label = r['cls']
        complete = r['complete']
        n = len(ids)
        cut = None
        if self.train:
            if prev and self.rng.random() < PREV_DROP:
                prev = ''
            if n >= 2 and self.rng.random() < PREFIX_P:
                cut = self.rng.randint(1, n - 1)
        elif self.prefix_ratio is not None and n >= 2:
            cut = max(1, min(n - 1, int(round(n * self.prefix_ratio))))
        if cut is not None:
            text = self.tok.decode(ids[:cut])
            complete = False
            if len(text) < CLS_MIN_CHARS or len(text) < CLS_MIN_RATIO * len(r['text']):
                cls_label = -100
        else:
            text = r['text']
        return {'prev': prev, 'text': text, 'cls': cls_label, 'complete': float(complete)}


def make_collate(tok):
    def collate(batch):
        # prev があれば pair、無ければ text 単独。ModernBERT の tokenizer が <s> a </s></s> b </s> 形に組む
        prevs = [b['prev'] for b in batch]
        texts = [b['text'] for b in batch]
        # pair と単独を分けて組んでから揃える
        seqs = []
        for p, t in zip(prevs, texts):
            if p:
                seqs.append(tok(p, t, truncation='only_first', max_length=MAX_LEN, add_special_tokens=True)['input_ids'])
            else:
                seqs.append(tok(t, truncation=True, max_length=MAX_LEN, add_special_tokens=True)['input_ids'])
        L = max(len(s) for s in seqs)
        pad = tok.pad_token_id
        input_ids = torch.full((len(seqs), L), pad, dtype=torch.long)
        attn = torch.zeros((len(seqs), L), dtype=torch.long)
        for k, s in enumerate(seqs):
            input_ids[k, :len(s)] = torch.tensor(s)
            attn[k, :len(s)] = 1
        return {
            'input_ids': input_ids,
            'attention_mask': attn,
            'cls': torch.tensor([b['cls'] for b in batch], dtype=torch.long),
            'complete': torch.tensor([b['complete'] for b in batch], dtype=torch.float),
        }
    return collate


class TwoHead(nn.Module):
    def __init__(self, name: str, n_cls: int, dropout: float = 0.1):
        super().__init__()
        self.encoder = AutoModel.from_pretrained(name)
        h = self.encoder.config.hidden_size
        self.drop = nn.Dropout(dropout)
        self.cls_head = nn.Linear(h, n_cls)
        self.complete_head = nn.Linear(h, 1)

    def forward(self, input_ids, attention_mask):
        out = self.encoder(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state
        pooled = self.drop(out[:, 0])  # <s> の位置
        return self.cls_head(pooled), self.complete_head(pooled).squeeze(-1)


def evaluate(model, loader, device) -> dict:
    model.eval()
    ys, ps, cy, cp = [], [], [], []
    with torch.no_grad():
        for b in loader:
            logits, clog = model(b['input_ids'].to(device), b['attention_mask'].to(device))
            pred = logits.argmax(-1).cpu()
            mask = b['cls'] != -100
            ys += b['cls'][mask].tolist()
            ps += pred[mask].tolist()
            cy += b['complete'].tolist()
            cp += torch.sigmoid(clog).cpu().tolist()
    model.train()
    res = {'n': len(ys)}
    if ys:
        res['acc'] = float(np.mean(np.array(ys) == np.array(ps)))
        res['macro_f1'] = float(f1_score(ys, ps, average='macro', labels=list(range(len(CLASSES)))))
        res['per_class_f1'] = dict(zip(CLASSES, f1_score(ys, ps, average=None, labels=list(range(len(CLASSES)))).round(3).tolist()))
        res['confusion'] = confusion_matrix(ys, ps, labels=list(range(len(CLASSES)))).tolist()
    if len(set(int(v) for v in cy)) == 2:
        res['complete_auc'] = float(roc_auc_score(cy, cp))
        res['complete_acc'] = float(np.mean((np.array(cp) > 0.5) == (np.array(cy) > 0.5)))
    return res


def print_confusion(cm: list[list[int]]):
    w = 6
    print(' ' * 8 + ''.join(f'{c[:5]:>{w}}' for c in CLASSES))
    for c, row in zip(CLASSES, cm):
        print(f'{c[:7]:8}' + ''.join(f'{v:>{w}}' for v in row))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--data', default=str(ROOT / 'data' / 'all.jsonl'))
    ap.add_argument('--out', default=str(ROOT / 'out' / 'run'))
    ap.add_argument('--epochs', type=int, default=5)
    ap.add_argument('--lr', type=float, default=5e-5)
    ap.add_argument('--batch', type=int, default=32)
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--eval_every', type=int, default=100)
    ap.add_argument('--wandb', default='asist-aizuchi')
    ap.add_argument('--val_scenes', default=','.join(VAL_SCENES),
                    help='検証に外す場面(カンマ区切り)。空なら場面を外さず乱数で5%を検証にする(最終モデル用)')
    args = ap.parse_args()

    random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed)
    out = pathlib.Path(args.out); out.mkdir(parents=True, exist_ok=True)
    device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')

    rows = load_rows(pathlib.Path(args.data))
    val_scenes = tuple(s for s in args.val_scenes.split(',') if s)
    if val_scenes:
        train_rows = [r for r in rows if r['src'] not in val_scenes]
        val_rows = [r for r in rows if r['src'] in val_scenes]
    else:
        shuffled = rows[:]; random.Random(args.seed).shuffle(shuffled)
        k = len(shuffled) // 20
        val_rows, train_rows = shuffled[:k], shuffled[k:]
    print(f'train {len(train_rows)}  val {len(val_rows)} (scenes {val_scenes or "random 5%"})  device {device}')

    tok = AutoTokenizer.from_pretrained(MODEL_NAME)
    collate = make_collate(tok)
    train_ds = AizuchiDataset(train_rows, tok, train=True, rng_seed=args.seed)
    train_dl = DataLoader(train_ds, batch_size=args.batch, shuffle=True, collate_fn=collate, drop_last=True)
    val_full = DataLoader(AizuchiDataset(val_rows, tok, train=False), batch_size=128, collate_fn=collate)
    val_p50 = DataLoader(AizuchiDataset(val_rows, tok, train=False, prefix_ratio=0.5), batch_size=128, collate_fn=collate)
    val_p75 = DataLoader(AizuchiDataset(val_rows, tok, train=False, prefix_ratio=0.75), batch_size=128, collate_fn=collate)

    # クラス重み: 件数の逆数の平方根(none/flow が少ないぶんを緩く補う)
    counts = collections.Counter(r['cls'] for r in train_rows)
    w = torch.tensor([1.0 / math.sqrt(counts[i]) for i in range(len(CLASSES))])
    w = (w / w.mean()).to(device)
    print('class weights', dict(zip(CLASSES, w.cpu().numpy().round(2).tolist())))

    model = TwoHead(MODEL_NAME, len(CLASSES)).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    total = args.epochs * len(train_dl)
    sched = get_linear_schedule_with_warmup(opt, int(0.1 * total), total)

    run = None
    if args.wandb:
        try:
            import wandb
            if wandb.api.api_key:
                run = wandb.init(project=args.wandb, name=out.name, config=vars(args) | {
                    'model': MODEL_NAME, 'val_scenes': val_scenes, 'prev_drop': PREV_DROP, 'prefix_p': PREFIX_P,
                    'cls_min_ratio': CLS_MIN_RATIO, 'train_rows': len(train_rows), 'val_rows': len(val_rows)})
        except Exception as e:  # wandb 未導入・未ログインなら黙って標準出力だけにする
            print('wandb disabled:', e)
    log = open(out / 'log.jsonl', 'a')

    def emit(rec: dict):
        rec['t'] = time.time()
        log.write(json.dumps(rec, ensure_ascii=False) + '\n'); log.flush()
        if run:
            run.log({k: v for k, v in rec.items() if isinstance(v, (int, float))}, step=rec.get('step'))

    best = -1.0
    step = 0
    model.train()
    t0 = time.time()
    for epoch in range(args.epochs):
        run_loss = collections.deque(maxlen=50)
        for b in train_dl:
            step += 1
            logits, clog = model(b['input_ids'].to(device), b['attention_mask'].to(device))
            cls_loss = F.cross_entropy(logits, b['cls'].to(device), weight=w, ignore_index=-100)
            comp_loss = F.binary_cross_entropy_with_logits(clog, b['complete'].to(device))
            loss = cls_loss + comp_loss
            opt.zero_grad(set_to_none=True)
            loss.backward()
            gn = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step(); sched.step()
            run_loss.append(loss.item())
            if step % 20 == 0:
                emit({'step': step, 'epoch': round(step / len(train_dl), 3),
                      'loss': float(np.mean(run_loss)), 'cls_loss': cls_loss.item(), 'comp_loss': comp_loss.item(),
                      'grad_norm': float(gn), 'lr': sched.get_last_lr()[0]})
                print(f'step {step:5d} ep {step / len(train_dl):.2f} loss {np.mean(run_loss):.3f} '
                      f'(cls {cls_loss.item():.3f} comp {comp_loss.item():.3f}) gn {float(gn):.2f} '
                      f'lr {sched.get_last_lr()[0]:.2e} {time.time() - t0:.0f}s', flush=True)
            if step % args.eval_every == 0 or step == total:
                full = evaluate(model, val_full, device)
                p50 = evaluate(model, val_p50, device)
                p75 = evaluate(model, val_p75, device)
                rec = {'step': step, 'val_f1': full['macro_f1'], 'val_acc': full['acc'],
                       'val_complete_auc': full.get('complete_auc'), 'val_f1_p50': p50['macro_f1'], 'val_f1_p75': p75['macro_f1'],
                       'per_class_f1': full['per_class_f1']}
                emit(rec)
                print(f'  EVAL step {step}: full f1 {full["macro_f1"]:.3f} acc {full["acc"]:.3f} | '
                      f'p75 f1 {p75["macro_f1"]:.3f} | p50 f1 {p50["macro_f1"]:.3f} | '
                      f'complete auc full {full.get("complete_auc", float("nan")):.3f} p50 {p50.get("complete_auc", float("nan")):.3f}', flush=True)
                print('  per-class', full['per_class_f1'], flush=True)
                if full['macro_f1'] > best:
                    best = full['macro_f1']
                    torch.save(model.state_dict(), out / 'best.pt')
                    (out / 'best.json').write_text(json.dumps({'step': step, **full, 'p50': p50, 'p75': p75}, ensure_ascii=False, indent=1))
                    print(f'  saved best ({best:.3f})', flush=True)
    print('done. best val macro-F1', best)
    best_info = json.loads((out / 'best.json').read_text())
    print_confusion(best_info['confusion'])
    if run:
        run.summary['best_val_f1'] = best
        run.finish()


if __name__ == '__main__':
    main()
