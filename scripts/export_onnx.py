#!/usr/bin/env python3
"""best.pt を ONNX に書き出し、int8(動的量子化)も作る。
入力: input_ids, attention_mask (int64 [batch, seq])。出力: cls_logits [batch,13], complete_logit [batch]
使い方: .venv/bin/python scripts/export_onnx.py out/final"""
import sys, pathlib, json, shutil, time
import numpy as np, torch
sys.path.insert(0, str(pathlib.Path(__file__).parent))
from train import TwoHead, CLASSES, MODEL_NAME, MAX_LEN, normalize, load_rows, ROOT, AizuchiDataset, make_collate
from transformers import AutoTokenizer
import onnxruntime as ort
from onnxruntime.quantization import quantize_dynamic, QuantType

run = pathlib.Path(sys.argv[1])
tok = AutoTokenizer.from_pretrained(MODEL_NAME)
model = TwoHead(MODEL_NAME, len(CLASSES))
model.load_state_dict(torch.load(run / 'best.pt', map_location='cpu'))
model.eval()
# 書き出しは eager attention で(unpad/flash の経路はトレースできない)
model.encoder.config._attn_implementation = 'eager'

class Wrapped(torch.nn.Module):
    def __init__(self, m): super().__init__(); self.m = m
    def forward(self, input_ids, attention_mask):
        cls_logits, complete_logit = self.m(input_ids, attention_mask)
        return cls_logits, complete_logit

enc = tok(['明日の予定教えて', 'いや違う月曜のほう'], padding=True, return_tensors='pt')
fp32 = run / 'aizuchi.onnx'
torch.onnx.export(
    Wrapped(model), (enc['input_ids'], enc['attention_mask']), str(fp32),
    input_names=['input_ids', 'attention_mask'], output_names=['cls_logits', 'complete_logit'],
    dynamic_axes={'input_ids': {0: 'batch', 1: 'seq'}, 'attention_mask': {0: 'batch', 1: 'seq'},
                  'cls_logits': {0: 'batch'}, 'complete_logit': {0: 'batch'}},
    opset_version=17, dynamo=False)
int8 = run / 'aizuchi.int8.onnx'
quantize_dynamic(str(fp32), str(int8), weight_type=QuantType.QInt8)
print('sizes MB fp32', round(fp32.stat().st_size / 1e6, 1), 'int8', round(int8.stat().st_size / 1e6, 1))

# tokenizer.json と labels を同梱する
snap = next((pathlib.Path.home() / '.cache/huggingface/hub/models--sbintuitions--modernbert-ja-70m/snapshots').iterdir())
shutil.copy(snap / 'tokenizer.json', run / 'tokenizer.json')
(run / 'labels.json').write_text(json.dumps({'classes': CLASSES, 'max_len': MAX_LEN, 'model': MODEL_NAME}, ensure_ascii=False))

# torch と ONNX の一致、および速度
rows = [r for r in load_rows(ROOT / 'data' / 'all.jsonl')][:512]
dl = torch.utils.data.DataLoader(AizuchiDataset(rows, tok, train=False), batch_size=64, collate_fn=make_collate(tok))
ref, got = {}, {}
for name, path in (('fp32', fp32), ('int8', int8)):
    sess = ort.InferenceSession(str(path), providers=['CPUExecutionProvider'])
    agree = 0; n = 0; maxdiff = 0.0
    for b in dl:
        with torch.no_grad():
            t_cls, _ = model(b['input_ids'], b['attention_mask'])
        o_cls, _ = sess.run(None, {'input_ids': b['input_ids'].numpy(), 'attention_mask': b['attention_mask'].numpy()})
        agree += int((t_cls.argmax(-1).numpy() == o_cls.argmax(-1)).sum()); n += len(o_cls)
        maxdiff = max(maxdiff, float(np.abs(torch.softmax(t_cls, -1).numpy() - np.exp(o_cls) / np.exp(o_cls).sum(-1, keepdims=True)).max()))
    # 1文の遅延(本番は部分認識ごとに1文ずつ)
    one = tok('えっとね来週の予定なんだけど', return_tensors='np')
    feed = {'input_ids': one['input_ids'].astype(np.int64), 'attention_mask': one['attention_mask'].astype(np.int64)}
    for _ in range(5): sess.run(None, feed)
    t = time.perf_counter(); [sess.run(None, feed) for _ in range(50)]; ms = (time.perf_counter() - t) / 50 * 1000
    print(f'{name}: argmax agreement {agree}/{n}, max prob diff {maxdiff:.4f}, 1-sentence latency {ms:.1f} ms')
