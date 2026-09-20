# 相槌分類器の学習

ユーザーの発話の途中・直後に ASIST が最初に出す音(相槌の種類)を決める分類器。
sbintuitions/modernbert-ja-70m を fine-tune し、ONNX(int8)で本体の runtime から動かす。
クラスの定義は [LABELS.md](LABELS.md)、データの生成に使った指示は [GEN_PROMPT.md](GEN_PROMPT.md)。

## データ

`data/<場面>.jsonl` は20の場面ごとに Agent が書いた合成データ(各 13クラス × 40件)。
1行の形は `{"text", "cls", "prev", "complete"}`。周囲の雑音や無関係な発話は入れない。
実ログはテレビの音が混じった日があるため学習には使わない。

```bash
python3 scripts/check_data.py --write   # 形式の検証、句読点を無視した重複の除去 → data/all.jsonl
```

2026-09-20 時点で重複除去後 9,045件。none と flow は短い定型が場面をまたいで重なるため少なめ(430前後)。

## 学習

```bash
uv sync
.venv/bin/wandb login                    # 任意。未ログインなら標準出力と out/<run>/log.jsonl だけに記録する
.venv/bin/python scripts/train.py --out out/run1                      # hobby/travel/shopping を検証に外す
.venv/bin/python scripts/train.py --out out/final --val_scenes "" --epochs 4   # 全場面で学習(最終モデル)
.venv/bin/python scripts/errors.py out/run1                           # 混同行列と間違えた例
.venv/bin/python scripts/export_onnx.py out/final                     # aizuchi.onnx / aizuchi.int8.onnx / tokenizer.json / labels.json
```

入力は「直前のアシスタント発話 [SEP] ユーザーの部分認識」で、句読点と空白は両方から落とす
(ASR の句読点は付いたり付かなかったりで、check だけ「?」が多いなど生成データに偏りがあるため)。
出力は cls(13クラス)と complete(言い切っているか)の2ヘッド。
学習時の水増しは prev の脱落(3割)とトークン境界の前方切り出し(6割)で、切った行は complete=False、
元の4割未満か4文字未満の切り出しには cls の損失を掛けない。詳細は `scripts/train.py` の冒頭。

## 結果(2026-09-20)

| 実験 | 検証 | macro-F1(全文) | 75%で切った文 | 50%で切った文 | 言い切り AUC(全文+切り出し) |
|---|---|---|---|---|---|
| run1 | 未見の3場面(1,382件) | 0.910 | 0.80 | 0.72 | 0.93 |
| final | 乱数5%(452件) | 0.863 | 0.74 | 0.63 | - |

final の値が低いのは検証に smalltalk・coding のような難しい場面が入るため。
残る誤りの大半は cheer/surprise、empathy/surprise、check/understand のように人が付けても割れる例で、
生成データのラベル自体が揺れている行も多い。本番の物差しは実ログに手で付けたテストセット(未作成)。

ONNX は batch=1 で torch と一致(fp32 は確率差 0、int8 は最大 0.08 で argmax は一致)。
padding を含む batch では traced graph の mask 経路が異なり確率がずれるので、本番は1文ずつ推論する。
M5 の CPU で1文 fp32 3ms、int8 1.7ms。
