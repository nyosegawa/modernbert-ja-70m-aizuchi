#!/usr/bin/env python3
"""合成データの検査。形式の検証、重複の除去、クラスごとの件数を出す。

使い方:
  python3 scripts/check_data.py            # 検査して集計を表示
  python3 scripts/check_data.py --write    # 重複と不正行を除いて data/all.jsonl に書く
"""
import json, sys, re, unicodedata, collections, pathlib

ROOT = pathlib.Path(__file__).resolve().parents[1]
DATA = ROOT / 'data'
CLASSES = ['none','flow','ack','work','think','check','understand','agree','empathy','cheer','surprise','correct','hold']

def normalize(text: str) -> str:
    t = unicodedata.normalize('NFKC', text)
    t = re.sub(r'[\s、。,.?？!！…・「」『』]', '', t)
    return t.lower()

def main() -> None:
    write = '--write' in sys.argv
    rows, bad, seen = [], [], {}
    per_file = collections.defaultdict(collections.Counter)
    for path in sorted(DATA.glob('*.jsonl')):
        if path.name == 'all.jsonl':
            continue
        for lineno, line in enumerate(path.read_text().splitlines(), 1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as e:
                bad.append((path.name, lineno, f'json: {e}')); continue
            text, cls, prev, complete = row.get('text'), row.get('cls'), row.get('prev'), row.get('complete')
            if not isinstance(text, str) or not text.strip():
                bad.append((path.name, lineno, 'text')); continue
            if cls not in CLASSES:
                bad.append((path.name, lineno, f'cls={cls}')); continue
            if prev is not None and not isinstance(prev, str):
                bad.append((path.name, lineno, 'prev')); continue
            if not isinstance(complete, bool) or complete != (cls != 'hold'):
                bad.append((path.name, lineno, f'complete={complete} for {cls}')); continue
            if cls in ('flow', 'correct', 'check') and not prev:
                bad.append((path.name, lineno, f'{cls} without prev')); continue
            key = normalize(text)
            if key in seen:
                bad.append((path.name, lineno, f'dup of {seen[key]}: {text}')); continue
            seen[key] = f'{path.name}:{lineno}'
            per_file[path.name][cls] += 1
            rows.append({'text': text.strip(), 'cls': cls, 'prev': prev, 'complete': complete, 'src': path.stem})

    total = collections.Counter(r['cls'] for r in rows)
    print(f'{"file":12}' + ''.join(f'{c[:5]:>6}' for c in CLASSES) + '  total')
    for name, counts in sorted(per_file.items()):
        print(f'{name[:12]:12}' + ''.join(f'{counts[c]:>6}' for c in CLASSES) + f'{sum(counts.values()):>7}')
    print(f'{"TOTAL":12}' + ''.join(f'{total[c]:>6}' for c in CLASSES) + f'{len(rows):>7}')
    short = collections.Counter(r['cls'] for r in rows if len(r['text']) < 10)
    print('short(<10 chars):', dict(short))
    with_prev = sum(1 for r in rows if r['prev'])
    print(f'with prev: {with_prev}/{len(rows)}   bad/dup lines: {len(bad)}')
    kinds = collections.Counter(b[2].split(':')[0].split(' ')[0] for b in bad)
    print('bad kinds:', dict(kinds))
    for b in bad[:30]:
        print('  ', *b)
    if write:
        out = DATA / 'all.jsonl'
        out.write_text(''.join(json.dumps(r, ensure_ascii=False) + '\n' for r in rows))
        print('wrote', out, len(rows))

if __name__ == '__main__':
    main()
