あなたは音声アシスタント ASIST の相槌分類器(ModernBERT-ja-70m)の学習データを書く。
まず /Users/sakasegawa/src/github.com/nyosegawa/asist/training/aizuchi/LABELS.md を読むこと。

## 作るもの

場面「{DOMAIN}」について、13クラス × 40件 = 520件を JSONL で書き、
/Users/sakasegawa/src/github.com/nyosegawa/asist/training/aizuchi/data/{SLUG}.jsonl に Write ツールで保存する。
1行1件、形は LABELS.md のとおり。Bash や Python で文を組み合わせて量産してはいけない。全件を自分で書く。
ファイルは1回で書き切れなければ、まず前半を Write し、残りは Bash の `cat >> file <<'EOF'` で追記してよい。
最後に `wc -l` と、各 cls の件数を `python3` で数えて 40 ずつあることを確かめる。40 に満たないクラスがあれば追記する。

## 文体(重要)

- 実際の音声認識の書き起こしの体裁で書く。句読点は少なめ(3割程度の文にだけ「、」「。」「?」を付ける)。
- 話し言葉の崩れを混ぜる: 助詞の省略(「明日の予定教えて」)、言い淀み(「あ、なんかさ」「えっと」「うんと」)、言い直し、語尾の伸び(「〜なんだけどさ」)。
- 丁寧語とくだけた口調を半々にする。長さは 3文字から 60文字まで幅広く。短い文(10文字未満)を各クラスに 8件以上入れる(hold は除く)。
- 40件の中で同じ語彙・同じ構文を繰り返さない。名詞(固有名詞、日付、時間、人名、数字)は毎回変える。
- 周囲の雑音、テレビの音、他人との会話、無関係な独り言は入れない。すべて ASIST に向けた発話にする。

## クラスの境界(必ず守る)

各クラスの 40件のうち 8件以上は「隣のクラスに見えるがこちら」という境界例にする。

- work と ack: 外(Web、メール、カレンダー、株価、ファイル)を見に行かないと答えられないなら work。「タイマー」「覚えて」「音量」「もう一回言って」のようにその場で済むなら ack。
- work と check: 「明日の天気教えて」は work、「明日って雨だっけ」は check(短い事実確認。ただし調べる必要があっても、確認の形なら check)。
- think と check: 「なんで」「どう思う」「どっちがいい」「説明して」は think。「〜だっけ」「〜でいい?」「〜した?」は check。
- understand と agree: 事実の報告は understand、良し悪し・好み・判断を述べたら agree。
- understand と empathy / cheer / surprise: 気持ちが乗っていない報告は understand。困っている・疲れた・不満は empathy、うまくいった・嬉しいは cheer、意外だ・信じられないは surprise。
- flow と none: こちらの問いへの返事(うん / いいよ / それでお願い / いや違う…は correct)は flow。挨拶・感謝・別れ・断片は none。
- correct と flow: 「いや、そうじゃなくて」「違う、〜のほう」「さっき言ったのは〜」など、こちらの誤りを直す発話は correct。prev には誤った返答を入れる。
- hold: 「えっとね」「あのさ、昨日の」「で、なんか」「〜なんだけど」「〜してて」など、続きが来る形で終わる。complete は false。他クラスは true。

## prev

- flow・correct・check は必ず prev を付ける(ASIST が直前に言ったこと。問い・提案・報告・誤った返答など、10〜40文字)。
- 他のクラスは 40件のうち 20件前後に prev を付け、残りは null にする。prev がある場合も、text 単独でクラスが決まるものが大半でよい。

## 出力例

{"text": "明日の予定教えて", "cls": "work", "prev": null, "complete": true}
{"text": "いや違う、月曜のほう", "cls": "correct", "prev": "火曜日の予定は3件あります。", "complete": true}
{"text": "えっとね、来週のことなんだけど", "cls": "hold", "prev": null, "complete": false}

書き終えたら、ファイルのパスと各クラスの件数だけを報告する。
