# modernbert-ja-70m-aizuchi

Training code and synthetic data for
[sakasegawa/modernbert-ja-70m-aizuchi](https://huggingface.co/sakasegawa/modernbert-ja-70m-aizuchi),
a small Japanese classifier that decides which kind of backchannel (相槌, aizuchi) a listener should
give first, and whether the user has finished the sentence. It fine-tunes
[sbintuitions/modernbert-ja-70m](https://huggingface.co/sbintuitions/modernbert-ja-70m) and exports
the result to ONNX (fp32 and int8).

The classes, the input and output, the evaluation and the limitations are described in the
[model card](https://huggingface.co/sakasegawa/modernbert-ja-70m-aizuchi). The data is also published
as the dataset [sakasegawa/aizuchi-synthetic-ja](https://huggingface.co/datasets/sakasegawa/aizuchi-synthetic-ja).

## Data

`data/<situation>.jsonl` holds the synthetic utterances for one of 20 situations a voice assistant
meets, 40 per class and 13 classes each. Each line is `{"text", "cls", "prev", "complete"}`: the
user's utterance written like speech-recognition output, its class, the assistant's previous
utterance or `null`, and whether the utterance is finished (false only for `hold`).

Claude (claude-fable-5-1) wrote every row on 2026-09-20, running as Claude Code subagents under a
Claude Max plan. [GEN_PROMPT.md](GEN_PROMPT.md) and [LABELS.md](LABELS.md) are the instructions it
was given, kept as they were, in Japanese. They were written for the voice assistant
[ASIST](https://github.com/nyosegawa/asist), where the model is used. No real conversation logs are
included.

```bash
python3 scripts/check_data.py --write   # validate, drop duplicates that differ only in punctuation, write data/all.jsonl
```

`data/all.jsonl` is generated and ignored by Git. It has 9,045 rows, each with `src` set to its
situation.

## Use of Claude's outputs

Claude was used under Anthropic's [Consumer Terms](https://www.anthropic.com/legal/consumer-terms),
which apply to Claude Code on the Free, Pro and Max plans
([Claude Code legal and compliance](https://code.claude.com/docs/en/legal-and-compliance)). The
Consumer Terms forbid using the services to develop products or services that compete with
Anthropic's, including training AI models, and the
[Usage Policy](https://www.anthropic.com/legal/aup) forbids training models on outputs without
authorization in the sense of model scraping and distillation.

Anthropic's help article
[Can I use my Outputs to train an AI model?](https://support.claude.com/en/articles/12326764)
(updated 2026-03-16) explains where that line is. It allows training models that do not compete with
Anthropic's own models and names specialized classifiers, among them content categorization, as
examples. It forbids general-purpose chatbots, models for open-ended text generation, using outputs as
training targets, and supporting a third party who tries to do those things.

This model is a 70M-parameter encoder that sorts an utterance into 13 classes and returns two
probabilities. It generates no text and cannot answer a question, so it falls under the specialized
classifiers the help article allows. The data is published under terms that keep it out of the
uses the help article forbids (see below), so that publishing it does not support a third party in
training a competing model.

This is the author's reading of Anthropic's terms as of 2026-10-12, not legal advice.

## Training

```bash
uv sync
.venv/bin/wandb login   # optional; without it the log goes to stdout and out/<run>/log.jsonl
.venv/bin/python scripts/train.py --out out/run1                                 # hold out hobby, travel and shopping
.venv/bin/python scripts/train.py --out out/final --val_scenes "" --epochs 4    # train on all situations (the released model)
.venv/bin/python scripts/errors.py out/run1                                      # confusion matrix and misclassified rows
.venv/bin/python scripts/export_onnx.py out/final                                # aizuchi.onnx, aizuchi.int8.onnx, tokenizer.json, labels.json
```

The input is the assistant's previous utterance and the user's partial transcript as a sentence pair,
with punctuation and whitespace removed from both. The model has two heads on the `<s>` vector: the
class (13 classes) and completeness. During training `prev` is dropped with probability 0.3, and the
text is cut at a token boundary with probability 0.6, which makes the row incomplete; a cut that keeps
less than 40% of the text or fewer than 4 characters gets no class loss. The details are at the top
of `scripts/train.py`.

The ONNX export matches PyTorch: the fp32 model gives the same probabilities, and the int8 model
differs by at most 0.08 with the same predicted class. Padded batches give the same probabilities as
single sentences with the fp32 model; the int8 model quantizes activations per run, so its
probabilities move by up to about 0.02 with the contents of the batch.

## Results (2026-09-20)

| run | validation set | macro-F1, full sentences | cut at 75% | cut at 50% | completeness AUC |
|---|---|---|---|---|---|
| run1 | 3 unseen situations, 1,382 rows | 0.910 | 0.80 | 0.72 | 0.93 (full and cut sentences) |
| final | random 5%, 452 rows | 0.863 | 0.74 | 0.63 | 0.99 (full sentences) |

`final` scores lower because its random validation set includes harder situations such as small talk
and coding. Most remaining errors are between cheer and surprise, empathy and surprise, and check and
understand, where people also disagree. On an Apple M5 CPU with one thread, one sentence takes 3 ms
with fp32 and 1.7 ms with int8.

## License

The code is under the MIT license ([LICENSE](LICENSE)).

The data is not under the MIT license. Its terms of use are in [data/LICENSE](data/LICENSE): it may
be used to train and evaluate classifiers and other models that do not generate open-ended text, and
not to train general-purpose chatbots, text-generation models or models that compete with
Anthropic's, nor as training targets for imitating Claude.
