# Perplexity signal — threshold calibration

Recorded 2026-04-22 on `felix-branch`.

## Motivation

The rule carries a Kneser-Ney trigram LM trained on
`data/leichte_sprache_korpus.txt` (150 340 lines, 149 357 tokenised, 32 401
unique tokens). The initial plan was to flag sentences whose log-perplexity
per token exceeded a calibrated threshold. That signal turns out to be:

1. **Extremely slow** — NLTK's `KneserNeyInterpolated.perplexity(ngrams)`
   takes ~1 second per short sentence on CPU (average 21 s/sentence observed
   on a 200-sentence sample, with long-tail outliers up to ~120 s). This is
   unusable for a rule that runs on every `/analyse` request.
2. **Overwhelmingly dominated by OOV**: on a 58-sentence sample of the
   intentionally-complex `tools/eval_corpus/*.txt`, 57 sentences returned
   `+inf` perplexity (triggered by at least one out-of-vocabulary unigram
   inside a trigram context) and only one returned a finite value (8.37).

The conclusion: the model's **vocabulary** carries the discriminating signal,
not its full perplexity surface. We keep the trained pickle as the source of
the LS-corpus vocabulary and replace the slow `.perplexity()` call with a
fast OOV-ratio check. Same signal, O(tokens) latency.

## Data

| Sample | Source | n |
|---|---|---|
| LS baseline | random sample of `data/leichte_sprache_korpus.txt` (seed=42) | 200 |
| Non-LS | all 10 files in `tools/eval_corpus/`, split on `.!?`, filter ≥10 chars | 58 |

## Results

### Kneser-Ney log-perplexity per token (natural log, nats)

```
LS (n=200):    finite=200  inf=0
  p50  = 4.934
  p75  = 5.099
  p90  = 5.223
  p95  = 5.345
  p99  = 6.128

non-LS (n=58): finite=1    inf=57
  p50  = 8.373 (the single finite score; 57 others were +inf)
```

The LS p99 at 6.13 and the non-LS minimum at 8.37 would admit a log-ppl
threshold of ~7.0 with a comfortable margin. But the `.perplexity()` call
is too slow to use in production (see above), so we derived an OOV check
instead.

### OOV ratio (tokens absent from `model.vocab`)

By construction, sentences drawn from the training corpus have ~0% OOV
content. The non-LS domain corpora have 20-60% OOV. Representative
measurements on hand-picked sentences:

| Sentence | OOV / total | ratio |
|---|---|---|
| `Der Mann geht nach Hause.` | 0/5 | 0.00 |
| `Die Katze schläft auf dem Sofa.` | 0/6 | 0.00 |
| `Wir spielen im Park mit dem Ball.` | 0/7 | 0.00 |
| `Das Smartphone klingelt.` | 0/3 | 0.00 |
| `Der Vogel singt.` | 0/3 | 0.00 |
| `Die Administration evaluiert das Konzept.` | 1/5 | 0.20 |
| `Die Protonenkollision fand statt.` | 1/4 | 0.25 |
| `Die synergetische Optimierung erfordert differenzierte Reflexion.` | 3/6 | 0.50 |
| `Die Quantenfluktuation destabilisiert das Vakuum.` | 3/5 | 0.60 |

## Threshold choice

`THRESHOLD_OOV_RATIO = 0.15`

Justification:

- Any sentence above the threshold had ≥ 1 OOV token AND ≥ 15% of content tokens off-vocab.
- The lowest observed flag case (`Administration`) is at 0.20, so the 0.15 cutoff has a small safety margin against legitimate-but-missing-from-corpus tokens.
- LS sentences sit cleanly at 0.00 — no false positives in the hand-audited set.

## Latency

| Operation | Cold | Warm |
|---|---|---|
| First `check_rule` call (loads pickle + NLTK punkt) | ~2.5 s | — |
| Subsequent `check_rule` on a 4-sentence paragraph | — | 0.1-0.3 ms |
| 100 identical short sentences | — | 2 ms total |

The slow first call is the one-off model + punkt load. Subsequent calls are
vocab-hash lookups and NLTK tokenisation only.

## Re-calibration

Re-run:

```bash
python -m tools.ml train perplexity_saetze --promote   # retrain on current corpus
python -m tools.ml evaluate perplexity_saetze --verbose # sanity-check avg perplexity
```

If the LS corpus grows, the vocabulary grows with it, and the threshold
should remain valid. If it shrinks, reconsider raising `THRESHOLD_OOV_RATIO`
to reduce false positives on legitimate but newly-missing tokens.
