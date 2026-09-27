# Methodology: Business Entity Resolution (Amazon ML Challenge 2026)

**Team:** TODO  **Final leaderboard macro F0.5:** 0.98046 (update if a later run scores higher)
**Train out-of-fold macro F0.5:** 0.98667 (3-fold by S1 entity, all 1,788,190 training S1 entities, singletons included)

## 1. Methodology overview

A large-scale entity resolution pipeline in two separately cached parts:

- **Data part:** normalisation → token and embedding candidate search → learned pre-filter →
  pair features. Everything is cached, so a model change never recomputes it.
- **Model part:** two-level gradient boosting, validated out-of-fold by S1 entity →
  one-to-one assignment → decision tuned for the exact metric.

The four stages:

1. **Normalisation** (country-agnostic):
   - Unicode → ASCII transliteration.
   - Legal-form and address-abbreviation tables that merge US, Indian and French conventions.
   - US state names mapped only when they are a whole address component.
   - Core name (legal words and noise removed), phonetic skeleton, house number, numbers,
     postal code, and splitting of "DBA / formerly" name variants.
   - Token synonyms learned from training pairs: Indic-script transliterations, state
     names in Indic scripts, city aliases. A mapping needs at least 15 occurrences and must
     look like a spelling variant, which rejects place swaps such as richmond→county.
2. **Candidate generation**, within each country block:
   - An IDF-weighted inverted index over hashed name/address tokens, name bigrams and
     numbers, searched both ways: each S2/S3 record's top 10 S1, and each S1's top 20 records.
   - Plus each S2/S3 record's 5 nearest S1 names by multilingual sentence-embedding cosine.
   - A cheap LightGBM pre-filter then keeps about 2.5 pairs per record.
3. **Matching model:**
   - Level 1: XGBoost (CUDA) on 136 pair features.
   - Cross-encoder: the multilingual MiniLM fine-tuned to read both records' raw
     "name | address" text together, scored out-of-fold (Section 4).
   - Level 2: XGBoost that also sees the level-1 out-of-fold probabilities aggregated over
     the record's competing S1 entities and over the S1 entity's other candidates, plus the
     cross-encoder probability and its context.
4. **Decision:**
   - Each S2/S3 record is kept only for its most probable S1 (ground truth is one-to-one
     from that side).
   - Then a probability threshold tuned for the exact macro F0.5 on out-of-fold
     predictions. Per-S1 rules were also compared (Section 5).

No external data, APIs or geocoders are used. Models used:

| Component | License | Size |
|---|---|---|
| XGBoost | Apache-2.0 | — |
| LightGBM | MIT | — |
| `paraphrase-multilingual-MiniLM-L12-v2`: sentence embedding of names, and the base of the fine-tuned cross-encoder (applied to the provided data only) | Apache-2.0 | ~118M parameters |

## 2. EDA findings that shaped the design

- **Scale.** Train has about 2.2M S1 entities and about 10.3M S2+S3 records. Test has 1.73M S1
  (India 46.8%, US 38.3%, France 15.0%) and about 10M S2+S3 records. This rules out all-pairs
  comparison, so blocking is exact but sparse, and every step is chunked and parallel.
- **One-to-one.** Each S2/S3 record belongs to at most one S1 entity, so the decision is
  one-to-one from the record side.
- **Matches per entity.** Most S1 entities have several matches (3.46 true matches per S1 on
  average), and only about 5.6% are singletons. An empty prediction therefore scores 0 for
  94% of entities.
- **Distractor density.** Train has fewer S2+S3 records per S1 than test (4.68 vs 5.75).
  Removing 19% of train S1 entities, whose records stay behind as unmatched distractors, gives
  train the same distractor density as test, so thresholds tuned on train transfer.
- **Indic scripts.** Many Indian records carry the name in an Indic script with a truncated
  address, which motivates learned transliteration synonyms and multilingual embeddings.
- **Unseen country.** France appears only in test. Country is used as an open-set block key
  and is never a model feature.

## 3. Candidate generation / blocking

| step | what | train result |
|---|---|---|
| token index | IDF-weighted overlap of hashed name/address tokens; per-token document-frequency cap = max(3000, 0.004 × block size); reverse top-10 + forward top-20 | pair recall 0.9809, 114.8M pairs |
| embedding neighbours | top-5 S1 names per S2/S3 record by MiniLM cosine (GPU) | recall alone 0.5328; union 0.9842 (+0.0033), 157.8M pairs |
| learned pre-filter | LightGBM on cheap scores and ranks; keeps each record's top pairs above a tuned cut-off | 25.4M pairs (2.46 per record), pair recall 0.9837 |

A perfect matcher on the scored pairs would reach macro F0.5 0.9946. `candidate_pairs.tsv`
contains exactly the pairs the model scores.

## 4. Model architecture and feature engineering

**Pair features (136).** All are country-agnostic.

- **Name strings:** rapidfuzz ratio, partial, token-sort, token-set, WRatio and Jaro-Winkler on
  the normalised, core and phonetic names; DBA best variant; legal-form conflict.
- **Script features:** the script of each side, script mismatch, and phonetic-skeleton
  prefix/length.
- **Character-level typo features:** character bag overlap, token anagram, sorted-token
  ratio, and initials equality.
- **IDF-weighted token overlap** for names and addresses: Jaccard, weighted Jaccard, and the
  rarest shared token. Rare tokens present on only one side are strong evidence of a
  different business.
- **Address and numbers:** house number (equal, Levenshtein, log gap), number Jaccard,
  postal code, and empty-address flags.
- **Frequency:** how often the core name appears among S1 and among S2+S3 (chains and
  generic names).
- **Context:** for key scores, the pair's rank, gap to the best, and best competing value
  among the record's candidate S1 entities and among the S1's candidate records; candidate
  counts.
- **Embeddings:** name-embedding cosine with the same rank, gap and best-other context.

**Model.**

- XGBoost on GPU: eta 0.08, loss-guided trees with 127 leaves, subsample and column sample
  0.7, λ = 10, early stopping.
- 3 folds by S1 entity (hash), giving out-of-fold probabilities for every training pair.
  Test is scored by the average of the fold models.
- Level 2 adds 13 aggregates of the level-1 out-of-fold probabilities: rank, gap and best
  other within the record and within the S1 entity, number of strong candidates, and the
  one-to-one winner flag. This lifts OOF macro F0.5 from 0.98314 to 0.98367 (AUC 0.99960 →
  0.99965).

**Cross-encoder (level-2 features).**

- **What it reads:** "S1 name | S1 address" [SEP] "S2/S3 name | S2/S3 address", raw text, with
  full attention between the two records. The country is not shown to it.
- **Model:** `paraphrase-multilingual-MiniLM-L12-v2` with a one-logit classification head.
  - Loss: binary cross-entropy. Optimiser: AdamW, lr 5e-5 with warm-up and linear decay.
  - Batch 256, bf16, maximum 96 tokens.
- **Training data:** each fold model is trained on 2M pairs of the other fold, 1 epoch.
  - Folds: 2, by S1 entity (hash), so every train pair gets an out-of-fold score.
  - Pairs: each record's best pre-filter pair plus every pair with pre-filter probability
    ≥ 0.05. That is 10.7M train pairs, covering 99.3% of the true pairs.
- **Test scoring:** 10.4M test pairs, scored by one fold model, so their distribution matches
  the out-of-fold scores.
- **Quality:** held-out AUC 0.99872 / 0.99871 for the cross-encoder alone.
- **Features passed to level 2:** its probability; its rank, best competing value and gap
  among the record's candidate S1 entities; and its best competing value among the S1's
  candidate records. Unscored pairs are left missing.
- **Level 1 unchanged:** the cross-encoder enters at level 2 only, so level 1 is untouched.
- **Result:**
  - OOF macro F0.5 0.98367 → **0.98667** (AUC 0.99975), India 0.97949 → 0.98353, US 0.98647 → 0.98876.
  - Leaderboard 0.972 → **0.98046**. The leaderboard gain is larger than the OOF gain, which
    suggests the pretrained multilingual model transfers to the unseen French data better
    than the hand-made string features.

## 5. Decision layer (optimising macro F0.5)

Per entity, F0.5 = 1.25·|P∩T| / (0.25·|T| + |P|). An empty P scores 1 only when T is empty.
Rules compared on out-of-fold probabilities with the exact metric:

| rule | OOF macro F0.5 |
|---|---|
| one-to-one + global threshold t = 0.72 | 0.98667 |
| one-to-one + two thresholds (S1's top pick at p ≥ 0.60, others at p ≥ 0.76) | 0.98676 |
| one-to-one + expected-F0.5 prefix per S1 (argmax_k 1.25·Σp_1..k / (0.25·Σp + k); empty if ∏(1−p) + 0.2 is larger) | 0.98676 |

(Without the cross-encoder the same comparison gave 0.98367 / 0.98384 / 0.98385.)

Per-country thresholds are adopted only when they beat the global threshold on that country
by more than 3·10⁻⁴; none did. France uses the global rule. Chosen rule: the global
threshold t = 0.72. The per-S1 rules gain less than 2·10⁻⁴ OOF, below the adoption bar.

## 6. Validation

- **OOF macro F0.5 by S1 group:** 0.98667 overall.
  - by group: singleton 0.9955 (99,930 S1) / one match 0.9542 (96,664) / several matches
    0.9881 (1,591,596)
  - by country: India 0.98353, US 0.98876
  - A perfect matcher on the scored pairs would reach 0.9946.
- **Unseen-country simulation**, a proxy for France. Train on one country, score the other,
  with equal training size:

  | target | trained on the target country | trained on the other country | loss |
  |---|---|---|---|
  | India | 0.97743 | 0.90008 (US-trained) | 7.7 points |
  | US | 0.98533 | 0.97231 (India-trained) | 1.3 points |

  The best threshold barely differed from the transferred one, so the loss comes from
  scoring, not calibration.
- **Error analysis** (out-of-fold, before the cross-encoder; lost points = 100 × (1 − macro F0.5)):
  - India: missing candidates 0.45, true pair below threshold 0.27, prediction with no true entity 0.13
  - US: missing candidates 0.22, true pair below threshold 0.46, prediction with no true entity 0.15

  Remaining hard cases:
  - Indic-script names with truncated addresses
  - typo'd names with empty addresses
  - look-alike S1 entities sharing a name or an address

## 7. Other relevant information

- **Reproducibility:** pinned `requirements.txt`, fixed hash-based folds and seeds.
  - The data part runs as `pipeline.py data`; the model part as `pipeline.py model`, which
    trains, tunes the decision and predicts.
  - Each step skips itself when its cached outputs are newer than its inputs.
- **Runtime:** measured on 32 CPU cores + one NVIDIA L4 (Modal).
  - Embedding neighbours (both splits): 1.2 h. Pre-filter: 17 min. Features: 29 min.
  - Model: 1.4 h to train and about 15 min to predict.
  - Cross-encoder (one NVIDIA H100): 2 × 23 min fine-tuning (≈1,460 pairs/s), 2 × 12 min
    out-of-fold scoring, 25 min test scoring (≈8,000 pairs/s).
  - Level 2 + decision + prediction with the cross-encoder: about 45 min (level 1 reused from
    its checkpoints).
  - Token candidate search and normalisation were cached from earlier runs and not re-timed.
- **Licenses:** XGBoost (Apache-2.0), LightGBM (MIT), scikit-learn (BSD-3), rapidfuzz (MIT),
  PyTorch (BSD), sentence-transformers and MiniLM (Apache-2.0). Unidecode (GPL-2) is used only
  as a preprocessing library, not a model.
- **Tried and not adopted:**
  - Pseudo-label self-training on test: the unseen-country simulation was mixed, so it was
    not submitted.
  - Per-S1 decision rules: +0.0001 to +0.0002 out-of-fold (Section 5), below the adoption bar.
