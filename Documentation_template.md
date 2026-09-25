# Methodology: Business Entity Resolution (Amazon ML Challenge 2026)

> Draft generated alongside the code. Fill the `TODO` numbers from
> `work/train_summary.json` and `reports/eda/eda_report.md` after the final run.
> If the official `Documentation_template.md` has different headings, paste
> these sections under them.

**Team:** TODO  **Final public LB F0.5:** TODO  **OOF (5-fold) macro F0.5:** TODO

## 1. Methodology overview

A classic ER pipeline with four stages, each tuned against the leaderboard metric
(macro F0.5 per Source-1 entity, singletons included):

1. **Normalisation**: country-agnostic text canonicalisation (transliteration to ASCII,
   legal-form and address-abbreviation tables covering US, India and France conventions,
   core-name extraction, phonetic skeleton, postal-code extraction, DBA splitting).
2. **Blocking / candidate generation**: union of six complementary blockers inside
   country blocks → `candidate_pairs.tsv` (exactly the pairs the model scores).
3. **Pairwise matching model**: LightGBM binary classifier on ~90 engineered features,
   trained with 5-fold GroupKFold (groups = S1 entity), final model = 3-seed bag.
4. **Decision layer**: probabilities → per-entity match set chosen to maximise
   macro F0.5 (threshold vs. expected-F0.5 rule, optional one-to-one constraint), tuned
   on out-of-fold predictions.

No external data, APIs, geocoders or pretrained entity databases are used. The only
models are LightGBM (MIT) and scikit-learn TF-IDF (BSD); no neural model is required.

## 2. EDA findings that shaped the design
(from `reports/eda/eda_report.md`)

- Dataset sizes: TODO. Singleton rate: TODO% (= score of an all-empty submission).
- Match-count distribution: TODO. Each S2/S3 id matches at most one S1 entity: TODO (yes/no), which motivates the one-to-one filter.
- Country agreement on matched pairs: TODO%, so country is used as a hard block key (open set; France is handled by the same code).
- Postal code agreement when both present: TODO%. Hard negatives with identical names (chains): TODO%, so address features are decisive.
- Name-only char-TF-IDF top-k recall: k=5 TODO, 10 TODO, 20 TODO, 50 TODO.

## 3. Candidate generation / blocking

All blockers run **within the country block** (country string lower-cased; an unseen label
simply forms its own block):

| id | blocker | k |
|----|---------|---|
| A | char 2–4-gram TF-IDF cosine on normalised name | 25 |
| B | char 2–4-gram TF-IDF on core name + normalised address | 25 |
| C | word TF-IDF on normalised address (renamed / DBA businesses) | 10 |
| D | reverse: each S2/S3 record → its top-5 S1 names (crowded neighbourhoods) | 5 |
| E | exact phonetic core-name key (transliteration variants) | block ≤ 30 |
| F | exact (postal code, first core-name token) | block ≤ 30 |

TF-IDF vocabularies are fitted on all records of the split (no labels). Top-k search
is an exact chunked sparse-dense product (no approximate index needed at this scale).

Train blocking quality: pair recall TODO, avg candidates per S1 TODO, reduction ratio TODO.
Per-blocker recall and unique contribution are logged by `pipeline.py train`.

## 4. Model architecture and feature engineering

**Features (≈90)**, all country-agnostic (the country label is never a feature):

- *Name strings*: rapidfuzz ratio / partial / token-sort / token-set / WRatio / Jaro-Winkler
  on the normalised name and on the core name (legal forms removed), normalised Levenshtein,
  phonetic-skeleton ratios, exact core/phonetic/no-space equality, acronym match,
  first-token equality, length and token-count differences, best score across DBA variants,
  legal-form conflict.
- *TF-IDF cosines*: char name, word name, char name+address, word address.
- *IDF-weighted token overlap*, for names and addresses: Jaccard, IDF-weighted Jaccard,
  max IDF of a shared token, and the total and max IDF of tokens found on only one side.
  A rare token that appears on one side only is strong evidence of a different business.
- *Address*: fuzzy scores, empty flags, length ratio, postal code both-present / equal /
  3-digit-prefix equal, house-number Jaccard / conflict, number agreement inside names.
- *Name frequency*: log-count of the core name in S1 and in S2+S3 (chains, generic names).
- *Context features* (key for precision): for 5 key scores, rank and gap-to-best of
  the pair among the S1 entity's candidates, rank and gap among the candidate record's
  competing S1 entities, second-best score in the S1 group, and candidate-set sizes.
- Source flag (S2 vs S3).

**Model**: LightGBM (`binary`, lr 0.03, 63 leaves, feature/bagging fraction 0.7/0.8,
L2 = 5), early-stopped per fold, then refit on all pairs with 1.1× the mean best iteration,
averaged over 3 seeds.

## 5. Decision layer (optimising macro F0.5)

For one entity with truth set T and prediction P, F0.5 = 1.25·|P∩T| / (0.25·|T| + |P|),
and an empty prediction scores 1 exactly when T is empty. Two rules are compared on OOF:

- **Threshold** t (grid then refinement).
- **Expected-F0.5**: sort the candidates by p, then for every prefix size k (including k = 0,
  worth ∏(1−p)) estimate E[F0.5] by Monte Carlo under independent Bernoulli(p) labels and
  pick the argmax, with an optional conservativeness bias.

Both can run with a **one-to-one filter** (each S2/S3 record is kept only for its most
probable S1 entity) when the EDA shows that ground truth is one-to-one. Chosen rule: TODO.

## 6. Validation

- 5-fold GroupKFold by S1 entity; OOF macro F0.5 over **all** train S1 entities
  (including those with no candidates): TODO. Breakdown: singleton TODO / one-match TODO / multi TODO.
- Blocking ceiling (a perfect matcher restricted to our candidates): TODO.
- **Leave-one-country-out** (train on US and score India, and the reverse) as a proxy for
  the unseen French test data: TODO / TODO.

## 7. Other relevant information

- Reproducibility: fixed seeds, pinned `requirements.txt`, single command
  (`pipeline.py all`), CPU only; runtime ≈ TODO min on 4 cores.
- License compliance: LightGBM (MIT), scikit-learn (BSD-3), rapidfuzz (MIT), Unidecode (GPL-2,
  preprocessing library only, not a model). TODO: swap it for `anyascii` (ISC) if organisers object.
- Things tried and not adopted: TODO.
