# Insights Summary — Model Results

This is the write-up of what came out of the analysis in the notebook
(`notebooks/individual_task1_analysis.ipynb`). Full code and cleaning
steps are there, this is more just pulling out the key takeaways.

## Results at a glance

| Dataset | Model | Accuracy | Precision | Recall | F1 | ROC-AUC |
|---|---|---|---|---|---|---|
| Walmart (store sales) | Random Forest | 0.758 | 0.510 | 0.821 | 0.629 | 0.858 |
| Walmart (store sales) | Neural Network | 0.814 | 0.669 | 0.509 | 0.578 | 0.852 |
| Online Retail II (customers) | Random Forest | 0.900 | 0.748 | 0.905 | 0.819 | 0.960 |
| Online Retail II (customers) | Neural Network | 0.907 | 0.879 | 0.728 | 0.796 | 0.951 |

## Insight 1: structural factors beat short-term ones, at both levels

Looking at what the Random Forest actually leans on (via feature
importance): for Walmart, store `Size` alone is responsible for roughly
half the model's predictive power, with CPI and Unemployment together
adding another ~27%. Promotional markdown activity barely shows up —
under 3% combined. On the Online Retail side, `Frequency` (~60%) and
`DistinctProducts` (~31%) dominate, with `Recency` sitting around 8%.

These are two completely different datasets — different country, different
kind of retail, different granularity (store-week vs. individual
customer) — and they're both telling a similar story: steady, structural
signals (how big a store is, how often and how broadly a customer buys)
predict commercial performance a lot better than short-term, transient
ones (a one-off promo, how recently someone last shopped). For a
Commercial Data Analyst role, that's a genuinely useful thing to flag —
reporting probably shouldn't lean too heavily on promo-level metrics if
structural signals are doing most of the work.

## Insight 2: the two models trade off precision and recall the same way on both datasets

On both datasets, Random Forest ends up with noticeably higher recall
than the Neural Network, while the Neural Network comes out ahead on
precision. ROC-AUC is fairly close between the two models within each
dataset, so they're roughly equally good overall at separating the
classes — the real difference is where each one sets its cutoff for
calling something "high".

Since this same pattern shows up on two datasets that otherwise have
nothing in common, it looks more like a general property of how these two
algorithms behave when the classes are imbalanced, rather than something
specific to either dataset. Practically, if missing a genuinely
high-performing store-week or high-value customer is worse than
occasionally over-flagging one — which seems reasonable for something
like inventory planning or customer retention — Random Forest's higher
recall makes it the more defensible pick of the two here.

## Why not just use accuracy

Both targets were built off a top-25% threshold, so both end up close to
a 75/25 split. A model that just always guessed "not high" would already
land around 75% accuracy without learning anything real. That's why
precision, recall, F1 and ROC-AUC are what's actually reported and
compared above, not accuracy on its own.
