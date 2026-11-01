# Customer Review Sentiment & Complaint Analysis

Automatically flagging unhappy customer reviews and finding what customers complain about, using NLP on Flipkart product reviews.

**Data:** 2,304 Flipkart reviews (phones, TVs, laptops, monitors, refrigerators, washing machines, power banks) with star ratings. 946 were exact duplicates, leaving **1,358 unique reviews**.
**Tools:** Python, pandas, NLTK (VADER), scikit-learn, matplotlib

## Business question
A company receives thousands of free-text reviews and complaints. Can we automatically flag the unhappy ones for follow-up, and see which product aspects drive the complaints?

## Approach
1. **Cleaning** — removed 946 duplicate reviews (41% of raw rows), split words that were stuck together (`performanceI` → `performance I`), stripped emojis and symbols, lowercased.
2. **Labels** — 4–5 stars = positive, 1–2 = negative (8.8% of reviews), 3 stars = neutral, held out.
3. **Baseline** — VADER, a rule-based sentiment lexicon that needs no training.
4. **ML models** — TF-IDF (unigrams + bigrams) with Naive Bayes and Logistic Regression. Stop words were *kept* on purpose so phrases like "not good" and "don't buy" survive.
5. **Threshold tuning** — negatives are rare (9%), so the default 0.5 cut-off misses many. The cut-off was chosen by 5-fold cross-validation on the training set only.
6. **Hybrid model** — TF-IDF features plus the VADER score as one extra feature.
7. **Aspect analysis** — keyword rules for battery, camera, performance, display, sound, heating, delivery/service and price.

## Results (test set: 259 reviews, 24 negative)
| Model | Precision | Recall | F1 (negative) |
|---|---|---|---|
| VADER lexicon (no training) | 0.73 | 0.79 | 0.76 |
| Naive Bayes (TF-IDF) | 1.00 | 0.21 | 0.34 |
| Logistic Regression, 0.5 cut-off | 0.92 | 0.46 | 0.61 |
| Logistic Regression, tuned cut-off | 0.81 | 0.71 | 0.76 |
| **Hybrid TF-IDF + VADER, tuned cut-off** | **0.86** | **0.75** | **0.80** |

Key lessons:
- On a small, imbalanced dataset, a plain ML model at the default cut-off did **worse** than a no-training lexicon.
- Tuning the decision threshold closed the gap, and combining the two approaches beat both.
- The test set has only 24 negative reviews, so these scores would move with more data.

## What customers complain about
| Category | Negative share (1–2 stars) |
|---|---|
| Phones | 13.3% |
| Washing machines | 12.0% |
| TVs | 6.4% |
| Monitors | 6.0% |
| Laptops | 3.3% |
| Refrigerators | 2.7% |

- **Camera** (in 21% of all negative reviews) and **performance/speed** (19%) are the most common complaint themes, and reviews mentioning them are more likely than average to be negative.
- Reviews mentioning **price/value** are mostly positive (only 4% negative), so price is a strength, not a complaint.
- The hybrid model flagged **24% of 3-star reviews** as hidden complaints: customers who gave a middling rating but wrote negatively.

## Recommendations
1. Use the hybrid model to auto-route flagged reviews to a support queue instead of reading everything manually.
2. Treat 3-star reviews the model flags as complaints, since star ratings alone undercount unhappy customers.
3. Prioritise camera and performance issues in phone listings and seller feedback.

## Files
- `review_analysis.py` — full pipeline
- `chart1`–`chart5` — rating distribution, category negative share, model comparison, top words, aspect complaints
- `model_results.csv`, `category_summary.csv`, `aspect_summary.csv` — outputs

Data: public Flipkart product reviews dataset.
