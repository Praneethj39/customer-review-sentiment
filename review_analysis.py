"""
Customer Review Sentiment & Complaint Analysis
Dataset: Flipkart product reviews (phones, TVs, laptops, monitors, refrigerators,
         washing machines, power banks) - columns: Product_name, Review, Rating

Steps:
 1. Load and clean text (dedupe, fix joined words, strip symbols)
 2. Explore ratings, categories, review length
 3. Label sentiment: 4-5 stars = positive, 1-2 = negative, 3 = neutral (held out)
 4. Baseline: VADER, a rule-based sentiment lexicon (no training)
 5. ML models on TF-IDF features + decision-threshold tuning + a hybrid model
 6. Explain: words that push a review towards negative / positive
 7. Aspect analysis: which product aspects (battery, camera, service...) drive complaints
 8. Use the model on 3-star "neutral" reviews to flag hidden complaints
"""
import re
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.sparse import hstack, csr_matrix

import nltk
from nltk.sentiment import SentimentIntensityAnalyzer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_predict
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import (f1_score, precision_score, recall_score,
                             confusion_matrix, classification_report)

RANDOM_STATE = 42
plt.rcParams.update({"figure.dpi": 120, "axes.spines.top": False, "axes.spines.right": False})

try:
    nltk.data.find("sentiment/vader_lexicon.zip")
except LookupError:
    nltk.download("vader_lexicon", quiet=True)

# ---------------------------------------------------------------
# 1. Load and clean
# ---------------------------------------------------------------
df = pd.read_csv("flipkart_reviews.csv")
n_raw = len(df)
df = df.drop_duplicates(subset=["Review"]).reset_index(drop=True)
print(f"Raw rows: {n_raw}  ->  duplicates removed: {n_raw - len(df)}  ->  unique reviews: {len(df)}")

def clean(text):
    text = str(text)
    text = re.sub(r"([a-z])([A-Z])", r"\1 \2", text)   # 'performanceI got' -> 'performance I got'
    text = re.sub(r"[^A-Za-z0-9\s']", " ", text)        # drop emojis / symbols
    return re.sub(r"\s+", " ", text).strip().lower()

def category(name):
    n = name.lower()
    rules = [("washing", "Washing Machine"), (" kg ", "Washing Machine"), ("refrigerator", "Refrigerator"),
             ("power bank", "Power Bank"), ("laptop", "Laptop"), ("monitor", "Monitor"),
             ("tv", "TV"), ("television", "TV")]
    for key, cat in rules:
        if key in f" {n} ":
            return cat
    return "Phone"   # remaining products are mobiles (incl. iPhone, Lava, Samsung Guru)

df["text"] = df["Review"].apply(clean)
df["words"] = df["text"].str.split().str.len()
df["category"] = df["Product_name"].apply(category)

# ---------------------------------------------------------------
# 2. Explore
# ---------------------------------------------------------------
df["sentiment"] = np.select([df["Rating"] >= 4, df["Rating"] <= 2], ["positive", "negative"], "neutral")
print("\nRating distribution:\n", df["Rating"].value_counts().sort_index().to_string())
print("\nSentiment split:\n", df["sentiment"].value_counts(normalize=True).round(3).to_string())
print("\nMedian words by rating:\n", df.groupby("Rating")["words"].median().to_string())

cat = (df.groupby("category")
         .agg(reviews=("Rating", "size"), avg_rating=("Rating", "mean"),
              negative_share=("sentiment", lambda s: (s == "negative").mean()))
         .sort_values("negative_share", ascending=False))
print("\nBy category:\n", cat.round(3).to_string())

fig, ax = plt.subplots(figsize=(6, 4))
rc = df["Rating"].value_counts().sort_index()
ax.bar(rc.index.astype(str), rc.values, color=["#c0392b", "#e67e22", "#95a5a6", "#5d8aa8", "#2c3e50"])
ax.set_xlabel("Star rating"); ax.set_ylabel("Reviews"); ax.set_title("Rating distribution (unique reviews)")
fig.tight_layout(); fig.savefig("chart1_rating_distribution.png")

fig, ax = plt.subplots(figsize=(6, 4))
ax.barh(cat.index[::-1], cat["negative_share"][::-1] * 100, color="#2c3e50")
ax.set_xlabel("% of reviews rated 1-2 stars"); ax.set_title("Negative review share by product category")
fig.tight_layout(); fig.savefig("chart2_negative_share_by_category.png")

# ---------------------------------------------------------------
# 3. Labelled set (positive vs negative)
# ---------------------------------------------------------------
lab = df[df["sentiment"] != "neutral"].copy()
lab["y"] = (lab["sentiment"] == "negative").astype(int)   # 1 = negative, the class we want to catch
X_train, X_test, y_train, y_test = train_test_split(
    lab["text"], lab["y"], test_size=0.2, stratify=lab["y"], random_state=RANDOM_STATE)
print(f"\nLabelled reviews: {len(lab)} (negative {lab['y'].mean():.1%}) | "
      f"train {len(X_train)} / test {len(X_test)} ({y_test.sum()} negatives in test)")

results = []
def report(name, pred):
    results.append({"model": name,
                    "neg_precision": precision_score(y_test, pred),
                    "neg_recall": recall_score(y_test, pred),
                    "neg_f1": f1_score(y_test, pred),
                    "macro_f1": f1_score(y_test, pred, average="macro")})

# ---------------------------------------------------------------
# 4. Baseline: VADER lexicon (no training)
# ---------------------------------------------------------------
sia = SentimentIntensityAnalyzer()
def vader(texts):
    return np.array([sia.polarity_scores(t)["compound"] for t in texts]).reshape(-1, 1)

vader_test = vader(X_test)
report("VADER lexicon (no training)", (vader_test[:, 0] < 0.05).astype(int))

# ---------------------------------------------------------------
# 5. ML models on TF-IDF features
# ---------------------------------------------------------------
tfidf = TfidfVectorizer(ngram_range=(1, 2), min_df=2, sublinear_tf=True)
A_train = tfidf.fit_transform(X_train)          # fit vocabulary on training data only
A_test = tfidf.transform(X_test)
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)

def tune_threshold(model, A, y):
    """Pick the probability cut-off that maximises F1 on out-of-fold training predictions."""
    p = cross_val_predict(model, A, y, cv=cv, method="predict_proba")[:, 1]
    grid = np.arange(0.10, 0.90, 0.05)
    scores = [f1_score(y, p >= t) for t in grid]
    return grid[int(np.argmax(scores))], max(scores)

# 5a. Naive Bayes and Logistic Regression at the default 0.5 cut-off
nb = MultinomialNB(alpha=0.3).fit(A_train, y_train)
report("Naive Bayes (TF-IDF)", nb.predict(A_test))

lr = LogisticRegression(max_iter=3000, class_weight="balanced", C=10).fit(A_train, y_train)
report("Logistic Regression (TF-IDF, 0.5 cut-off)", lr.predict(A_test))

# 5b. Same model, cut-off tuned with cross-validation on the training set
t_lr, cv_lr = tune_threshold(LogisticRegression(max_iter=3000, class_weight="balanced", C=10), A_train, y_train)
report(f"Logistic Regression (TF-IDF, tuned cut-off {t_lr:.2f})", (lr.predict_proba(A_test)[:, 1] >= t_lr).astype(int))

# 5c. Hybrid: TF-IDF words + VADER score as one extra feature
H_train = hstack([A_train, csr_matrix(vader(X_train) * 3)]).tocsr()
H_test = hstack([A_test, csr_matrix(vader_test * 3)]).tocsr()
hyb = LogisticRegression(max_iter=3000, class_weight="balanced", C=10)
t_h, cv_h = tune_threshold(hyb, H_train, y_train)
hyb.fit(H_train, y_train)
hyb_pred = (hyb.predict_proba(H_test)[:, 1] >= t_h).astype(int)
report(f"Hybrid: TF-IDF + VADER (tuned cut-off {t_h:.2f})", hyb_pred)
print(f"CV F1 on training set - TF-IDF LR: {cv_lr:.3f}, Hybrid: {cv_h:.3f}")

res = pd.DataFrame(results).set_index("model")
print("\nTest-set results (catching negative reviews):\n", res.round(3).to_string())
print("\nHybrid confusion matrix [[TN FP],[FN TP]]:\n", confusion_matrix(y_test, hyb_pred))
print(classification_report(y_test, hyb_pred, target_names=["positive", "negative"]))

fig, ax = plt.subplots(figsize=(7.5, 4))
r = res["neg_f1"]
labels = ["VADER lexicon", "Naive Bayes", "LogReg (0.5)", "LogReg (tuned)", "Hybrid (tuned)"]
ax.barh(labels[::-1], r.values[::-1],
        color=["#2c3e50", "#5d8aa8", "#95a5a6", "#95a5a6", "#bdc3c7"])
for i, v in enumerate(r.values[::-1]):
    ax.text(v + 0.01, i, f"{v:.2f}", va="center")
ax.set_xlim(0, 1); ax.set_xlabel("F1 score on negative reviews (test set)")
ax.set_title("Catching negative reviews: lexicon vs ML")
fig.tight_layout(); fig.savefig("chart3_model_comparison.png")

# ---------------------------------------------------------------
# 6. Words that drive the prediction
# ---------------------------------------------------------------
coef = pd.Series(lr.coef_[0], index=tfidf.get_feature_names_out()).sort_values()
top_neg, top_pos = coef.tail(12)[::-1], coef.head(12)
print("Words pushing towards NEGATIVE:", ", ".join(top_neg.index))
print("Words pushing towards POSITIVE:", ", ".join(top_pos.index))

fig, axes = plt.subplots(1, 2, figsize=(9, 4.2))
axes[0].barh(top_neg.index[::-1], top_neg.values[::-1], color="#c0392b"); axes[0].set_title("Signals of a negative review")
axes[1].barh(top_pos.index[::-1], -top_pos.values[::-1], color="#2c3e50"); axes[1].set_title("Signals of a positive review")
for a in axes:
    a.set_xlabel("Model weight (absolute)")
fig.tight_layout(); fig.savefig("chart4_top_words.png")

# ---------------------------------------------------------------
# 7. Aspect analysis: what are people complaining about?
# ---------------------------------------------------------------
aspects = {
    "Battery / charging": r"\bbattery|\bcharg",
    "Camera": r"\bcamera",
    "Performance / speed": r"\bperformance|\bslow|\blag|\bhang|\bspeed",
    "Display / picture": r"\bdisplay|\bscreen|\bpicture",
    "Sound / noise": r"\bsound|\bnoise|\bnoisy|\bloud|\bspeaker",
    "Heating": r"\bheat",
    "Delivery / service": r"\bdeliver|\bservice|\binstall|\btechnician|\breturn|\breplace",
    "Price / value": r"\bprice|\bvalue|\bworth|\bcost",
}
base_rate = (df["sentiment"] == "negative").mean()
rows = []
for name, pat in aspects.items():
    m = df["text"].str.contains(pat, regex=True)
    rows.append({"aspect": name, "mentions": int(m.sum()),
                 "negative_rate_when_mentioned": df.loc[m, "sentiment"].eq("negative").mean(),
                 "share_of_all_negative_reviews": (m & df["sentiment"].eq("negative")).sum() / df["sentiment"].eq("negative").sum()})
asp = pd.DataFrame(rows).set_index("aspect").sort_values("negative_rate_when_mentioned", ascending=False)
asp["lift_vs_average"] = asp["negative_rate_when_mentioned"] / base_rate
print(f"\nAspect analysis (average negative rate {base_rate:.1%}):\n", asp.round(3).to_string())

fig, ax = plt.subplots(figsize=(7, 4.2))
ax.barh(asp.index[::-1], asp["negative_rate_when_mentioned"][::-1] * 100, color="#2c3e50")
ax.axvline(base_rate * 100, ls="--", color="#c0392b", lw=1)
ax.text(base_rate * 100 + 0.5, -0.6, f"average {base_rate:.0%}", color="#c0392b", fontsize=8)
ax.set_xlabel("% of reviews mentioning the aspect that are 1-2 stars")
ax.set_title("Which aspects show up in unhappy reviews?")
fig.tight_layout(); fig.savefig("chart5_aspect_complaints.png")

# ---------------------------------------------------------------
# 8. Hidden complaints inside 3-star reviews
# ---------------------------------------------------------------
neu = df[df["sentiment"] == "neutral"]
H_neu = hstack([tfidf.transform(neu["text"]), csr_matrix(vader(neu["text"]) * 3)]).tocsr()
flag = hyb.predict_proba(H_neu)[:, 1] >= t_h
print(f"\n3-star reviews flagged as complaints by the hybrid model: {flag.mean():.1%} of {len(neu)}")

res.round(3).to_csv("model_results.csv")
cat.round(3).to_csv("category_summary.csv")
asp.round(3).to_csv("aspect_summary.csv")
