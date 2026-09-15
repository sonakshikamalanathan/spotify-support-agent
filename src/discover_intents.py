"""Unsupervised first pass at an intent taxonomy.

Clusters historical customer messages (TF-IDF -> SVD -> k-means) and writes a Markdown
report with top terms and examples per cluster. A human reads the report and writes the
final codebook (labels/codebook.json); clusters are a starting point, not the taxonomy.
"""
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import normalize

from config import DATA_PROCESSED, PAIRS_CSV, REPORTS_DIR, SEED

N_CLUSTERS = 24
SAMPLE_SIZE = 15000
EXAMPLES_PER_CLUSTER = 10


def main():
    pairs = pd.read_csv(PAIRS_CSV)
    history = pairs[pairs["split"] == "history"]
    sample = history.sample(min(SAMPLE_SIZE, len(history)), random_state=SEED).reset_index(drop=True)

    vectorizer = TfidfVectorizer(stop_words="english", ngram_range=(1, 2), min_df=5, max_df=0.5, sublinear_tf=True)
    tfidf = vectorizer.fit_transform(sample["customer_text"].str.replace("<url>", " ").str.replace("@user", " "))
    embedded = normalize(TruncatedSVD(n_components=100, random_state=SEED).fit_transform(tfidf))
    labels = KMeans(n_clusters=N_CLUSTERS, n_init=10, random_state=SEED).fit_predict(embedded)
    sample["cluster"] = labels

    terms = np.array(vectorizer.get_feature_names_out())
    lines = [f"# Intent discovery: {N_CLUSTERS} k-means clusters over {len(sample):,} historical messages\n"]
    for cluster_id, size in sample["cluster"].value_counts().items():
        members = sample[sample["cluster"] == cluster_id]
        centroid = np.asarray(tfidf[members.index].mean(axis=0)).ravel()
        top_terms = ", ".join(terms[centroid.argsort()[::-1][:12]])
        lines.append(f"## Cluster {cluster_id} - {size} msgs ({size / len(sample):.1%})")
        lines.append(f"**Top terms:** {top_terms}\n")
        for row in members.sample(min(EXAMPLES_PER_CLUSTER, len(members)), random_state=SEED).itertuples():
            lines.append(f"- C: {row.customer_text[:220]}\n  - B: {row.brand_reply[:220]}")
        lines.append("")

    # Cluster assignments are reused as weak labels for the simple baseline classifier.
    sample[["conv_id", "cluster"]].to_csv(DATA_PROCESSED / "history_clusters.csv", index=False)

    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    out = REPORTS_DIR / "intent_clusters.md"
    out.write_text("\n".join(lines), encoding="utf-8")
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()
