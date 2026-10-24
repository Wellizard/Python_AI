import csv
import io
import nltk
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer, WordNetLemmatizer
from textblob import TextBlob
import matplotlib.pyplot as plt
import pandas as pd

# Завантаження необхідних мовних ресурсів
nltk.download("punkt", quiet=True)
nltk.download("punkt_tab", quiet=True)
nltk.download("stopwords", quiet=True)
nltk.download("wordnet", quiet=True)
nltk.download("omw-1.4", quiet=True)

file_path = r"ШАГ\4.NLP\reviews.csv"

with open(file_path, "r", encoding="utf-8", errors="replace") as f:
    outer_rows = list(csv.reader(f))

if len(outer_rows) > 0 and len(outer_rows[0]) == 1:
    unwrapped_csv = "\n".join(row[0] for row in outer_rows if row)
    inner_rows = list(csv.reader(io.StringIO(unwrapped_csv)))
    df = pd.DataFrame(inner_rows[1:], columns=inner_rows[0])
else:
    df = pd.DataFrame(outer_rows[1:], columns=outer_rows[0])

df.columns = df.columns.astype(str).str.strip().str.strip('"')
print("Стовпці у таблиці:", df.columns.tolist())

target_col = "Feedback Comment"

df = df.dropna(subset=[target_col]).copy()
df[target_col] = df[target_col].astype(str)

stop_words = set(stopwords.words("english"))
stemmer = PorterStemmer()
lemmatizer = WordNetLemmatizer()

def preprocess_text(text):
    words = word_tokenize(text)
    filtered_words = [
        word for word in words if word.lower() not in stop_words and word.isalpha()
    ]
    lemmatized_words = [lemmatizer.lemmatize(word.lower()) for word in filtered_words]
    return " ".join(lemmatized_words)

df["cleaned_feedback"] = df[target_col].apply(preprocess_text)

first_review = df[target_col].iloc[0]
print("\n=== Приклад обробки першого відгуку ===")
print("Оригінал:", first_review)
print("Речення:", sent_tokenize(first_review))
print("Після лемматизації:", df["cleaned_feedback"].iloc[0])

def get_sentiment(text):
    polarity = TextBlob(text).sentiment.polarity
    if polarity > 0:
        return "Positive"
    elif polarity < 0:
        return "Negative"
    else:
        return "Neutral"

df["polarity"] = df[target_col].apply(lambda x: TextBlob(x).sentiment.polarity)
df["sentiment"] = df[target_col].apply(get_sentiment)

print("\n=== Розподіл відгуків за тональністю ===")
sentiment_counts = df["sentiment"].value_counts()
print(sentiment_counts)

categories = ["Positive", "Neutral", "Negative"]
counts = [sentiment_counts.get(cat, 0) for cat in categories]
colors = ["#4CAF50", "#9E9E9E", "#F44336"]

plt.figure(figsize=(8, 5))
bars = plt.bar(categories, counts, color=colors, edgecolor="black", width=0.6)

for bar in bars:
    height = bar.get_height()
    plt.text(
        bar.get_x() + bar.get_width() / 2.0,
        height,
        f"{int(height)}",
        ha="center",
        va="bottom",
        fontsize=11,
        fontweight="bold"
    )

plt.title("Sentiment Analysis of Customer Reviews", fontsize=14, fontweight="bold")
plt.xlabel("Sentiment Category", fontsize=12)
plt.ylabel("Number of Reviews", fontsize=12)
plt.ylim(0, max(counts) * 1.15 if max(counts) > 0 else 10)
plt.grid(axis="y", linestyle="--", alpha=0.5)
plt.tight_layout()

plt.show()