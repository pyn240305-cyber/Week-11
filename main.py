# NLP Comparison: NLTK vs spaCy
# Prepare 3 simple different sentences

import nltk
import spacy
import string
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# Download NLTK data
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')

# Load spaCy model
nlp = spacy.load("en_core_web_sm")

# Input Text (3 sentences)
texts = [
    "I love learning English every day.",
    "She is reading a new book now.",
    "They play football in the park."
]

print("===== INPUT TEXT =====")
for t in texts:
    print(t)

# -------------------------
# NLTK PREPROCESSING
# -------------------------
stop_words = set(stopwords.words("english"))
stemmer = PorterStemmer()

print("\n===== NLTK RESULT =====")
nltk_result = []

for text in texts:
    tokens = word_tokenize(text)                      # Tokenization
    tokens = [w.lower() for w in tokens]             # Lowercasing
    tokens = [w for w in tokens if w not in stop_words]
    tokens = [w for w in tokens if w not in string.punctuation]
    tokens = [stemmer.stem(w) for w in tokens]       # Stemming
    nltk_result.extend(tokens)
    print(tokens)

# -------------------------
# SPACY PREPROCESSING
# -------------------------
print("\n===== SPACY RESULT =====")
spacy_result = []

for text in texts:
    doc = nlp(text)
    tokens = []
    for token in doc:
        if not token.is_stop and not token.is_punct:
            tokens.append(token.lemma_.lower())      # Lemmatization
    spacy_result.extend(tokens)
    print(tokens)

# -------------------------
# GOLD DATA
# -------------------------
gold = [
    "love","learn","english","every","day",
    "read","new","book",
    "play","football","park"
]

# Convert to binary labels
all_words = list(set(gold + nltk_result + spacy_result))

gold_labels  = [1 if w in gold else 0 for w in all_words]
nltk_labels  = [1 if w in nltk_result else 0 for w in all_words]
spacy_labels = [1 if w in spacy_result else 0 for w in all_words]

# -------------------------
# EVALUATION FUNCTION
# -------------------------
def evaluate(name, y_true, y_pred):
    print(f"\n{name} Evaluation")
    print("Accuracy :", round(accuracy_score(y_true,y_pred),2))
    print("Precision:", round(precision_score(y_true,y_pred),2))
    print("Recall   :", round(recall_score(y_true,y_pred),2))
    print("F1-score :", round(f1_score(y_true,y_pred),2))

# Evaluate
evaluate("NLTK", gold_labels, nltk_labels)
evaluate("spaCy", gold_labels, spacy_labels)