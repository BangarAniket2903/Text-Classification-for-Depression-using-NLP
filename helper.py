import re
import spacy
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords

# ---------------- SAFE NLTK SETUP ----------------
nltk.download('stopwords')
nltk.download('punkt')
nltk.download('punkt_tab')

# ---------------- SAFE SPACY SETUP ----------------
try:
    nlp = spacy.load("en_core_web_sm")
except:
    import subprocess
    import sys
    subprocess.run([sys.executable, "-m", "spacy", "download", "en_core_web_sm"])
    nlp = spacy.load("en_core_web_sm")


def preprocess_text(text):

    text = str(text).lower()

    # Remove URLs
    text = re.sub(r'https?://\S+|www\.\S+', ' ', text)

    # Remove HTML tags
    text = re.sub(r'<.*?>', ' ', text)

    # Remove special characters, numbers
    text = re.sub(r'[^a-zA-Z\s]', ' ', text)

    # Remove extra spaces
    text = re.sub(r'\s+', ' ', text).strip()

    # Tokenization
    tokens = word_tokenize(text)

    # Stopwords removal
    stop_words = set(stopwords.words('english'))
    tokens = [w for w in tokens if w not in stop_words]

    # Lemmatization
    doc = nlp(" ".join(tokens))
    tokens = [token.lemma_ for token in doc]

    return " ".join(tokens)
