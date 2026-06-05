import re
import spacy
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords

# ---------------- SAFE INIT ----------------
nltk.download('stopwords')
nltk.download('punkt')

try:
    nlp = spacy.load("en_core_web_sm")
except:
    import subprocess
    import sys
    subprocess.run([sys.executable, "-m", "spacy", "download", "en_core_web_sm"])
    nlp = spacy.load("en_core_web_sm")


def preprocess_text(text):

    text = str(text).lower()

    text = re.sub(r'https?://\S+|www\.\S+', ' ', text)
    text = re.sub(r'<.*?>', ' ', text)
    text = re.sub(r'[^a-zA-Z\s]', ' ', text)
    text = re.sub(r'\s+', ' ', text).strip()

    tokens = word_tokenize(text)

    stop_words = set(stopwords.words('english'))

    tokens = [w for w in tokens if w not in stop_words]

    doc = nlp(" ".join(tokens))

    tokens = [token.lemma_ for token in doc]

    return " ".join(tokens)
