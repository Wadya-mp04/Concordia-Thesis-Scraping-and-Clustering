# Concordia-Thesis-Scraping-and-Clustering

A full IR pipeline that scrapes Concordia Spectrum theses, extracts and tokenizes PDF text, constructs an inverted index, generates TF-IDF vectors, and performs unsupervised document clustering (K-Means) to reveal topic structure in the collection.

# COMP 479 – Project 2: Web Crawler, Indexer, and Clustering

This project implements:

1. A **polite web crawler** for Concordia’s Spectrum (`https://spectrum.library.concordia.ca`)
2. A **PDF text extractor + tokenizer**
3. An **inverted index** over a set of Spectrum theses
4. A **“My-collection”** of documents related to _sustainability_ and _waste_
5. A **TF–IDF representation** of documents in My-collection
6. **K-Means clustering** on My-collection, with top terms printed per cluster

Everything is driven from a single script: `main.py`.

---

## 1. Python Version and Environment Setup

This project was developed and tested with:

- **Python 3.10+** (e.g., 3.10 / 3.11 / 3.12)

It is strongly recommended to use a **virtual environment**:

```bash
python3 -m venv venv
source venv/bin/activate       # macOS / Linux
# .\venv\Scripts\activate      # Windows PowerShell

pip install --upgrade pip



## Required Python Packages

requests==2.31.0
https://pypi.org/project/requests/2.31.0/

beautifulsoup4==4.12.2
https://pypi.org/project/beautifulsoup4/4.12.2/

PyPDF2==3.0.1
https://pypi.org/project/PyPDF2/3.0.1/

nltk==3.8.1
https://pypi.org/project/nltk/3.8.1/

scikit-learn==1.4.0
https://pypi.org/project/scikit-learn/1.4.0/

urllib3==2.0.7
https://pypi.org/project/urllib3/2.0.7/
```
