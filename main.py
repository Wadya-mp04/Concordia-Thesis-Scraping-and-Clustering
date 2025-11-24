import requests
from bs4 import BeautifulSoup as bs
import re
import PyPDF2
import nltk
from urllib.parse import urlparse, urljoin
import pickle
from io import BytesIO
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
import numpy as np
import time
import sys

# make sure tokenizer is available
nltk.download("punkt")

# robot.txt handling

BASE_URL = "https://spectrum.library.concordia.ca"
BASE_NETLOC = urlparse(BASE_URL).netloc
DISALLOWED_PATHS = []  # will contain all paths crawler is banned from opening


def makeSession() -> requests.Session:
    """
    Create a requests.Session with retry logic for unstable/SSL-flaky servers.
    """
    session = requests.Session()
    retries = Retry(
        total=5,                # retry up to 5 times
        backoff_factor=0.5,     # 0.5s, 1s, 2s, 4s, ...
        status_forcelist=[500, 502, 503, 504],
        allowed_methods=["GET"],
        raise_on_status=False,
    )
    adapter = HTTPAdapter(max_retries=retries)
    session.mount("https://", adapter)
    session.mount("http://", adapter)
    return session


def robotTxtChecker(baseURL: str):
    global DISALLOWED_PATHS

    robotURL = urljoin(baseURL, "/robots.txt")

    try:
        # robots.txt can use plain requests; if it fails, we assume no restrictions
        res = requests.get(robotURL, timeout=10)
    except requests.RequestException as e:
        print(f"[robots] Could not fetch robots.txt: {e}")
        DISALLOWED_PATHS = []
        return
    
    if res.status_code != 200:
        print(f"[robots] No robots.txt found (status {res.status_code}), assuming no restrictions.")
        DISALLOWED_PATHS = []
        return

    disallowedPaths = []
    current_ua = None
    for line in res.text.splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        if line.lower().startswith("user-agent:"):
            current_ua = line.split(":", 1)[1].strip()
        elif line.lower().startswith("disallow:") and (current_ua == "*" or current_ua is None):
            path = line.split(":", 1)[1].strip()
            if path:
                disallowedPaths.append(path)

    DISALLOWED_PATHS = disallowedPaths
    print(f"[robots] Disallow rules for *: {DISALLOWED_PATHS}")


def isAllowed(url: str) -> bool:
    """Check if url is inside Spectrum and not under any Disallow path."""
    parsed = urlparse(url)
    # stay inside the same host
    if parsed.netloc and parsed.netloc != BASE_NETLOC:
        return False

    path = parsed.path or "/"
    for dis in DISALLOWED_PATHS:
        if dis and path.startswith(dis):
            return False
    return True


def safeGet(url: str, session: requests.Session | None = None, **kwargs):
    """
    Wrapper around GET that:
    - respects robots.txt
    - uses a retry-enabled Session
    - handles SSL EOF errors more gracefully
    """
    if not isAllowed(url):
        print(f"[robots] Skipping disallowed URLs: {url}")
        return None

    if session is None:
        session = makeSession()

    try:
        return session.get(url, timeout=15, **kwargs)
    except requests.exceptions.SSLError as e:
        print(f"[ssl] SSL error fetching {url}: {e} — retrying with fresh session...")
        try:
            session = makeSession()
            return session.get(url, timeout=20, **kwargs)
        except Exception as e2:
            print(f"[ssl] Failed again for {url}: {e2}")
            return None
    except requests.RequestException as e:
        print(f"[request] Error fetching {url}: {e}")
        return None


# helper functions:
def containsLetterOrDigit(token: str) -> bool:
    return any(ch.isdigit() or ch.isalpha() for ch in token)


def hasFourDigits(s: str) -> bool:
    return re.fullmatch(r"^\d{4}$", s) is not None


def saveObject(obj, path: str):  # for saving inverted index and doc dict as .pkl files!
    with open(path, "wb") as f:
        pickle.dump(obj, f)
    print(f"[save] Saved object to {path}")


def loadObject(path: str):
    with open(path, "rb") as f:
        obj = pickle.load(f)
    print(f"[load] Loaded object from {path}")
    return obj


def getDocsForTerms(terms, index: dict[str, set[int]]):
    """Return a combined set of docIds containing ANY of the given terms."""
    docs = set()
    for t in terms:
        t = t.lower().strip()
        docs |= index.get(t, set())
    return docs


def getTextsForDocs(docIds, docTexts: dict[int, str]):
    """Return sorted doc IDs and the text of each document."""
    sortedIds = sorted(docIds)
    docs = []
    for docId in sortedIds:
        docs.append(docTexts.get(docId, ""))
    return sortedIds, docs


def runKMeans(X, k: int):
    """Run KMeans and return (model, labels, centers)."""
    model = KMeans(
        n_clusters=k,
        random_state=42,
        n_init="auto"
    )
    model.fit(X)
    labels = model.labels_
    centers = model.cluster_centers_
    return model, labels, centers


# ------------------- PDF discovery + indexing ------------------- #

def extractPDFs(url: str, totalPdfsDownloaded: list[str], upperbound: int) -> list[str]:
    """
    Given a thesis-year URL, find 'eprint' links and add them (as full URLs)
    to totalPdfsDownloaded until upperbound is reached.
    """
    res = safeGet(url)
    if res is None or res.status_code != 200:
        return totalPdfsDownloaded

    soup = bs(res.text, "html.parser")
    links = soup.find_all("a", href=True)

    for link in links:
        href = link["href"]
        if "eprint" in href and len(totalPdfsDownloaded) < upperbound:
            fullUrl = urljoin(BASE_URL, href)
            totalPdfsDownloaded.append(fullUrl)
            print(f"[crawler] Added eprint URL: {fullUrl}")

        if len(totalPdfsDownloaded) >= upperbound:
            break

    return totalPdfsDownloaded


def downloadPdf(pdfURL: str,
                invertedIndex: dict[str, set[int]],
                docId: int,
                docTexts: dict[int, str]) -> dict[str, set[int]]:

    session = makeSession()
    pdfURL = urljoin(BASE_URL, pdfURL)

    # open the eprint page
    pageRes = safeGet(pdfURL, session=session, allow_redirects=True)
    if pageRes is None or pageRes.status_code != 200:
        return invertedIndex

    soup = bs(pageRes.text, "html.parser")

    # meta robots tag check
    metaRobots = soup.find("meta", attrs={"name": "robots"})
    if metaRobots:
        content = (metaRobots.get("content") or "").lower()
        if "noindex" in content or "none" in content:
            print(f"[robots-meta] Skipping PDF page due to meta robots: {pdfURL}")
            return invertedIndex

    # find actual PDF link
    a = soup.find("a", class_="ep_document_link")
    if not a:
        print(f"[pdf] No ep_document_link found on {pdfURL}")
        return invertedIndex

    pdfDirectUrl = urljoin(BASE_URL, a["href"])

    # download the PDF
    pdfRes = safeGet(pdfDirectUrl, session=session, allow_redirects=True)
    if pdfRes is None or pdfRes.status_code != 200 or len(pdfRes.content) < 1000:
        print(f"[pdf] Skipping invalid or tiny PDF at: {pdfDirectUrl}")
        return invertedIndex

    # read PDF from memory
    pdfBytes = BytesIO(pdfRes.content)
    try:
        reader = PyPDF2.PdfReader(pdfBytes)
    except Exception as e:
        print(f"[pdf] Error reading PDF for {pdfURL}: {e}")
        return invertedIndex

    text = ""
    for pageNum, page in enumerate(reader.pages):
        try:
            pageText = page.extract_text() or ""
            text += pageText
        except Exception as e:
            print(f"[pdf] Skipping unreadable page {pageNum} in {pdfURL}: {e}")
            continue

    if text.strip() == "":
        print(f"[pdf] No extractable text in {pdfURL}, skipping this PDF.")
        return invertedIndex

    # tokenize and update inverted index + docTexts
    docTokens = []
    for token in nltk.word_tokenize(text):
        if not containsLetterOrDigit(token):
            continue
        if "//" in token or "v=" in token or "+" in token:
            continue
        token = token.lower().strip()
        if token == "":
            continue

        if token not in invertedIndex:
            invertedIndex[token] = set()
        invertedIndex[token].add(docId)
        docTokens.append(token)

    print(f"[pdf] finished tokenizing {pdfURL}")
    docTexts[docId] = " ".join(docTokens)
    return invertedIndex


# ------------------- Main crawler ------------------- #

def buildCrawler(upperbound=5):
    print("starting up crawler!")

    robotTxtChecker(BASE_URL)

    # Use BASE_URL as starting point
    url = BASE_URL
    res = safeGet(url)
    if res is None or res.status_code != 200:
        print("Could not open start URL")
        return

    soup = bs(res.text, "html.parser")
    links = soup.find_all("a", href=True)

    # find the 'browse' link
    for link in links:
        href = link["href"]
        if "browse" in href:
            url = urljoin(BASE_URL, href)
            # print(f"opening: {url}")
            break

    print(f"now opening URL {url}")
    res = safeGet(url)
    if res is None or res.status_code != 200:
        print("Could not open browse URL")
        return

    # find the 'doctype' page
    soup = bs(res.text, "html.parser")
    links = soup.find_all("a", href=True)
    for link in links:
        href = link["href"]
        if "doctype" in href:
            url = urljoin(url, href)
            # print(f"opening: {url}")
            break

    print(f"now opening URL {url}")
    res = safeGet(url)
    if res is None or res.status_code != 200:
        print("Could not open doctype URL")
        return

    # find the 'thesis/' page
    soup = bs(res.text, "html.parser")
    links = soup.find_all("a", href=True)
    for link in links:
        href = link["href"]
        if "thesis/" in href:
            url = urljoin(url, href)
            break

    print(f"now opening URL {url}")
    res = safeGet(url)
    if res is None or res.status_code != 200:
        print("Could not open thesis URL")
        return

    # collect year-specific thesis URLs (e.g., .../thesis/2020.html)
    soup = bs(res.text, "html.parser")
    links = soup.find_all("a", href=True)
    thesisYearsUrl = []
    for link in links:
        href = link["href"]
        href2 = href.replace(".html", "")
        if hasFourDigits(href2):
            yearUrl = urljoin(url, href)
            thesisYearsUrl.append(yearUrl)

    # extract eprint URLs up to upperbound
    totalPdfsDownloaded: list[str] = []
    for yearUrl in thesisYearsUrl:
        totalPdfsDownloaded = extractPDFs(yearUrl, totalPdfsDownloaded, upperbound)
        if len(totalPdfsDownloaded) >= upperbound:
            break

    print(f"[crawler] Collected {len(totalPdfsDownloaded)} eprint URLs.")

    # build inverted index / docTexts
    invertedIndex: dict[str, set[int]] = {}
    docTexts: dict[int, str] = {}
    docId = 0
    for pdfURL in totalPdfsDownloaded:
        invertedIndex = downloadPdf(pdfURL, invertedIndex, docId, docTexts)
        docId += 1

    # save to disk
    saveObject(invertedIndex, "index.pkl")
    saveObject(docTexts, "docTexts.pkl")

    print(f"[crawler] Indexed {len(docTexts)} documents.")
    return invertedIndex, docTexts


def buildCollections(index: dict[str, set[int]], docTexts: dict[int, str]):
    sustainabilityTerms = ["sustainability", "sustainable", "sustainably"]
    wasteTerms = ["waste", "wastes", "garbage", "trash", "recycling"]

    t1Docs = getDocsForTerms(sustainabilityTerms, index)
    t2Docs = getDocsForTerms(wasteTerms, index)

    bothDocs = t1Docs & t2Docs
    myCollection = t1Docs | t2Docs

    print(f"Docs for sustainability: {len(t1Docs)}")
    print(f"Docs for waste:         {len(t2Docs)}")
    print(f"Docs in both:           {len(bothDocs)}")
    print(f"My-collection size:     {len(myCollection)}")

    return t1Docs, t2Docs, bothDocs, myCollection


if __name__ == "__main__":

    upperbound = 50

    # If user passed an argument, override
    if len(sys.argv) > 1:
        try:
            upperbound = int(sys.argv[1])
        except ValueError:
            print("Invalid argument. Usage: python3 main.py 50")
            sys.exit(1)

    print(f"[main] Running crawler with upperbound = {upperbound}")
    
    # Crawl and index
    buildCrawler(upperbound=upperbound)
    
    # Load index + docTexts
    index = loadObject("index.pkl")
    docTexts = loadObject("docTexts.pkl")

    # Build collections based on sustainability/waste terms
    _, _, _, myCollection = buildCollections(index, docTexts)

    # Get texts for docs in My-collection
    docIds, docsForClustering = getTextsForDocs(myCollection, docTexts)

    if len(docsForClustering) == 0:
        print("No documents in My-collection, skipping clustering.")
    else:
        vectorizer = TfidfVectorizer(
            max_df=0.8,       # ignore very frequent terms
            min_df=1,         # keep terms that appear at least once
            max_features=5000 # cap vocab size
        )

        X = vectorizer.fit_transform(docsForClustering)
        print("TF-IDF matrix shape:", X.shape)  # (num_docs, num_terms)
        nDocs = X.shape[0]

        for k in [2, 10, 20]:
            print(f"\n=== K-Means with k = {k} ===")
            if nDocs < k:
                print(f"Skipping k={k} because we only have {nDocs} documents in My-collection.")
                continue

            model, labels, centers = runKMeans(X, k)
            # get top terms per cluster
            terms = vectorizer.get_feature_names_out()
            for clusterId in range(k):
                center = centers[clusterId]
                # indices of top 50 terms for this cluster
                topIndices = np.argsort(center)[::-1][:50]
                topTerms = [terms[i] for i in topIndices]
                print(f"\nCluster {clusterId}:")
                print(", ".join(topTerms))
