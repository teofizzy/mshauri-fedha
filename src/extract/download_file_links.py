# Import dependencies
from typing import Any, Union, List, Dict
import time
import pandas as pd
import logging
import requests
from gnews import GNews
import feedparser
from io import BytesIO
import time, re
from bs4 import BeautifulSoup
import urllib3
import certifi
from urllib.parse import urljoin, urlparse
from urllib.robotparser import RobotFileParser
from collections import Counter
from tqdm.auto import tqdm
from newspaper import Article
from concurrent.futures import ThreadPoolExecutor, as_completed

urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

# Set up basic logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


class CBKExplorer:
    def __init__(self, github_username):
        self.user_agent = f"MshauriFedhaBot/0.1 (+https://github.com/{github_username}/mshaurifedha)"
        self.session = requests.Session()
        self.session.headers.update({"User-Agent": self.user_agent})

    def is_allowed_by_robots(self, base_url, target_url):
        """Check robots.txt for permission."""
        parsed = urlparse(base_url)
        robots_url = f"{parsed.scheme}://{parsed.netloc}/robots.txt"
        rp = RobotFileParser()
        try:
            rp.set_url(robots_url)
            rp.read()
            return rp.can_fetch(self.user_agent, target_url)
        except Exception as e:
            print(f"[robots] Could not read robots.txt ({e}). Proceed cautiously.")
            return False

    def fetch(self, url, timeout=25, allow_proxy_fallback=True):
        """
        Robust fetch that tries:
        1) requests with certifi bundle (secure)
        2) http fallback (if https fails)
        3) requests with verify=False (insecure)
        4) optional external proxy fetch (r.jina.ai) as last resort

        Returns (response, soup) or (None, None)
        """
        # helper to parse response->soup
        def resp_to_soup(r):
            try:
                r.raise_for_status()
                return r, BeautifulSoup(r.text, "lxml")
            except Exception:
                return None, None

        # 1) Try with certifi (preferred)
        try:
            r = self.session.get(url, timeout=timeout, verify=certifi.where())
            ok_resp, soup = resp_to_soup(r)
            if ok_resp:
                return ok_resp, soup
        except requests.exceptions.SSLError as ssl_err:
            print(f"[fetch] SSL error with certifi for {url}: {ssl_err}")
        except Exception as e:
            print(f"[fetch] Primary attempt failed for {url}: {e}")

        # 2) Try http fallback if URL is https
        try:
            parsed = urlparse(url)
            if parsed.scheme == "https":
                http_url = url.replace("https://", "http://", 1)
                try:
                    r = self.session.get(http_url, timeout=timeout)
                    ok_resp, soup = resp_to_soup(r)
                    if ok_resp:
                        print(f"[fetch] HTTP fallback succeeded for {http_url}")
                        return ok_resp, soup
                except Exception as e:
                    print(f"[fetch] HTTP fallback failed for {http_url}: {e}")
        except Exception as e:
            print(f"[fetch] HTTP fallback: error preparing URL: {e}")

        # 3) Try insecure (verify=False) as last direct option
        try:
            print(f"[fetch] Trying insecure fetch (verify=False) for {url} — not recommended for sensitive data.")
            r = self.session.get(url, timeout=timeout, verify=False)
            ok_resp, soup = resp_to_soup(r)
            if ok_resp:
                return ok_resp, soup
        except Exception as e:
            print(f"[fetch] Insecure fetch also failed for {url}: {e}")

        # 4) Optional: external proxy/relay (last resort)
        if allow_proxy_fallback:
            try:
                # Jina.ai simple fetch service: returns rendered HTML as text
                # NOTE: this is an external service — use only for public/cached pages.
                proxy_url = "https://r.jina.ai/http://" + url.replace("https://", "").replace("http://", "")
                print(f"[fetch] Trying proxy fetch via {proxy_url}")
                r = requests.get(proxy_url, timeout=30)  # using plain requests (no verify issues; it's https to jina)
                if r.status_code == 200 and r.text:
                    return r, BeautifulSoup(r.text, "lxml")
                else:
                    print(f"[fetch] Proxy fetch returned status {r.status_code}")
            except Exception as e:
                print(f"[fetch] Proxy fetch failed: {e}")

        # give up
        print(f"[fetch] All fetch strategies failed for {url}")
        return None, None

    def abs_link(self, base, href):
        """Make absolute link from relative href."""
        if not href:
            return None
        return urljoin(base, href)

    def explore_url(self, url, print_anchors=40):
        """Explore a CBK URL: meta, headings, nav links, anchor samples, file-like links."""
        print("URL:", url)
        print("Allowed by robots.py? ->", self.is_allowed_by_robots(url, url))
        resp, soup = self.fetch(url)
        if not resp:
            return None

        # Basic meta
        print("Status code:", resp.status_code)
        title = soup.title.string.strip() if soup.title else ""
        print("Title:", title)
        desc = ""
        meta_desc = soup.find("meta", attrs={"name":"description"}) or soup.find("meta", attrs={"property":"og:description"})
        if meta_desc and meta_desc.get("content"):
            desc = meta_desc["content"].strip()
            print("Meta description:", desc[:300])

        # Headings
        h1s = [h.get_text(strip=True) for h in soup.find_all("h1")]
        h2s = [h.get_text(strip=True) for h in soup.find_all("h2")]
        print("H1s:", h1s[:5])
        print("H2s:", h2s[:8])

        # Nav / header anchors
        navs = soup.find_all("nav")
        if navs:
            print(f"Found {len(navs)} <nav> block(s). Sample nav links:")
            nav_links = []
            for nav in navs:
                for a in nav.find_all("a", href=True):
                    nav_links.append((a.get_text(strip=True), self.abs_link(url, a["href"])))
            for t, link in nav_links[:20]:
                print(" -", t or "<no-text>", "->", link)
        else:
            print("No <nav> block found (or it's rendered by JS).")

        # Sample anchors across page
        anchors = []
        for a in soup.find_all("a", href=True):
            text = a.get_text(strip=True)
            href = a["href"].strip()
            anchors.append((text, self.abs_link(url, href)))
        anchors = [a for a in anchors if a[1] is not None]
        print(f"Total anchors on page: {len(anchors)}. Showing first {min(print_anchors,len(anchors))}:")
        for t, link in anchors[:print_anchors]:
            print(" *", (t[:60] or "<no-text>"), "->", link)

        # Class name frequencies
        classes = []
        for tag in soup.find_all(True):
            cls = tag.get("class")
            if cls:
                classes.extend(cls if isinstance(cls, list) else [cls])
        class_counts = Counter(classes)
        print("Top 15 classes used on page (class_name:count):")
        for k,v in class_counts.most_common(15):
            print("  ", k, ":", v)

        # Links that look like files
        file_like = []
        for text, link in anchors:
            if re.search(r"\.pdf$|\.xls$|\.xlsx$|\.csv$", link, re.IGNORECASE):
                file_like.append((text, link))
        print("File-like links found on page:", len(file_like))
        for t, l in file_like[:20]:
            print("  FILE:", (t[:80] or "<no-text>"), "->", l)

        return {"title": title, "anchors": anchors, "file_links": file_like, "class_counts": class_counts}

    def inspect_pages(self, urls):
        results = {}
        for u in urls:
            print("\n" + "="*80)
            print("Inspecting:", u)
            out = self.explore_url(u, print_anchors=80)
            results[u] = out
            time.sleep(1.0)  # polite pause
        return results

    def collect_file_links(self, url, allowed_exts=(".pdf", ".xls", ".xlsx", ".csv")):
        _, soup = self.fetch(url)
        if not soup:
            return pd.DataFrame()   # instead of returning []

        found = []
        for a in soup.find_all("a", href=True):
            href = a["href"].strip()
            ab = self.abs_link(url, href)
            if not ab:
                continue
            # only same domain (safety)
            if urlparse(ab).netloc.endswith("centralbank.go.ke") or urlparse(ab).netloc == "":
                if any(ab.lower().endswith(ext) for ext in allowed_exts):
                    found.append({"page":url, "text": a.get_text(strip=True), "file_url":ab})

        # dedupe
        seen = set()
        dedup = []
        for row in found:
            if row["file_url"] not in seen:
                dedup.append(row)
                seen.add(row["file_url"])

        df = pd.DataFrame(dedup)
        print(f"Found {len(df)} file links on {url}")
        return df


    def crawl_links_for_files(self, start_url, allowed_exts=(".pdf", ".xls", ".xlsx", ".csv"), max_pages=50):
        _, soup = self.fetch(start_url)
        if not soup:
            return []
        pages = []
        for a in soup.find_all("a", href=True):
            href = a["href"].strip()
            ab = self.abs_link(start_url, href)
            if not ab:
                continue
            # only same domain
            if urlparse(ab).netloc.endswith("centralbank.go.ke"):
                pages.append(ab)
        pages = list(dict.fromkeys(pages))[:max_pages]
        print(f"Will inspect {len(pages)} linked pages from {start_url}")
        results = []
        for p in tqdm(pages):
            df = self.collect_file_links(p, allowed_exts=allowed_exts)
            if not df.empty:
                results.append(df)
            time.sleep(0.8)
        if results:
            return pd.concat(results, ignore_index=True)
        return pd.DataFrame()

def download_files(file_links, root_dir, save_dir,
                   allowed_exts=(".pdf", ".xls", ".xlsx", ".csv"),
                   overwrite=False):
    """
    Download multiple files from a list of (title, url) pairs.

    Args:
        file_links: list of (title, url) tuples, or list of dicts {"text":..., "file_url":...}
        root_dir: base folder to save under
        save_dir: subdirectory under root_dir
        allowed_exts: file extensions to allow
        overwrite: if True, re-download even if file exists

    Returns:
        metadata: list of dicts (title, url, local_path, size, status)
    """
    save_dir_path = os.path.join(root_dir, save_dir)
    os.makedirs(save_dir_path, exist_ok=True)
    metadata = []

    # Normalize file_links into [(title, url), ...]
    norm_links = []
    for item in file_links:
        if isinstance(item, tuple):
            title, url = item
        elif isinstance(item, dict):
            title, url = item.get("text", "file"), item.get("file_url")
        else:
            continue
        norm_links.append((title.strip(), url.strip()))

    for title, url in norm_links:
        # filter by extension
        if not any(url.lower().endswith(ext) for ext in allowed_exts):
            continue

        # guess extension from URL
        ext = os.path.splitext(urlparse(url).path)[1] or ".bin"
        # clean filename
        safe_title = re.sub(r"[^A-Za-z0-9._-]+", "_", title)[:100]
        fname = f"{safe_title}{ext}"
        path = os.path.join(save_dir_path, fname)

        if os.path.exists(path) and not overwrite:
            print(f"[skip] {fname} already exists.")
            status = "skipped"
        else:
            try:
                print(f"[download] {title} -> {fname}")
                r = requests.get(url, stream=True, timeout=60)
                r.raise_for_status()
                with open(path, "wb") as f:
                    for chunk in r.iter_content(8192):
                        if chunk:
                            f.write(chunk)
                status = "ok"
            except Exception as e:
                print(f"[error] Failed: {url} ({e})")
                status = "error"

        size = os.path.getsize(path) if os.path.exists(path) else 0
        metadata.append({
            "title": title,
            "url": url,
            "local_path": path,
            "size": size,
            "status": status
        })

    return metadata

import os, subprocess, importlib, sys

def load_repo(repo):
    local = repo.split("/")[-1]
    if not os.path.exists(local):
        subprocess.run(["git", "clone", f"https://github.com/{repo}.git"], check=True)
    else:
        subprocess.run(["git", "-C", local, "pull"], check=True)
    if local not in sys.path:
        sys.path.insert(0, local)
    mod = importlib.import_module(local)
    importlib.reload(mod)
    return mod
    
def fetch_kenya_gnews(api_key):
    # Free tier: 100 requests/day
    url = f"https://gnews.io/api/v4/top-headlines?category=business&country=ke&token={api_key}"

    response = requests.get(url)
    data = response.json()

    articles = []
    for article in data.get('articles', []):
        articles.append({
            'title': article.get('title'),
            'content': article.get('description'),
            'url': article.get('url'),
            'date': article.get('publishedAt'),
            'source': article.get('source', {}).get('name')
        })

    df = pd.DataFrame(articles)
    return df
   
def is_valid_url(url):
    try:
        result = urlparse(url)
        return all([result.scheme, result.netloc])
    except:
        return False

def fetch_kenya_thenewsapi(api_key):

    url = f"https://api.thenewsapi.com/v1/news/all?api_token={api_key}&search=kenya+economy&language=en"

    response = requests.get(url)
    data = response.json()

    articles = []
    for article in data.get('data', []):
        articles.append({
            'title': article.get('title'),
            'content': article.get('description'),
            'url': article.get('url'),
            'date': article.get('published_at'),
            'source': article.get('source')
        })

    df = pd.DataFrame(articles)
    return df

def scrape_google_news_kenya():
    google_news = GNews(
        language='en',
        country='KE',
        period='7d',
        max_results=50
    )

    # Search for Kenya business news
    articles = google_news.get_news('Kenya economy OR inflation OR central bank')

    df = pd.DataFrame(articles)
    return df

# Install: pip install gnews

def scrape_african_business_rss():
    feeds = [
        'https://african.business/feed/',  # African Business Magazine
        'https://www.cnbcafrica.com/feed/',  # CNBC Africa
        'https://allafrica.com/tools/headlines/rdf/economy/headlines.rdf',  # AllAfrica Economy
    ]

    articles = []

    for feed_url in feeds:
        feed = feedparser.parse(feed_url)

        for entry in feed.entries[:20]:
            articles.append({
                'title': entry.get('title', ''),
                'url': entry.get('link', ''),
                'date': entry.get('published', ''),
                'summary': entry.get('summary', ''),
                'source': feed.feed.get('title', '')
            })

    df = pd.DataFrame(articles)
    return df

def scrape_article(url: str, metadata: dict) -> dict:
    """Scrape single article"""
    try:
        article = Article(url)
        article.download()
        article.parse()

        if len(article.text) > 200:
            return {
                'title': article.title,
                'full_content': article.text,
                'summary': metadata.get('summary', ''),
                'url': url,
                'date': metadata.get('date'),
                'source': metadata.get('source'),
                'authors': ', '.join(article.authors) if article.authors else '',
                'image': article.top_image,
                'word_count': len(article.text.split()),
                'status': 'success'
            }
        return None
    except Exception as e:
        return {'url': url, 'status': 'failed', 'error': str(e)}

def fetch_newsdata_multi(api_key: str) -> List[Dict]:
    """Multiple NewsData.io requests with pagination"""
    all_articles = []

    # Different queries to maximize coverage
    queries = [
        'kenya economy',
        'kenya inflation',
        'kenya central bank',
        'kenya business',
        'kenya finance'
    ]

    for query in queries:
        try:
            page = None
            for _ in range(3):  # Up to 3 pages per query
                params = {
                    'apikey': api_key,
                    'q': query,
                    'country': 'ke',
                    'language': 'en'
                }
                if page:
                    params['page'] = page

                response = requests.get('https://newsdata.io/api/1/latest', params=params, timeout=10)
                data = response.json()

                if data.get('status') != 'success':
                    break

                for item in data.get('results', []):
                    if is_valid_url(item.get('link')):
                        all_articles.append({
                            'url': item.get('link'),
                            'summary': item.get('description', ''),
                            'date': item.get('pubDate'),
                            'source': item.get('source_id')
                        })

                page = data.get('nextPage')
                if not page:
                    break

                time.sleep(1)
        except Exception as e:
            print(f"NewsData query '{query}': {e}")
            continue

    return all_articles

def fetch_gnews_multi(api_key: str) -> List[Dict]:
    """Multiple GNews requests"""
    all_articles = []

    # Different search terms
    searches = [
        'kenya economy',
        'kenya inflation',
        'kenya business',
        'nairobi stock exchange'
    ]

    for search in searches:
        try:
            params = {
                'apikey': api_key,
                'q': search,
                'country': 'ke',
                'lang': 'en',
                'max': 10  # Free tier max
            }

            response = requests.get('https://gnews.io/api/v4/search', params=params, timeout=10)
            data = response.json()

            for item in data.get('articles', []):
                if is_valid_url(item.get('url')):
                    all_articles.append({
                        'url': item.get('url'),
                        'summary': item.get('description', ''),
                        'date': item.get('publishedAt'),
                        'source': item.get('source', {}).get('name')
                    })

            time.sleep(1)
        except Exception as e:
            print(f"GNews search '{search}': {e}")
            continue

    return all_articles

def fetch_thenewsapi_multi(api_key: str) -> List[Dict]:
    """Multiple TheNewsAPI requests (only 3 articles per request!)"""
    all_articles = []

    # Multiple searches to compensate for 3-article limit
    searches = [
        'kenya economy',
        'kenya business',
        'kenya inflation',
        'kenya central bank',
        'kenya finance',
        'nairobi economy',
        'kenya investment',
        'kenya banking'
    ]

    for search in searches:
        try:
            params = {
                'api_token': api_key,
                'search': search,
                'language': 'en',
                'limit': 3  # Free tier limit
            }

            response = requests.get('https://api.thenewsapi.com/v1/news/all', params=params, timeout=10)
            data = response.json()

            for item in data.get('data', []):
                if is_valid_url(item.get('url')):
                    all_articles.append({
                        'url': item.get('url'),
                        'summary': item.get('description', ''),
                        'date': item.get('published_at'),
                        'source': item.get('source')
                    })

            time.sleep(1)
        except Exception as e:
            print(f"TheNewsAPI search '{search}': {e}")
            continue

    return all_articles

def scrape_kenya_news_maximum(
    newsdata_key: str = None,
    gnews_key: str = None,
    thenewsapi_key: str = None,
    max_workers: int = 8
) -> pd.DataFrame:
    """Get MAXIMUM articles from all sources"""

    print("🔍 Fetching maximum articles from all APIs...\n")

    all_articles = []

    # Fetch from all sources
    if newsdata_key:
        print("NewsData.io: ", end="", flush=True)
        articles = fetch_newsdata_multi(newsdata_key)
        all_articles.extend(articles)
        print(f"{len(articles)} URLs")

    if gnews_key:
        print("GNews.io: ", end="", flush=True)
        articles = fetch_gnews_multi(gnews_key)
        all_articles.extend(articles)
        print(f"{len(articles)} URLs")

    if thenewsapi_key:
        print("TheNewsAPI: ", end="", flush=True)
        articles = fetch_thenewsapi_multi(thenewsapi_key)
        all_articles.extend(articles)
        print(f"{len(articles)} URLs (limited to 3/request on free)")

    if not all_articles:
        print("\n No articles found")
        return pd.DataFrame()

    # Deduplicate by URL
    seen = set()
    unique = []
    for a in all_articles:
        if a['url'] not in seen:
            seen.add(a['url'])
            unique.append(a)

    print(f"\n Total unique URLs: {len(unique)}\n")

    # Parallel scraping
    results = []
    failed = 0

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(scrape_article, a['url'], a): a for a in unique}

        with tqdm(total=len(futures), desc="📄 Scraping", unit="article") as pbar:
            for future in as_completed(futures):
                result = future.result()

                if result and result.get('status') == 'success':
                    results.append(result)
                elif result:
                    failed += 1
                    if failed <= 3:  # Only show first 3 errors
                        print(f"\n {result['url'][:50]}... | {result['error']}")

                pbar.update(1)
                time.sleep(0.2)

    # Save
    if results:
        df = pd.DataFrame(results)
        df = df.drop('status', axis=1, errors='ignore')

        print(f"\n {len(results)} articles scraped | {failed} failed | {len(results)/(len(results)+failed)*100:.1f}% success")
        print(f" Avg: {df['word_count'].mean():.0f} words | {df['source'].nunique()} sources")
        return df

    return pd.DataFrame()
