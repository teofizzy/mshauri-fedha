"""
Access module for the fynesse framework.

This module handles data access functionality including:
- Data loading from various sources (web, local files, databases)
- Legal compliance (intellectual property, privacy rights)
- Ethical considerations for data usage
- Error handling for access issues

Legal and ethical considerations are paramount in data access.
Ensure compliance with e.g. .GDPR, intellectual property laws, and ethical guidelines.

Best Practice on Implementation
===============================

1. BASIC ERROR HANDLING:
   - Use try/except blocks to catch common errors
   - Provide helpful error messages for debugging
   - Log important events for troubleshooting

2. WHERE TO ADD ERROR HANDLING:
   - File not found errors when loading data
   - Network errors when downloading from web
   - Permission errors when accessing files
   - Data format errors when parsing files

3. SIMPLE LOGGING:
   - Use print() statements for basic logging
   - Log when operations start and complete
   - Log errors with context information
   - Log data summary information

4. EXAMPLE PATTERNS:
   
   Basic error handling:
   try:
       df = pd.read_csv('data.csv')
   except FileNotFoundError:
       print("Error: Could not find data.csv file")
       return None
   
   With logging:
   print("Loading data from data.csv...")
   try:
       df = pd.read_csv('data.csv')
       print(f"Successfully loaded {len(df)} rows of data")
       return df
   except FileNotFoundError:
       print("Error: Could not find data.csv file")
       return None
"""

from typing import Any, Union
import pandas as pd
import logging
import requests
from io import BytesIO
import time, re
from bs4 import BeautifulSoup
from urllib.parse import urljoin, urlparse
from urllib.robotparser import RobotFileParser
from collections import Counter
from tqdm.auto import tqdm

# Set up basic logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def data() -> Union[pd.DataFrame, None]:
    """
    Read the data from the web or local file, returning structured format such as a data frame.

    IMPLEMENTATION GUIDE
    ====================

    1. REPLACE THIS FUNCTION WITH YOUR ACTUAL DATA LOADING CODE:
       - Load data from your specific sources
       - Handle common errors (file not found, network issues)
       - Validate that data loaded correctly
       - Return the data in a useful format

    2. ADD ERROR HANDLING:
       - Use try/except blocks for file operations
       - Check if data is empty or corrupted
       - Provide helpful error messages

    3. ADD BASIC LOGGING:
       - Log when you start loading data
       - Log success with data summary
       - Log errors with context

    4. EXAMPLE IMPLEMENTATION:
       try:
           print("Loading data from data.csv...")
           df = pd.read_csv('data.csv')
           print(f"Successfully loaded {len(df)} rows, {len(df.columns)} columns")
           return df
       except FileNotFoundError:
           print("Error: data.csv file not found")
           return None
       except Exception as e:
           print(f"Error loading data: {e}")
           return None

    Returns:
        DataFrame or other structured data format
    """
    logger.info("Starting data access operation")

    try:
        # IMPLEMENTATION: Replace this with your actual data loading code
        # Example: Load data from a CSV file
        logger.info("Loading data from data.csv")
        df = pd.read_csv("data.csv")

        # Basic validation
        if df.empty:
            logger.warning("Loaded data is empty")
            return None

        logger.info(
            f"Successfully loaded data: {len(df)} rows, {len(df.columns)} columns"
        )
        return df

    except FileNotFoundError:
        logger.error("Data file not found: data.csv")
        print("Error: Could not find data.csv file. Please check the file path.")
        return None
    except Exception as e:
        logger.error(f"Unexpected error loading data: {e}")
        print(f"Error loading data: {e}")
        return None

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

    def fetch(self, url, timeout=25):
        """Fetch url with basic error handling. Returns (resp, soup) or (None,None)."""
        try:
            r = self.session.get(url, timeout=timeout)
            r.raise_for_status()
            soup = BeautifulSoup(r.text, "lxml")
            return r, soup
        except Exception as e:
            print(f"[fetch] Error fetching {url}: {e}")
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
            return []
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
