import re
import tqdm
import requests
import pandas as pd
import concurrent.futures
from bs4 import BeautifulSoup
from wikidata.client import Client

client = Client()

def getWikiCode(url):
    return url.strip().split("/")[-1]

def getWikiLink(qid, lang='en'):
    try:
        entity = client.get(qid, load=True)
        sitelinks = entity.data.get('sitelinks', {})
        page_info = sitelinks.get(f'{lang}wiki')
        return page_info['url'] if page_info else None
    except Exception as e:
        print(f"ERROR retrieving Wikipedia link for {qid}: {e}")
        return None

def getParagraph(wikipedia_link, min_chars=200):
    try:
        response = requests.get(wikipedia_link, allow_redirects=True)
        response.raise_for_status()
        if response.is_redirect:
            print(f"WARNING: Redirecting... {response.status_code}")
        soup = BeautifulSoup(response.content, 'html.parser')

        content = soup.find('div', class_='mw-content-ltr mw-parser-output')
        if not content:
            return ""

        paragraph_text = ""
        for p in content.find_all('p'):
            text = p.get_text(separator=" ", strip=True)
            text = re.sub(r'\[\d+\]', '', text)
            paragraph_text += " " + text
            if len(paragraph_text) > min_chars:
                break

        return paragraph_text.strip()

    except requests.exceptions.RequestException as e:
        print(f"ERROR fetching paragraph from {wikipedia_link}: {e}")
        return ""
    except Exception as e:
        print(f"ERROR parsing content from {wikipedia_link}: {e}")
        return ""
    
def process_item(index, item, df, lang):
    try:
        qid = getWikiCode(item)
        link = getWikiLink(qid, lang)
        paragraph = getParagraph(link)

        if not link:
            print(f"WARNING: missing Wikipedia link for QID {qid}")
            return index, df['description'][df['item'] == item].values[0]

        if not paragraph:
            print(f"WARNING: empty or missing content for {link}")
            return index, df['description'][df['item'] == item].values[0]

        return index, paragraph

    except Exception as e:
        print(f"ERROR processing item {item} (QID: {qid if 'qid' in locals() else 'UNKNOWN'}): {e}")
        return index, df['description'][df['item'] == item].values[0]

def getText(df, lang='en', max_workers=10):
    results = [None] * len(df)
    items = list(enumerate(df['item']))

    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(process_item, idx, item, df, lang): idx for idx, item in items}

        for future in tqdm.tqdm(concurrent.futures.as_completed(futures), total=len(futures)):
            idx, paragraph = future.result()
            results[idx] = paragraph

    return results