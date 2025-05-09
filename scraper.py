import requests
from bs4 import BeautifulSoup
import newspaper
from newspaper import Article
import nltk
from urllib.parse import urlparse, urljoin, quote_plus
import re
from datetime import datetime
import time
import random
import json

# Download necessary NLTK data on first run
try:
    nltk.data.find('punkt')
except LookupError:
    nltk.download('punkt')

# Updated News source configurations
NEWS_SOURCES = {
    "Google News": {
        "search_url": "https://news.google.com/search?q={query}&hl=en-US&gl=US&ceid=US:en",
        "article_selector": "a.VDXfz",
        "base_url": "https://news.google.com"
    },
    "BBC": {
        "search_url": "https://www.bbc.co.uk/search?q={query}",
        "article_selector": ".ssrcss-1tf7rdf-PromoLink, .ssrcss-rn1ljk-PromoLink, .ssrcss-1pv3r3s-PromoLink, .ssrcss-oeofyu-PromoLink",
        "base_url": "https://www.bbc.co.uk"
    },
    "Reuters": {
        "search_url": "https://www.reuters.com/search/news?blob={query}",
        "article_selector": ".search-result__headline a, .media-story-card__headline__eqhp9 a, .text__text__1FZLe",
        "base_url": "https://www.reuters.com"
    },
    "NDTV": {
        "search_url": "https://www.ndtv.com/search?searchtext={query}",
        "article_selector": ".news_Itm a, .src_itm-ttl a, .item-title a",
        "base_url": "https://www.ndtv.com"
    },
    "CNN": {
        "search_url": "https://www.cnn.com/search?q={query}",
        "article_selector": ".cnn-search__result-headline a, .container__headline, .container_lead-plus-headlines__headline",
        "base_url": "https://www.cnn.com"
    },
    "Al Jazeera": {
        "search_url": "https://www.aljazeera.com/search/{query}",
        "article_selector": ".gc__title a, .article-card__title a",
        "base_url": "https://www.aljazeera.com"
    }
}

def get_headers():
    """Return request headers to mimic a browser."""
    user_agents = [
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36",
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/98.0.4758.102 Safari/537.36",
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/100.0.4896.75 Safari/537.36",
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:99.0) Gecko/20100101 Firefox/99.0"
    ]
    
    return {
        "User-Agent": random.choice(user_agents),
        "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8",
        "Accept-Language": "en-US,en;q=0.5",
        "Connection": "keep-alive",
        "Upgrade-Insecure-Requests": "1"
    }

def clean_text(text):
    """Clean and normalize text."""
    if not text:
        return ""
    # Remove excess whitespace
    text = re.sub(r'\s+', ' ', text).strip()
    # Remove special characters
    text = re.sub(r'[^\w\s.,?!-]', '', text)
    return text

def extract_keywords(text, n=5):
    """Extract important keywords from text."""
    # Very common words to filter out
    common_words = {'the', 'a', 'an', 'and', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'is', 'are', 'that', 
                   'was', 'were', 'have', 'has', 'had', 'not', 'news', 'from', 'after', 'before', 'during', 'while',
                   'said', 'says', 'will', 'would', 'could', 'should', 'been', 'being', 'their', 'they', 'them',
                   'about', 'over', 'under', 'who', 'what', 'when', 'where', 'which', 'there', 'here'}
    
    # Split text into words and normalize
    text = text.lower()
    words = re.sub(r'[^\w\s]', ' ', text).split()
    
    # First prioritize important words like names, places and important nouns
    priority_words = ['killed', 'attack', 'military', 'soldier', 'civilian', 'border', 'died', 'terrorist', 
                     'president', 'minister', 'government', 'official', 'leader', 'forces', 'country', 'army',
                     'casualties', 'strike', 'bombing', 'gunfire', 'shooter', 'troops', 'violence']
    
    # Find words that match our priority list
    priority_matches = [word for word in words if word in priority_words]
    
    # Then filter regular words
    filtered_words = [word for word in words if word not in common_words and len(word) > 3]
    
    # Count frequency
    word_freq = {}
    # First add priority words with higher weight
    for word in priority_matches:
        word_freq[word] = word_freq.get(word, 0) + 3
        
    # Then add other words
    for word in filtered_words:
        word_freq[word] = word_freq.get(word, 0) + 1
    
    # Sort by frequency and return top n
    sorted_words = sorted(word_freq.items(), key=lambda x: x[1], reverse=True)
    keywords = [word for word, _ in sorted_words[:n]]
    
    # Make sure we include important nouns and entities even if they're short
    # These are particularly important for news articles about specific events
    important_short_words = ['war', 'hit', 'UN', 'EU', 'US', 'UK', 'IDF', 'Iran', 'Iraq', 'Gaza', 'ISIS', 
                           'NATO', 'bomb', 'gun', 'kill', 'dead', 'die', 'shot', 'fire']
    
    # Extract short important words from text
    short_matches = [word for word in words if word in important_short_words]
    
    # Add short but important words if we don't have enough keywords
    if len(keywords) < n and short_matches:
        keywords.extend(short_matches[:n-len(keywords)])
    
    # If we still need more keywords, try with shorter words
    if len(keywords) < min(n, 3):
        shorter_words = [word for word in words if word not in common_words and len(word) > 2]
        word_freq = {}
        for word in shorter_words:
            word_freq[word] = word_freq.get(word, 0) + 1
        
        sorted_words = sorted(word_freq.items(), key=lambda x: x[1], reverse=True)
        additional_keywords = [word for word, _ in sorted_words[:n-len(keywords)]]
        keywords.extend(additional_keywords)
    
    # Return unique keywords
    return list(dict.fromkeys(keywords))

def generate_search_terms(query):
    """Generate multiple search terms from a query to improve results."""
    # First, try using the query directly
    search_terms = [query]
    
    # Then extract keywords
    keywords = extract_keywords(query, n=6)
    
    # If we have enough keywords, create combination searches
    if len(keywords) >= 3:
        # Try different combinations of keywords
        search_terms.append(' '.join(keywords[:4]))
        if len(keywords) > 4:
            search_terms.append(' '.join(keywords[2:]))
    
    # Add the original query as a fallback
    if keywords:
        search_terms.append(' '.join(keywords))
    
    # Make sure search terms are unique
    unique_terms = []
    for term in search_terms:
        if term and term not in unique_terms:
            unique_terms.append(term)
    
    return unique_terms

def search_google_news(query, max_results=5):
    """Special handler for Google News which has unique URL structure."""
    source_config = NEWS_SOURCES["Google News"]
    search_url = source_config["search_url"].format(query=quote_plus(query))
    
    print(f"Searching Google News at URL: {search_url}")
    
    try:
        response = requests.get(search_url, headers=get_headers(), timeout=15)
        if response.status_code != 200:
            print(f"Error: Status code {response.status_code} for Google News")
            return []
            
        soup = BeautifulSoup(response.text, 'html.parser')
        article_links = soup.select(source_config["article_selector"])
        
        print(f"Found {len(article_links)} potential articles from Google News")
        
        articles = []
        for link in article_links[:max_results*2]:
            if len(articles) >= max_results:
                break
                
            href = link.get('href')
            if not href:
                continue
                
            # Google News has a special URL structure
            if href.startswith('./'):
                href = href[2:]  # Remove the ./ prefix
                article_url = urljoin(source_config["base_url"], href)
                
                # Extract the actual article URL from Google's redirect
                try:
                    article_text = link.text
                    source_text = ""
                    source_elem = link.parent.parent.select_one(".KbnJ8")
                    if source_elem:
                        source_text = source_elem.text
                        
                    time_elem = link.parent.parent.select_one("time")
                    pub_time = ""
                    if time_elem:
                        pub_time = time_elem.get('datetime', '')
                    
                    # Get the actual URL by following the Google redirect
                    redirect_response = requests.get(article_url, headers=get_headers(), allow_redirects=False)
                    if redirect_response.status_code == 302:
                        actual_url = redirect_response.headers.get('Location')
                    else:
                        # Try to extract the URL from the page
                        actual_url = article_url
                        
                    # Skip if we couldn't get the real URL
                    if not actual_url or "accounts.google.com" in actual_url:
                        continue
                        
                    # Create a simplified article object
                    article_data = {
                        'title': article_text,
                        'content': f"{article_text}\n\nSource: {source_text}\n\nPublished: {pub_time}",
                        'summary': article_text,
                        'url': actual_url,
                        'source': f"Google News - {source_text}" if source_text else "Google News",
                        'keywords': extract_keywords(article_text),
                        'date': datetime.now().strftime("%Y-%m-%d") if not pub_time else pub_time[:10]
                    }
                    
                    articles.append(article_data)
                    print(f"Added Google News article: {article_data['title']}")
                except Exception as e:
                    print(f"Error parsing Google News article {article_url}: {e}")
                    continue
        
        return articles
    except Exception as e:
        print(f"Error searching Google News: {e}")
        return []

def search_source(query, source_name, max_results=3):
    """Search a news source for articles related to the query."""
    if source_name not in NEWS_SOURCES:
        print(f"Source {source_name} not found in configured sources")
        return []
        
    # Special handling for Google News
    if source_name == "Google News":
        return search_google_news(query, max_results)
    
    source_config = NEWS_SOURCES[source_name]
    all_articles = []
    
    # Generate multiple search terms to try
    search_terms = generate_search_terms(query)
    print(f"Generated search terms: {search_terms}")
    
    for search_query in search_terms:
        if search_query and len(all_articles) < max_results:
            # Format the search URL with the query
            search_url = source_config["search_url"].format(query=quote_plus(search_query))
            
            print(f"Searching {source_name} at URL: {search_url}")
            
            try:
                # Add a small delay to avoid rate limiting
                time.sleep(random.uniform(0.5, 1.5))
                
                response = requests.get(search_url, headers=get_headers(), timeout=15)
                if response.status_code != 200:
                    print(f"Error: Status code {response.status_code} for {source_name}")
                    continue
                
                soup = BeautifulSoup(response.text, 'html.parser')
                
                # Try multiple selectors
                article_links = []
                selectors = source_config["article_selector"].split(", ")
                
                for selector in selectors:
                    links = soup.select(selector)
                    if links:
                        article_links.extend(links)
                
                if not article_links:
                    # Backup approach - look for any a tags with titles
                    print(f"No article links found for {source_name} using selectors. Trying backup approach.")
                    potential_links = soup.find_all('a', href=True)
                    article_links = [link for link in potential_links if link.text and len(link.text.strip()) > 20]
                
                print(f"Found {len(article_links)} potential articles from {source_name}")
                
                # Process found articles
                for link in article_links:
                    if len(all_articles) >= max_results:
                        break
                        
                    href = link.get('href')
                    if not href:
                        continue
                    
                    # Handle relative URLs
                    if href.startswith('/'):
                        article_url = urljoin(source_config["base_url"], href)
                    else:
                        article_url = href
                    
                    # Skip non-article URLs and duplicates
                    if ('search' in article_url.lower() or 'login' in article_url.lower() or 
                        any(article['url'] == article_url for article in all_articles)):
                        continue
                    
                    try:
                        print(f"Parsing article: {article_url}")
                        article_data = parse_article(article_url, source_name)
                        if article_data and is_relevant(article_data['content'], query):
                            all_articles.append(article_data)
                            print(f"Added relevant article: {article_data['title']}")
                        else:
                            print(f"Article not relevant or failed to parse: {article_url}")
                    except Exception as e:
                        print(f"Error parsing article {article_url}: {e}")
                        continue
            
            except Exception as e:
                print(f"Error searching {source_name} with query {search_query}: {e}")
                continue
    
    return all_articles

def parse_article(url, source_name):
    """Parse article content using newspaper3k."""
    try:
        article = Article(url)
        article.download()
        # Add small delay to avoid rate limiting
        time.sleep(random.uniform(0.3, 0.8))
        article.parse()
        
        try:
            article.nlp()  # This extracts keywords, summary, etc.
        except:
            print(f"NLP processing failed for {url}, continuing with basic parsing")
        
        # Get the publication date or use current date if not available
        pub_date = article.publish_date
        if not pub_date:
            pub_date = datetime.now()
            
        # If the article text is too short, it might not be a proper article
        if len(article.text) < 50:
            print(f"Article text too short ({len(article.text)} chars): {url}")
            # Try to get at least the title as content if the text is too short
            if article.title and len(article.title) > 10:
                article_text = article.title
            else:
                return None
        else:
            article_text = article.text
        
        return {
            'title': article.title or f"Article from {source_name}",
            'content': article_text,
            'summary': getattr(article, 'summary', article_text[:200] + "..."),
            'url': url,
            'source': source_name,
            'keywords': getattr(article, 'keywords', extract_keywords(article_text)),
            'date': pub_date.strftime("%Y-%m-%d")
        }
    except Exception as e:
        print(f"Error parsing article {url}: {e}")
        return None

def is_relevant(article_content, query, threshold=0.01):
    """Check if an article is relevant to the query."""
    # Simple relevance check
    query_keywords = set(extract_keywords(query))
    article_keywords = set(extract_keywords(article_content))
    
    # Check keyword overlap
    if not query_keywords:
        return True  # If no keywords could be extracted, consider it relevant
    
    # Check if the full query appears in the article
    if query.lower() in article_content.lower():
        print(f"Direct match found in article content")
        return True
        
    # Check individual words (including shorter ones since keywords might be names or places)
    query_words = [word.lower() for word in query.split() if len(word) > 2]
    for word in query_words:
        if word.lower() in article_content.lower():
            print(f"Found keyword '{word}' in article content")
            return True
    
    overlap = query_keywords.intersection(article_keywords)
    relevance_score = len(overlap) / len(query_keywords) if query_keywords else 0
    
    print(f"Relevance score: {relevance_score}, threshold: {threshold}")
    print(f"Query keywords: {query_keywords}")
    print(f"Article keywords: {article_keywords}")
    print(f"Overlap: {overlap}")
    
    # Consider it relevant if we have any overlap at all
    if len(overlap) > 0:
        return True
        
    # Also check if there's significant partial word matching (like plurals, etc.)
    partial_matches = 0
    for q_word in query_keywords:
        if len(q_word) < 4:
            continue
        for a_word in article_keywords:
            if len(a_word) < 4:
                continue
            # Check if one is contained in the other 
            if q_word in a_word or a_word in q_word:
                partial_matches += 1
                
    if partial_matches > 0:
        print(f"Found {partial_matches} partial word matches")
        return True
    
    return relevance_score >= threshold

def fetch_news_articles(query, sources, max_results_per_source=3):
    """Fetch related news articles from multiple sources."""
    print(f"Fetching news for query: '{query}' from sources: {sources}")
    
    all_articles = []
    
    for source in sources:
        print(f"\n--- Searching source: {source} ---")
        source_articles = search_source(query, source, max_results_per_source)
        all_articles.extend(source_articles)
        print(f"Found {len(source_articles)} articles from {source}")
    
    print(f"Total articles found: {len(all_articles)}")
    return all_articles 