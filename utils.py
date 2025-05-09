import re
import json
import os
import requests
from urllib.parse import urlparse
from datetime import datetime, timedelta
import hashlib
from typing import List, Dict, Tuple, Optional

# Cache settings
CACHE_DIR = os.path.join(os.path.dirname(__file__), 'cache')
CACHE_EXPIRY = 3600  # Cache expiry in seconds (1 hour)

def is_valid_url(url):
    """Check if a URL is valid."""
    try:
        result = urlparse(url)
        return all([result.scheme, result.netloc])
    except:
        return False

def extract_domain(url):
    """Extract domain from a URL."""
    if not is_valid_url(url):
        return None
    
    domain = urlparse(url).netloc
    # Remove www. if present
    if domain.startswith('www.'):
        domain = domain[4:]
    
    return domain

def get_source_credibility(domain):
    """
    Get the credibility score for a news source domain.
    Higher score means more credible (0-10 scale).
    """
    # Dictionary of known news sources and their credibility scores
    # This is a simplified version and should be expanded with real data
    credibility_scores = {
        'bbc.co.uk': 9.5,
        'bbc.com': 9.5,
        'reuters.com': 9.3,
        'apnews.com': 9.2,
        'nytimes.com': 8.7,
        'wsj.com': 8.5,
        'theguardian.com': 8.3,
        'economist.com': 8.7,
        'npr.org': 8.5,
        'altnews.in': 8.0,
        'politifact.com': 8.2,
        'factcheck.org': 8.3,
        'ndtv.com': 7.5,
        'thehindu.com': 7.8,
        'cnn.com': 7.0,
        'foxnews.com': 6.5,
        'buzzfeed.com': 5.5,
        'dailymail.co.uk': 4.5,
        'infowars.com': 2.0,
    }
    
    return credibility_scores.get(domain, 5.0)  # Default to neutral score

def get_cache_path(key):
    """Get the cache file path for a given key."""
    # Create a hash of the key to use as filename
    hashed_key = hashlib.md5(key.encode()).hexdigest()
    return os.path.join(CACHE_DIR, f"{hashed_key}.json")

def load_from_cache(key):
    """Load data from cache if it exists and is not expired."""
    cache_path = get_cache_path(key)
    
    if not os.path.exists(cache_path):
        return None
    
    try:
        with open(cache_path, 'r') as f:
            cache_data = json.load(f)
        
        # Check if cache is expired
        timestamp = datetime.fromisoformat(cache_data.get('timestamp', '2000-01-01'))
        if datetime.now() - timestamp > timedelta(seconds=CACHE_EXPIRY):
            return None
        
        return cache_data.get('data')
    
    except Exception as e:
        print(f"Error loading from cache: {e}")
        return None

def save_to_cache(key, data):
    """Save data to cache."""
    if not os.path.exists(CACHE_DIR):
        os.makedirs(CACHE_DIR, exist_ok=True)
    
    cache_path = get_cache_path(key)
    
    try:
        cache_data = {
            'timestamp': datetime.now().isoformat(),
            'data': data
        }
        
        with open(cache_path, 'w') as f:
            json.dump(cache_data, f)
    
    except Exception as e:
        print(f"Error saving to cache: {e}")

def fetch_url_with_cache(url, headers=None, timeout=10):
    """Fetch URL content with caching."""
    # Try to load from cache first
    cache_data = load_from_cache(url)
    if cache_data:
        return cache_data
    
    # If not in cache or expired, fetch from web
    try:
        response = requests.get(url, headers=headers, timeout=timeout)
        response.raise_for_status()
        
        # Save to cache
        save_to_cache(url, response.text)
        
        return response.text
    
    except Exception as e:
        print(f"Error fetching URL {url}: {e}")
        return None

def clean_html(html_text):
    """Clean HTML content by removing tags and unnecessary whitespace."""
    if not html_text:
        return ""
    
    # Remove HTML tags
    text = re.sub(r'<[^>]+>', ' ', html_text)
    
    # Remove extra whitespace
    text = re.sub(r'\s+', ' ', text).strip()
    
    return text

def extract_text_from_html(html_content):
    """Extract useful text from HTML content."""
    from bs4 import BeautifulSoup
    
    soup = BeautifulSoup(html_content, 'html.parser')
    
    # Remove script and style elements
    for element in soup(['script', 'style', 'header', 'footer', 'nav']):
        element.decompose()
    
    # Get text
    text = soup.get_text(separator=' ')
    
    # Clean up text
    lines = (line.strip() for line in text.splitlines())
    chunks = (phrase.strip() for line in lines for phrase in line.split("  "))
    text = ' '.join(chunk for chunk in chunks if chunk)
    
    return text

def query_ollama(prompt: str, model: str = "qwen2.5:3b") -> Optional[str]:
    """
    Query the Ollama API with the given prompt and model.
    
    Args:
        prompt: The prompt to send to the model
        model: The model to use (default: qwen2.5:3b)
        
    Returns:
        The model's response text or None if there was an error
    """
    try:
        print(f"Querying Ollama with model: {model}")
        response = requests.post(
            "http://localhost:11434/api/generate",
            json={
                "model": model,
                "prompt": prompt,
                "stream": False,
                "options": {
                    "temperature": 0.7,
                    "top_p": 0.9,
                    "num_predict": 1000
                }
            },
            timeout=60  # Increased timeout
        )
        
        if response.status_code == 200:
            result = response.json().get('response', '')
            print(f"Ollama response received: {len(result)} characters")
            return result
        else:
            print(f"Error querying Ollama: {response.status_code}")
            return None
    except requests.exceptions.Timeout:
        print("Ollama request timed out. The model might be taking too long to respond.")
        return None
    except requests.exceptions.ConnectionError:
        print("Connection error when querying Ollama. Make sure Ollama is running.")
        return None
    except Exception as e:
        print(f"Exception when querying Ollama: {e}")
        return None

def check_ollama_available() -> bool:
    """
    Check if Ollama is available on the system.
    
    Returns:
        True if Ollama is available, False otherwise
    """
    try:
        print("Checking if Ollama is available...")
        response = requests.get("http://localhost:11434/api/tags", timeout=5)
        available = response.status_code == 200
        print(f"Ollama available: {available}")
        return available
    except requests.exceptions.ConnectionError:
        print("Ollama is not running or not installed")
        return False
    except Exception as e:
        print(f"Error checking Ollama availability: {e}")
        return False

def get_ai_verification(user_input: str, articles: List[Dict]) -> Tuple[str, str]:
    """
    Get AI-powered verification using Ollama's qwen2.5:3b model.
    
    Args:
        user_input: The user's input text
        articles: List of articles from trusted sources
        
    Returns:
        AI verdict and explanation
    """
    if not articles:
        return "Unable to verify", "No reference articles found to compare with."
    
    # Create a summary of the top 3 articles
    article_summaries = []
    for i, article in enumerate(articles[:3]):
        article_summaries.append(f"Article {i+1} from {article['source']}: {article['content'][:300]}...")
    
    article_text = "\n\n".join(article_summaries)
    
    # Create the prompt for the AI
    prompt = f"""You are a fact-checking AI assistant. Your task is to determine if the following news is true or false based on reference articles.

NEWS TO VERIFY:
{user_input}

REFERENCE ARTICLES FROM TRUSTED SOURCES:
{article_text}

Analyze the news and reference articles carefully. Focus on:
1. Do the reference articles confirm the key facts in the news?
2. Are there any contradictions between the news and reference articles?
3. Is the news missing important context that changes its meaning?
4. What specific facts from the news are confirmed or contradicted by the reference articles?

Provide your verdict as "TRUE", "PARTIALLY TRUE", "UNVERIFIED", or "FALSE", followed by a brief explanation.
"""
    
    # Query the model
    print("Sending verification request to Ollama...")
    response = query_ollama(prompt)
    
    if response:
        # Extract verdict and explanation
        print(f"Analyzing Ollama response for verdict...")
        if "TRUE" in response.upper() and "PARTIALLY" not in response.upper() and "NOT TRUE" not in response.upper():
            verdict = "TRUE"
        elif "PARTIALLY TRUE" in response.upper():
            verdict = "PARTIALLY TRUE"
        elif "FALSE" in response.upper() or "NOT TRUE" in response.upper():
            verdict = "FALSE"
        else:
            verdict = "UNVERIFIED"
            
        # Return the verdict and explanation
        print(f"AI verdict: {verdict}")
        return verdict, response
    else:
        return "Unable to verify", "Could not connect to the AI verification model."

def summarize_article(article: Dict, max_length: int = 300) -> str:
    """
    Create a concise summary of an article.
    
    Args:
        article: Article dictionary
        max_length: Maximum length of the summary
        
    Returns:
        Article summary
    """
    title = article.get('title', 'Untitled')
    source = article.get('source', 'Unknown')
    content = article.get('content', '')
    
    # Use the article's summary if available, otherwise use the first part of the content
    if 'summary' in article and article['summary']:
        summary = article['summary']
    else:
        summary = content[:max_length]
        if len(content) > max_length:
            summary += "..."
    
    return f"{title} ({source}): {summary}"

def format_non_zero_scores(similarity_scores: List[float]) -> str:
    """
    Format the non-zero similarity scores for display.
    
    Args:
        similarity_scores: List of similarity scores
        
    Returns:
        Formatted string with non-zero scores and their count
    """
    non_zero_scores = [score for score in similarity_scores if score > 0]
    
    if not non_zero_scores:
        return "No matching articles found"
    
    return f"Found {len(non_zero_scores)} matching articles with scores: {', '.join([f'{score:.1f}' for score in non_zero_scores])}"

def get_ai_summary(article_content: str, max_length: int = 150) -> str:
    """
    Get an AI-generated summary of an article.
    
    Args:
        article_content: The article content to summarize
        max_length: Maximum desired length of the summary
        
    Returns:
        AI-generated summary or empty string if failed
    """
    if len(article_content) < 100:
        return article_content  # No need to summarize very short content
    
    prompt = f"""Summarize the following article in about {max_length} words:

{article_content[:2000]}

Keep the summary concise and focus on the key facts and information.
"""
    
    response = query_ollama(prompt)
    if response:
        return response
    else:
        # Fall back to a simple extraction if AI summarization fails
        return article_content[:max_length] + "..." 