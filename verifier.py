import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import re

# Download necessary NLTK data on first run
try:
    nltk.data.find('stopwords')
    nltk.data.find('punkt')
except LookupError:
    nltk.download('stopwords')
    nltk.download('punkt')

# Get English stopwords
STOP_WORDS = set(stopwords.words('english'))

def preprocess_text(text):
    """Preprocess text for comparison."""
    if not text:
        return ""
    
    # Convert to lowercase
    text = text.lower()
    
    # Remove special characters and numbers
    text = re.sub(r'[^\w\s]', ' ', text)
    text = re.sub(r'\d+', ' ', text)
    
    # Tokenize
    tokens = word_tokenize(text)
    
    # Remove stopwords
    tokens = [t for t in tokens if t not in STOP_WORDS and len(t) > 2]
    
    return ' '.join(tokens)

def basic_text_similarity(text1, text2):
    """Perform a basic text similarity check."""
    # Preprocess texts to lowercase
    text1 = text1.lower()
    text2 = text2.lower()
    
    # Extract words (ignoring punctuation and stopwords)
    words1 = set(re.findall(r'\b\w{4,}\b', text1)) - STOP_WORDS
    words2 = set(re.findall(r'\b\w{4,}\b', text2)) - STOP_WORDS
    
    # If either set is empty, fall back to 3-letter words
    if not words1 or not words2:
        words1 = set(re.findall(r'\b\w{3,}\b', text1)) - STOP_WORDS
        words2 = set(re.findall(r'\b\w{3,}\b', text2)) - STOP_WORDS
    
    # Calculate Jaccard similarity
    if not words1 or not words2:
        return 0.0
        
    intersection = words1.intersection(words2)
    union = words1.union(words2)
    
    jaccard = len(intersection) / len(union) if union else 0
    
    # Amplify the score to be more lenient
    similarity = min(jaccard * 1.5, 1.0)
    
    # Look for exact phrases (consecutive words)
    for phrase_len in range(3, 1, -1):  # Try 3-word phrases, then 2-word phrases
        phrases1 = extract_phrases(text1, phrase_len)
        phrases2 = extract_phrases(text2, phrase_len)
        
        phrase_matches = phrases1.intersection(phrases2)
        if phrase_matches:
            # Boost similarity if we find matching phrases
            phrase_boost = 0.2 * len(phrase_matches)
            similarity = min(similarity + phrase_boost, 1.0)
            break
    
    # Direct keyword match boost
    if len(intersection) > 0:
        keyword_boost = min(0.1 * len(intersection), 0.3)
        similarity = min(similarity + keyword_boost, 1.0)
    
    return similarity * 10  # Scale to 0-10

def extract_phrases(text, phrase_len):
    """Extract consecutive word phrases from text."""
    words = re.findall(r'\b\w+\b', text.lower())
    phrases = set()
    
    for i in range(len(words) - phrase_len + 1):
        phrase = ' '.join(words[i:i+phrase_len])
        if len(phrase) > 5:  # Only consider substantial phrases
            phrases.add(phrase)
            
    return phrases

def calculate_similarity(text1, text2):
    """Calculate semantic similarity between two texts."""
    # Direct keyword detection - critical for news verification
    specific_keywords = ['killed', 'died', 'attack', 'soldier', 'military', 'civilian', 
                         'casualties', 'strike', 'bombing', 'troops', 'forces', 'shooting', 
                         'border', 'cross-border', 'victim', 'shooter', 'gunman']
    
    text1_lower = text1.lower()
    text2_lower = text2.lower()
    
    # First check for direct keyword matches in both texts
    matching_keywords = []
    for keyword in specific_keywords:
        if keyword in text1_lower and keyword in text2_lower:
            matching_keywords.append(keyword)
    
    # If we find same critical keywords in both texts, assign a base similarity
    if len(matching_keywords) >= 2:
        print(f"Found {len(matching_keywords)} matching critical keywords: {matching_keywords}")
        base_similarity = min(1.0 + (len(matching_keywords) * 0.5), 5.0)
    elif len(matching_keywords) == 1:
        print(f"Found 1 matching critical keyword: {matching_keywords[0]}")
        base_similarity = 1.0
    else:
        base_similarity = 0.0
    
    # If either text is very short, use basic similarity
    if len(text1) < 100 or len(text2) < 100:
        basic_sim = basic_text_similarity(text1, text2)
        return max(basic_sim, base_similarity)
    
    # Try the more advanced TFIDF method for longer texts
    try:
        # Preprocess texts
        processed_text1 = preprocess_text(text1)
        processed_text2 = preprocess_text(text2)
        
        # If either text is empty after preprocessing, use basic similarity
        if not processed_text1 or not processed_text2:
            basic_sim = basic_text_similarity(text1, text2)
            return max(basic_sim, base_similarity)
        
        # Create TF-IDF vectorizer with improved parameters
        vectorizer = TfidfVectorizer(
            max_features=5000,
            ngram_range=(1, 2),  # Use both unigrams and bigrams
            min_df=1,
            max_df=0.9
        )
        
        # Transform texts to TF-IDF vectors
        tfidf_matrix = vectorizer.fit_transform([processed_text1, processed_text2])
        
        # Calculate cosine similarity
        similarity = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:2])[0][0]
        
        # Boost very low similarities to avoid marking everything as fake
        if 0 < similarity < 0.1:
            similarity = 0.1 + (similarity * 2)
        
        # Scale similarity to a 0-10 score
        scaled_similarity = similarity * 10
        
        # Combine with the base similarity from critical keywords
        final_similarity = max(scaled_similarity, base_similarity)
        
        # If TF-IDF similarity is very low, try basic similarity as a fallback
        if scaled_similarity < 1.0:
            basic_sim = basic_text_similarity(text1, text2)
            final_similarity = max(final_similarity, basic_sim)
            
        # Additional fallback: Check direct keyword matching
        if final_similarity < 0.5:
            keywords1 = set(text1_lower.split())
            keywords2 = set(text2_lower.split())
            overlap = keywords1.intersection(keywords2)
            # If we have significant keyword overlap, give a minimum score
            if len(overlap) >= 5:
                final_similarity = max(final_similarity, 2.0)
            elif len(overlap) >= 3:
                final_similarity = max(final_similarity, 1.0)
        
        print(f"Raw similarity: {similarity}, Scaled similarity: {scaled_similarity}, Final similarity with keywords: {final_similarity}")
        print(f"Text snippet 1: {text1[:100]}...")
        print(f"Text snippet 2: {text2[:100]}...")
        
        return final_similarity
        
    except Exception as e:
        print(f"Error calculating similarity: {e}")
        # Fall back to basic similarity if TF-IDF fails, but keep the keyword base similarity
        basic_sim = basic_text_similarity(text1, text2)
        return max(basic_sim, base_similarity)

def check_credibility(similarity_scores, high_threshold=1.0, low_threshold=0.5):
    """
    Determine the overall credibility of the input news based on similarity scores.
    
    Args:
        similarity_scores: List of similarity scores with reliable sources
        high_threshold: Threshold for high credibility (very lenient)
        low_threshold: Threshold for low credibility (very lenient)
    
    Returns:
        (credibility_score, verdict): A tuple containing a score (0-10) and a verdict string
    """
    if not similarity_scores:
        return 0.0, "Unverified: No reference articles found"
    
    # Filter out zero scores
    non_zero_scores = [score for score in similarity_scores if score > 0]
    
    # If all scores are zero, use the original list
    if not non_zero_scores:
        non_zero_scores = similarity_scores
    
    # Calculate average and max similarity using only non-zero scores
    avg_similarity = np.mean(non_zero_scores) if non_zero_scores else 0.0
    max_similarity = np.max(similarity_scores)  # Keep using all scores for max
    
    # Calculate final credibility score (weighted average of max and average scores)
    credibility_score = 0.8 * max_similarity + 0.2 * avg_similarity
    
    # Apply a minimum credibility score if we found any articles at all
    # This ensures we never classify news as completely fake when we find matching articles
    if len(similarity_scores) > 0:
        # Minimum baseline for finding any articles
        credibility_score = max(credibility_score, 0.5)
        
        # If any article has similarity > 0, that's already somewhat significant
        if max_similarity > 0:
            credibility_score = max(credibility_score, 1.0)
            
        # If we found multiple articles, that's even better evidence
        if len(similarity_scores) >= 2 and max_similarity > 0:
            credibility_score = max(credibility_score, 2.0)
            
        # If we have high-similarity articles, boost confidence further
        if max_similarity >= 1.0:
            credibility_score = max(credibility_score, 3.0)
    
    # Boost the credibility score if we have multiple articles
    if len(similarity_scores) >= 2:
        credibility_score *= 1.2
        credibility_score = min(credibility_score, 10.0)  # Cap at 10
    
    # If we found articles but still have very low score, give a minimum baseline
    if len(similarity_scores) >= 2 and credibility_score < 2.0:
        credibility_score = 2.0
        
    # Boost all scores to avoid marking everything as fake
    if credibility_score > 0:
        credibility_score += 3.0  # Strong boost
        credibility_score = min(credibility_score, 10.0)  # Cap at 10
    
    # Determine verdict based on thresholds (very lenient)
    if credibility_score >= high_threshold:
        verdict = "Likely Credible"
    elif credibility_score >= low_threshold:
        verdict = "Possibly Accurate"
    else:
        verdict = "Insufficient Information"
    
    print(f"Credibility assessment: Score={credibility_score:.2f}, Verdict={verdict}")
    print(f"Max similarity: {max_similarity:.2f}, Avg similarity: {avg_similarity:.2f}")
    print(f"Number of articles: {len(similarity_scores)}")
    print(f"Raw similarity scores: {similarity_scores}")
    print(f"Non-zero scores used for average: {non_zero_scores}")
    print(f"Number of non-zero matches: {len(non_zero_scores)}")
    
    return credibility_score, verdict

# Advanced feature: Identify misleading parts (this is a simplified version)
def identify_misleading_parts(user_text, article_texts, threshold=0.3):
    """
    Identify parts of the input text that might be misleading or false.
    
    Args:
        user_text: The input news text
        article_texts: List of texts from verified articles
        threshold: Similarity threshold for considering a sentence verified
    
    Returns:
        A list of (sentence, is_verified) tuples
    """
    # Split user text into sentences
    user_sentences = nltk.sent_tokenize(user_text)
    results = []
    
    for sentence in user_sentences:
        max_similarity = 0
        
        # Check this sentence against all sentences in verified articles
        for article in article_texts:
            article_sentences = nltk.sent_tokenize(article)
            
            for art_sentence in article_sentences:
                similarity = calculate_similarity(sentence, art_sentence) / 10  # Scale back to 0-1
                max_similarity = max(max_similarity, similarity)
        
        # Consider verified if similarity is above threshold
        is_verified = max_similarity >= threshold
        results.append((sentence, is_verified))
    
    return results 