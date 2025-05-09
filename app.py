import streamlit as st
from scraper import fetch_news_articles, NEWS_SOURCES
from verifier import calculate_similarity, check_credibility
import pandas as pd
import sys
import importlib.util
import time

st.set_page_config(
    page_title="TruthLens - Fake News Detector",
    page_icon="🔍",
    layout="wide"
)

# Check if utils module is available
try:
    from utils import format_non_zero_scores, get_ai_summary
    
    # Check if Ollama is available
    ollama_available = False
    try:
        # Try to import the check_ollama_available function
        from utils import check_ollama_available, get_ai_verification
        ollama_available = check_ollama_available()
    except (ImportError, Exception) as e:
        print(f"Error checking Ollama availability: {e}")
        ollama_available = False
except ImportError:
    # Define fallback function if utils module is not available
    def format_non_zero_scores(similarity_scores):
        non_zero_scores = [score for score in similarity_scores if score > 0]
        if not non_zero_scores:
            return "No matching articles found"
        return f"Found {len(non_zero_scores)} matching articles with scores: {', '.join([f'{score:.1f}' for score in non_zero_scores])}"
    
    def get_ai_summary(article_content, max_length=150):
        return article_content[:max_length] + "..." if len(article_content) > max_length else article_content
    
    ollama_available = False

def main():
    st.title("🔍 TruthLens - Fake News Detector")
    st.subheader("Verify the credibility of news headlines and articles")
    
    # Show AI status
    if ollama_available:
        st.sidebar.success("✅ AI-powered verification is available")
    else:
        st.sidebar.warning("⚠️ AI-powered verification is not available. Install and run Ollama with qwen2.5:3b model.")
        st.sidebar.markdown("To install Ollama, visit [ollama.com](https://ollama.com)")
    
    # Show information about the application
    with st.expander("ℹ️ About this application", expanded=False):
        st.markdown("""
        This application helps verify the credibility of news by comparing it with information from trusted sources.
        
        **How it works:**
        1. Enter a news headline or article text in the input field
        2. Select which news sources to check against (Google News is recommended)
        3. Click "Analyze Text" to compare your input with real-time news from trusted sources
        
        **Note:** The app works best with recent and specific news topics. The more detailed your input, the better the results.
        """)
    
    tab1, tab2 = st.tabs(["Text Input", "URL Input"])
    
    with tab1:
        user_input = st.text_area("Enter a news headline or article text:", height=150, 
                                 placeholder="Enter a news headline or story to verify...")
        col1, col2 = st.columns(2)
        with col1:
            sources = st.multiselect(
                "Select news sources to check against:",
                list(NEWS_SOURCES.keys()),
                default=["Google News", "NDTV"]
            )
            
            # Show a message if Google News is not selected
            if "Google News" not in sources:
                st.info("💡 Tip: Google News is recommended for most up-to-date results")
                
        with col2:
            num_articles = st.slider("Number of articles to compare with:", 3, 15, 5)
            use_ai = st.checkbox("Use AI-powered verification", 
                                value=ollama_available, 
                                disabled=not ollama_available)
            
            if use_ai:
                ai_model = st.selectbox(
                    "Select AI model:",
                    ["qwen2.5:3b", "llama3:8b", "mistral:7b"],
                    index=0,
                    disabled=not ollama_available
                )
            else:
                ai_model = "qwen2.5:3b"  # Default model
        
        analyze_button = st.button("Analyze Text", type="primary", use_container_width=True)
        
        if analyze_button and user_input:
            if len(user_input.split()) < 3:
                st.warning("Please enter a more detailed text (at least 3 words) for better results.")
            else:
                with st.spinner("Analyzing the news... This may take a moment as we gather real-time information."):
                    # Fetch related articles from reliable sources
                    articles = fetch_news_articles(user_input, sources, num_articles)
                    
                    if not articles:
                        st.error("No related articles found to verify this information. Try a different query or select different news sources.")
                        st.info("Tips: Use specific search terms or try a recent news topic that is likely to be covered by major media outlets. Adding Google News as a source often helps find recent articles.")
                    else:
                        # Calculate similarity and credibility
                        similarities = [calculate_similarity(user_input, article['content']) for article in articles]
                        credibility_score, verdict = check_credibility(similarities)
                        
                        # Get AI verification if enabled and available
                        ai_verdict = None
                        ai_explanation = None
                        if use_ai and ollama_available:
                            with st.spinner("Getting AI-powered verification... This may take up to a minute."):
                                try:
                                    # Add a progress bar for AI processing
                                    progress_bar = st.progress(0)
                                    for i in range(100):
                                        # Update progress bar
                                        progress_bar.progress(i + 1)
                                        if i == 20:
                                            st.info("Analyzing articles...")
                                        elif i == 50:
                                            st.info("Comparing with your input...")
                                        elif i == 80:
                                            st.info("Generating verification results...")
                                        time.sleep(0.1)
                                    
                                    ai_verdict, ai_explanation = get_ai_verification(user_input, articles)
                                    progress_bar.empty()
                                except Exception as e:
                                    st.error(f"Error with AI verification: {e}")
                                    ai_verdict = None
                        
                        # Generate AI summaries for articles if enabled
                        if use_ai and ollama_available:
                            for article in articles:
                                if len(article['content']) > 300:
                                    try:
                                        article['ai_summary'] = get_ai_summary(article['content'])
                                    except Exception as e:
                                        print(f"Error generating AI summary: {e}")
                                        article['ai_summary'] = None
                        
                        # Display results
                        display_results(credibility_score, verdict, articles, similarities, user_input, ai_verdict, ai_explanation, use_ai and ollama_available)
    
    with tab2:
        url_input = st.text_input("Enter a news article URL:", placeholder="https://example.com/news/article")
        analyze_url_button = st.button("Analyze URL", type="primary", use_container_width=True)
        
        if analyze_url_button and url_input:
            st.info("URL analysis feature coming soon!")

def display_results(credibility_score, verdict, articles, similarities, user_input, ai_verdict=None, ai_explanation=None, show_ai_summaries=False):
    # Display the overall verdict with appropriate color
    st.subheader("Analysis Results")
    
    col1, col2 = st.columns([1, 2])
    
    with col1:
        st.metric("Credibility Score", f"{credibility_score:.1f}/10")
        
        if credibility_score >= 7:
            st.success(f"Verdict: {verdict}")
        elif credibility_score >= 4:
            st.success(f"Verdict: {verdict}")
        elif credibility_score >= 2:
            st.warning(f"Verdict: {verdict}")
        else:
            st.info(f"Verdict: {verdict}")
        
        # Display non-zero scores
        non_zero_scores = [score for score in similarities if score > 0]
        if non_zero_scores:
            st.success(f"Found {len(non_zero_scores)} matching articles out of {len(similarities)} total")
        else:
            st.warning("No direct matches found in articles")
    
    with col2:
        if credibility_score >= 7:
            st.markdown("✅ **This information appears to be credible** based on similarity with trusted sources.")
        elif credibility_score >= 4:
            st.markdown("✅ **This information is likely accurate** based on partial matches with trusted sources.")
        elif credibility_score >= 2:
            st.markdown("⚠️ **This information contains some elements that match trusted sources**, but requires further verification.")
        elif len(articles) > 0:
            st.markdown("⚠️ **We found some related articles, but couldn't verify the specific claims** in your text. The information may be partially accurate.")
        else:
            st.markdown("ℹ️ **Insufficient information available** to verify this claim. Consider searching for more specific details.")
        
        # Display a note about our confidence
        if len(articles) > 0:
            if credibility_score >= 4:
                st.markdown("**Confidence level:** High - Multiple sources confirm this information")
            elif credibility_score >= 2:
                st.markdown("**Confidence level:** Medium - Some elements are confirmed by trusted sources")
            else:
                st.markdown("**Confidence level:** Low - Related news found but specific claims not verified")
    
    # Display AI verification if available
    if ai_verdict:
        st.subheader("AI-Powered Verification")
        
        verdict_col, explanation_col = st.columns([1, 3])
        
        with verdict_col:
            if ai_verdict == "TRUE":
                st.success(f"AI Verdict: {ai_verdict}")
            elif ai_verdict == "PARTIALLY TRUE":
                st.warning(f"AI Verdict: {ai_verdict}")
            elif ai_verdict == "FALSE":
                st.error(f"AI Verdict: {ai_verdict}")
            else:
                st.info(f"AI Verdict: {ai_verdict}")
        
        with explanation_col:
            if ai_explanation:
                st.markdown("**AI Analysis:**")
                st.markdown(ai_explanation)
    
    # Highlight aspects of the query
    if len(articles) > 0:
        st.markdown("### Key aspects of your query")
        from scraper import extract_keywords
        query_keywords = extract_keywords(user_input, n=8)
        st.write("We looked for information about: " + ", ".join([f"**{k}**" for k in query_keywords]))
    
    # Display reference articles
    st.subheader(f"Reference Articles ({len(articles)} found)")
    
    # Sort articles by similarity
    sorted_articles = sorted(zip(articles, similarities), key=lambda x: x[1], reverse=True)
    
    if len(sorted_articles) > 0:
        # Create tabs for different views
        article_tab, similarity_tab = st.tabs(["Article View", "Similarity View"])
        
        with article_tab:
            for i, (article, similarity) in enumerate(sorted_articles):
                with st.expander(f"{article['title']} ({article['source']}) - Similarity: {similarity:.1f}/10"):
                    st.markdown(f"**Source:** {article['source']}")
                    st.markdown(f"**Published:** {article['date']}")
                    st.markdown(f"**URL:** [Read full article]({article['url']})")
                    
                    # Display AI summary if available
                    if show_ai_summaries and 'ai_summary' in article and article['ai_summary']:
                        st.markdown("**AI Summary:**")
                        st.markdown(article['ai_summary'])
                    
                    st.markdown("**Excerpt:**")
                    st.markdown(article['content'][:500] + "..." if len(article['content']) > 500 else article['content'])
                    
                    # Add a button to open the article in a new tab
                    st.markdown(f"<a href='{article['url']}' target='_blank'><button style='background-color:#4CAF50;color:white;padding:8px 16px;border:none;border-radius:4px;cursor:pointer;'>Open Article</button></a>", unsafe_allow_html=True)
                
        with similarity_tab:
            # Create a table of similarity scores
            similarity_data = {
                "Article": [article['title'] for article, _ in sorted_articles],
                "Source": [article['source'] for article, _ in sorted_articles],
                "Similarity Score": [f"{similarity:.1f}/10" for _, similarity in sorted_articles],
                "Link": [article['url'] for article, _ in sorted_articles]
            }
            
            # Create DataFrame without index column
            df = pd.DataFrame(similarity_data).reset_index(drop=True)
            st.dataframe(df)
            
            # Display formatted non-zero scores
            st.markdown("### Matching Articles")
            st.markdown(format_non_zero_scores(similarities))
    else:
        st.info("No reference articles found matching your query.")

if __name__ == "__main__":
    main() 