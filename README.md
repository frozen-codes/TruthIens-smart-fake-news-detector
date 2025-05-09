# TruthLens - Fake News Detector

TruthLens is a powerful tool that helps verify the credibility of news by comparing it with information from trusted sources in real-time. It uses advanced text similarity algorithms and optional AI-powered verification to determine if news is likely credible or potentially misleading.

## Features

- **Real-time News Verification**: Compare input text with articles from trusted news sources
- **Multiple Source Support**: Search across Google News, BBC, Reuters, NDTV, CNN, and more
- **Advanced Similarity Analysis**: Uses sophisticated algorithms to detect matching content
- **Zero-Exclusion Analysis**: Calculates credibility scores using only non-zero matches for better accuracy
- **AI-Powered Verification**: Optional integration with Ollama for AI-powered analysis and summaries
- **Detailed Results**: View similarity scores, article matches, and credibility assessments

## Installation

1. Clone this repository:
   ```
   git clone https://github.com/yourusername/truthlens.git
   cd truthlens
   ```

2. Create and activate a virtual environment:
   ```
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. Install the required packages:
   ```
   pip install -r requirements.txt
   ```

4. (Optional) Install Ollama for AI-powered verification:
   - Visit [ollama.com](https://ollama.com) and follow the installation instructions
   - Pull the required model: `ollama pull qwen2.5:3b`

## Usage

1. Run the application:
   ```
   streamlit run app.py
   ```

2. Enter a news headline or article text in the input field

3. Select which news sources to check against (Google News is recommended)

4. Enable AI-powered verification if Ollama is installed

5. Click "Analyze Text" to compare your input with real-time news from trusted sources

## How It Works

1. **Input Analysis**: The system extracts key information and keywords from your input
2. **Source Search**: It searches selected news sources for relevant articles
3. **Similarity Calculation**: Each article is compared with your input using advanced algorithms
4. **Credibility Assessment**: A credibility score is calculated based on matches with trusted sources
5. **AI Verification** (Optional): If enabled, the AI analyzes the input and reference articles to provide an additional assessment

## Components

- **app.py**: The main Streamlit application
- **scraper.py**: Handles fetching articles from news sources
- **verifier.py**: Contains the similarity and credibility assessment algorithms
- **utils.py**: Utility functions including Ollama integration
- **model.py**: Machine learning model for additional verification (optional)

## AI Integration

TruthLens can use Ollama's AI models to provide additional verification:

1. **Fact Checking**: The AI analyzes your input against reference articles
2. **Article Summarization**: Generates concise summaries of reference articles
3. **Verdict Generation**: Provides a verdict (TRUE, PARTIALLY TRUE, FALSE, UNVERIFIED)

Supported models:
- qwen2.5:3b (default)
- llama3:8b
- mistral:7b

## Tips for Best Results

- Use specific, detailed news claims rather than vague statements
- Include key facts, names, dates, and locations in your query
- Select Google News as one of your sources for most up-to-date results
- For breaking news, use NDTV or other sources that update frequently
- If using AI verification, allow sufficient time for processing

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- Built with Streamlit, NLTK, scikit-learn, and other open-source libraries
- Uses Ollama for local AI model integration
- Inspired by the need for accessible tools to combat misinformation

## Project Structure

```
truthlens/
├── app.py                  # Main Streamlit app
├── scraper.py              # Web scraping functions
├── verifier.py             # Text comparison and similarity scoring
├── model.py                # ML model (optional)
├── utils.py                # Helper functions
├── cache/                  # Cached data
├── requirements.txt        # Dependencies
└── README.md               # Documentation
```

## Future Enhancements

- URL input for direct article verification
- Advanced machine learning model for better classification
- Highlight potentially false statements within the article
- Support for multiple languages
- Browser extension integration

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgements

- [Streamlit](https://streamlit.io/) for the web application framework
- [NLTK](https://www.nltk.org/) and [scikit-learn](https://scikit-learn.org/) for NLP capabilities
- [Newspaper3k](https://newspaper.readthedocs.io/) for article extraction
- [BeautifulSoup](https://www.crummy.com/software/BeautifulSoup/) for web scraping 