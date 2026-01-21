# AliExpress Review Sentiment Analysis Pipeline

A comprehensive Streamlit application that integrates web scraping, text preprocessing, feature extraction, and sentiment analysis for AliExpress product reviews.

## Features

- **Web Scraping**: Automatically scrape reviews from AliExpress product URLs
- **Text Preprocessing**: Clean and process review text with language detection
- **Feature Extraction**: TF-IDF vectorization for text features
- **Sentiment Analysis**: BERT-based sentiment classification using Transformers
- **Interactive Dashboard**: Real-time visualization of results with Plotly
- **Multi-language Support**: Handles reviews in multiple languages
- **Export Functionality**: Download results as CSV files

## Prerequisites

- Python 3.8 or higher
- pip (Python package manager)
- At least 4GB of RAM (8GB recommended for larger datasets)

## Installation

### 1. Clone or Download the Files

Save all the provided files to a directory:
- `streamlit_app.py` - Main Streamlit application
- `scraper.py` - Web scraping module
- `preprocessing.py` - Text preprocessing module
- `sentiment.py` - Sentiment analysis module
- `requirements.txt` - Python dependencies

### 2. Create a Virtual Environment (Recommended)

```bash
# Windows
python -m venv venv
venv\Scripts\activate

# macOS/Linux
python3 -m venv venv
source venv/bin/activate
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

### 4. Download NLTK Data

Run Python and execute:
```python
import nltk
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('averaged_perceptron_tagger')
nltk.download('omw-1.4')
```

##  Running the Application

### Start the Streamlit App

```bash
streamlit run streamlit_app.py
```

The application will open in your default web browser at `http://localhost:8501`

## How to Use

### Method 1: Scrape New Reviews

1. **Enter Product URL**: 
   - Navigate to the "Input" tab
   - Paste an AliExpress product URL
   - Example: `https://www.aliexpress.com/item/3256808746369608.html`

2. **Configure Settings** (Optional):
   - Use the sidebar to adjust pipeline settings
   - Set maximum pages to scrape
   - Adjust batch size for processing
   - Set minimum review length filter

3. **Start Analysis**:
   - Click the "Start Analysis" button
   - The pipeline will automatically:
     - Scrape reviews from AliExpress
     - Preprocess and clean the text
     - Extract TF-IDF features
     - Perform sentiment analysis

4. **View Results**:
   - Navigate through the tabs to see each stage
   - Check the "Results" tab for comprehensive visualizations
   - Download results using the sidebar buttons

### Method 2: Upload Existing Data

1. **Prepare CSV File**:
   - Format should include columns: `review_text`, `rating`, `buyer_country` (optional)

2. **Upload File**:
   - Use the file uploader in the "Input" tab
   - Select your CSV file

3. **Process Data**:
   - Navigate to "Preprocessing" tab and click "Start Preprocessing"
   - Continue to "Analysis" tab for sentiment analysis

##  Understanding the Results

### Sentiment Labels
- **POSITIVE**: Positive sentiment detected
- **NEGATIVE**: Negative sentiment detected
- **NEUTRAL**: Neutral or mixed sentiment (if supported by model)

### Confidence Scores
- Range: 0.0 to 1.0
- Higher scores indicate more confidence in the prediction
- Scores above 0.8 are considered high confidence

### Visualizations
- **Sentiment Distribution**: Pie chart showing overall sentiment breakdown
- **Rating Distribution**: Histogram of product ratings
- **Language Distribution**: Bar chart of review languages
- **Sentiment by Country**: Grouped bar chart (if country data available)
- **Confidence Distribution**: Histogram of model confidence scores

## Configuration Options

### Sentiment Models

The app supports multiple pre-trained models:

1. **DistilBERT SST-2** (Default)
   - Fast and accurate for English
   - Binary classification (Positive/Negative)

2. **BERT Multilingual**
   - Supports multiple languages
   - 5-star rating scale

3. **RoBERTa Twitter**
   - Optimized for social media text
   - Three-way classification (Positive/Neutral/Negative)

### Advanced Settings

- **Max Pages**: Limit number of pages to scrape (default: 10)
- **Batch Size**: Processing batch size (default: 16)
- **Min Review Length**: Filter short reviews (default: 10 chars)

## Troubleshooting

### Common Issues and Solutions

1. **"Model not found" Error**
   - Ensure you have internet connection for first-time model download
   - The model will be cached after first download

2. **Memory Error**
   - Reduce batch size in advanced settings
   - Process smaller number of reviews at a time

3. **Scraping Returns No Data**
   - Verify the URL is a valid AliExpress product page
   - Check your internet connection
   - The product may have no reviews

4. **Slow Processing**
   - First run downloads models (one-time process)
   - Consider using a smaller model or reducing batch size
   - GPU acceleration can significantly speed up processing

## Output Files

The application generates several output files:

- `sentiment_results_[timestamp].csv` - Complete analysis results
- `summary_[timestamp].csv` - Summary statistics

## Tips for Best Results

1. **URL Format**: Ensure you're using the full product URL, not a shortened version
2. **Language**: The app works best with English reviews but supports multiple languages
3. **Review Quality**: Longer, more detailed reviews typically yield better sentiment analysis
4. **Batch Processing**: For large datasets, consider processing in smaller batches

## Data Privacy

- All processing is done locally on your machine
- No data is sent to external servers (except for model downloads)
- Scraped data is only stored locally

## Contributing

Feel free to modify and extend the code for your needs. Some ideas for enhancement:
- Add more visualization types
- Integrate additional sentiment models
- Add aspect-based sentiment analysis
- Implement review summarization

## License

This project is for educational and research purposes. Please respect AliExpress's Terms of Service when scraping data.

## Support

If you encounter issues:
1. Check the console/terminal for error messages
2. Ensure all dependencies are properly installed
3. Verify your Python version is 3.8+
4. Try with a different product URL

## Ready to Analyze!

You're all set! Start analyzing AliExpress reviews and gain valuable insights into customer sentiment.
