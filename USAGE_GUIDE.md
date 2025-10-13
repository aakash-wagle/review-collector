# 📖 Usage Guide - iPhone Review Analysis Pipeline

## Quick Start (5 Minutes)

### Step 1: Install Dependencies

**Option A: Using UV (Recommended)**
```bash
pip install uv
uv sync
```

**Option B: Using pip**
```bash
pip install -r requirements.txt
```

### Step 2: Set API Key

Choose **ONE** of the following:

**OpenAI (Recommended)**
```python
import os
os.environ["OPENAI_API_KEY"] = "sk-your-key-here"
```

**Google Gemini (Alternative)**
```python
import os
os.environ["GOOGLE_API_KEY"] = "your-gemini-key-here"
```

**No API Key? No problem!**
The pipeline works without an LLM using rule-based recommendations.

### Step 3: Run the Pipeline

Open `iphone_analysis_pipeline.ipynb` in Jupyter and run all cells!

## 📊 What You'll Get

### 1. Console Output
```
🚀 LANGCHAIN ORCHESTRATED REVIEW ANALYSIS PIPELINE
================================================================================
📅 Started at: 2024-10-13 14:30:00
📂 Input file: iphone_17_pro_max_reviews.csv
🤖 LLM enabled: Yes

✅ Stage 1: Data Ingestion
   4,700 records loaded
✅ Stage 2: Data Cleaning
   4,650 clean records
✅ Stage 3: Exploratory Analysis
   Statistics and visualizations complete
...
```

### 2. Visual Analytics

**Rating Distribution**
- Histogram showing star rating frequency
- Bar chart of rating counts

**Sentiment Analysis**
- Positive/Neutral/Negative distribution
- Compound score histogram
- Sentiment vs. Rating correlation

**Word Clouds**
- Overall review terms
- Positive keywords (green)
- Negative keywords (red)

**Topic Analysis**
- Aspect frequency bar chart
- Topic-sentiment heatmap
- Waterfall chart of sentiment scores

**Timeline**
- Review volume over time
- Trend analysis

### 3. Markdown Report

File: `iphone_17_analysis_report.md`

Sections:
- Executive Summary
- Dataset Overview
- Sentiment Analysis Results
- Topic & Aspect Analysis
- **Design Improvement Recommendations** ⭐
- Limitations & Future Work
- Conclusion

## 🎯 Example Workflow

### Basic Usage

```python
# 1. Import and initialize
from iphone_analysis_pipeline import ReviewAnalysisPipeline, initialize_llm

# 2. Set up LLM
llm = initialize_llm()  # Auto-detects OpenAI or Gemini

# 3. Create pipeline
pipeline = ReviewAnalysisPipeline(
    csv_path="iphone_17_pro_max_reviews.csv",
    llm=llm
)

# 4. Run analysis
results = pipeline.run_pipeline(save_report=True)

# 5. Access results
print(f"Total reviews: {results['ingestion']['total_records']}")
print(f"Positive sentiment: {results['sentiment']['positive_ratio']:.1%}")
print(f"Recommendations: {len(results['recommendations'])}")
```

### Advanced: Custom Analysis

```python
# Access individual agents
from iphone_analysis_pipeline import (
    DataIngestionAgent,
    SentimentAnalysisAgent,
    TopicModelingAgent
)

# Custom data ingestion
ingestion = DataIngestionAgent("custom_reviews.csv")
df = ingestion.load_data()

# Custom sentiment analysis
sentiment = SentimentAnalysisAgent(df)
df_sentiment, results = sentiment.run_sentiment_analysis()

# Custom topic modeling with modified aspects
topic = TopicModelingAgent(df_sentiment)
topics, sentiments = topic.run_topic_modeling()
```

## 🔧 Customization Examples

### 1. Change LLM Model

```python
# Use GPT-4 instead of GPT-4o-mini
from langchain_openai import ChatOpenAI

llm = ChatOpenAI(model="gpt-4", temperature=0.3)
```

### 2. Modify Aspect Keywords

Edit in `TopicModelingAgent`:

```python
aspects = {
    'battery': ['battery', 'charge', 'charging', 'power', 'life'],
    'camera': ['camera', 'photo', 'photos', 'picture', 'video'],
    'performance': ['fast', 'speed', 'performance', 'smooth', 'lag'],
    'design': ['design', 'look', 'color', 'weight', 'feel'],
    # Add custom aspects
    'software': ['ios', 'software', 'app', 'apps', 'update'],
    'connectivity': ['wifi', '5g', 'signal', 'network', 'bluetooth']
}
```

### 3. Adjust Visualization Style

```python
# Change matplotlib style
import matplotlib.pyplot as plt
plt.style.use('ggplot')  # or 'seaborn', 'bmh', etc.

# Change seaborn palette
import seaborn as sns
sns.set_palette("Set2")  # or "husl", "pastel", etc.
```

### 4. Filter Reviews by Date

```python
# Only analyze recent reviews
import pandas as pd
df = pd.read_csv('iphone_17_pro_max_reviews.csv')
df['parsed_date'] = pd.to_datetime(df['date_iso'])
df_recent = df[df['parsed_date'] >= '2024-10-01']

# Run pipeline on filtered data
df_recent.to_csv('recent_reviews.csv', index=False)
pipeline = ReviewAnalysisPipeline('recent_reviews.csv', llm)
```

## 🐛 Troubleshooting

### Issue: "No API key found"

**Solution**: Set environment variable before running

```python
import os
os.environ["OPENAI_API_KEY"] = "your-key"
```

Or use `.env` file:
```bash
# Create .env file
echo "OPENAI_API_KEY=your-key" > .env
```

```python
# Load in notebook
from dotenv import load_dotenv
load_dotenv()
```

### Issue: "Module not found"

**Solution**: Ensure all dependencies are installed

```bash
pip install -r requirements.txt
```

### Issue: NLTK data missing

**Solution**: Download NLTK resources

```python
import nltk
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('punkt_tab')
```

### Issue: Memory error with large datasets

**Solution**: Process in batches

```python
# Sample data for testing
df_sample = df.sample(1000)  # Use 1,000 reviews
df_sample.to_csv('sample_reviews.csv', index=False)
```

## 📊 Understanding the Output

### Sentiment Scores

- **Compound**: Overall sentiment (-1 to +1)
  - `> 0.05`: Positive
  - `-0.05 to 0.05`: Neutral
  - `< -0.05`: Negative

- **Positive**: Proportion of positive language (0 to 1)
- **Negative**: Proportion of negative language (0 to 1)
- **Neutral**: Proportion of neutral language (0 to 1)

### Topic Sentiment Interpretation

Example:
```
BATTERY: avg_sentiment=0.456
- Mentions: 850 reviews
- Positive Ratio: 75.2%
- Negative Ratio: 15.8%
```

**Interpretation**: Battery aspect is well-received (75% positive), but 15.8% of mentions are negative - investigate battery complaints.

### Recommendation Format

```
**1. Enhance Battery Management Under Gaming Loads**

Based on 85 mentions with 35.2% negative sentiment (avg: 0.125), 
customers express concerns about battery drain during gaming. 
Recommend optimizing thermal management and background processes.
```

**Action Items**:
1. Review battery drain patterns
2. Optimize power consumption
3. Consider hardware improvements

## 🚀 Google Colab Setup

### 1. Upload Files

```python
from google.colab import files

# Upload CSV
uploaded = files.upload()  # Select iphone_17_pro_max_reviews.csv
```

### 2. Install Dependencies

```python
!pip install -q langchain langchain-openai pandas matplotlib seaborn
!pip install -q wordcloud nltk vaderSentiment keybert bertopic plotly
```

### 3. Set API Key

```python
import os
from getpass import getpass

# Secure input
api_key = getpass('Enter your OpenAI API key: ')
os.environ["OPENAI_API_KEY"] = api_key
```

### 4. Run Pipeline

Upload the notebook to Colab and run all cells!

## 📈 Performance Tips

### 1. Optimize for Speed

```python
# Use smaller sample for testing
df_sample = df.sample(500)

# Reduce KeyBERT keywords
keywords = keybert_model.extract_keywords(text, top_n=10)  # instead of 15

# Skip visualizations in batch mode
# Comment out visualization cells
```

### 2. Reduce LLM Costs

```python
# Use cheaper model
llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0.3)

# Limit LLM calls
# Only generate summaries for top 3 aspects instead of 5
```

### 3. Parallel Processing

```python
# Process sentiment in parallel (advanced)
from multiprocessing import Pool

def compute_sentiment(text):
    return vader.polarity_scores(text)

with Pool(4) as p:
    sentiments = p.map(compute_sentiment, df['review_text'])
```

## 🎓 Educational Use

### Learning Objectives

1. **Data Engineering**: ETL pipeline design
2. **NLP Techniques**: Sentiment analysis, topic modeling
3. **LangChain**: Agent orchestration, prompt engineering
4. **Visualization**: Data storytelling with charts
5. **Automation**: End-to-end reproducible workflows

### Suggested Modifications (for Students)

1. **Add new sentiment analyzer**: Integrate TextBlob or transformers
2. **Implement caching**: Save intermediate results to disk
3. **Create API endpoint**: Wrap pipeline in FastAPI/Flask
4. **Build dashboard**: Use Streamlit/Dash for interactivity
5. **Add testing**: Write unit tests for each agent

## 📞 Support

- **Documentation**: See README.md
- **Examples**: Check notebook comments
- **Issues**: Review error messages carefully
- **Community**: Share findings and improvements!

---

**Happy Analyzing! 🎉**

Built with ❤️ using LangChain, Python, and Modern NLP

