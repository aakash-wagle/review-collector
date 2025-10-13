# 🔬 LangChain Agentic Pipeline for iPhone 17 Pro Max Review Analysis

A fully automated, modular data science pipeline that extracts, cleans, analyzes, and interprets customer review data to produce evidence-backed design improvement recommendations using LangChain and advanced NLP techniques.

## 📋 Table of Contents

- [Overview](#overview)
- [Features](#features)
- [Architecture](#architecture)
- [Installation](#installation)
- [Usage](#usage)
- [Pipeline Stages](#pipeline-stages)
- [Output](#output)
- [Requirements](#requirements)
- [Project Structure](#project-structure)
- [Customization](#customization)
- [Limitations](#limitations)
- [License](#license)

## 🎯 Overview

This project implements a comprehensive analytical pipeline that:

1. **Ingests** raw review data from CSV files
2. **Cleans** and normalizes text, ratings, and dates
3. **Analyzes** review patterns through exploratory data analysis
4. **Computes** sentiment using VADER sentiment analyzer
5. **Extracts** key topics and themes using KeyBERT
6. **Generates** actionable recommendations using LLM-based reasoning
7. **Visualizes** findings with publication-quality charts
8. **Reports** insights in a comprehensive markdown document

**Dataset**: 4,700+ Google Reviews for iPhone 17 Pro Max  
**Framework**: LangChain orchestration with OpenAI/Gemini LLM  
**Output**: Automated insights and design recommendations

## ✨ Features

### 🤖 **8 Specialized Agents**

1. **Data Ingestion Agent** - Validates and loads review data
2. **Data Cleaning Agent** - Normalizes text, extracts ratings, removes duplicates
3. **EDA Agent** - Generates statistics, word clouds, and timeline trends
4. **Sentiment Analysis Agent** - Applies VADER for polarity scoring
5. **Topic Modeling Agent** - Identifies key aspects using KeyBERT
6. **Insight Generation Agent** - LLM-powered recommendation synthesis
7. **Visualization Agent** - Creates comprehensive charts and plots
8. **Report Generation Agent** - Produces markdown summary with findings

### 🎨 **Visualization Suite**

- Rating distribution histograms
- Sentiment vs. rating scatter plots
- Positive/negative word clouds
- Topic-sentiment heatmaps
- Review timeline trends
- Aspect frequency bar charts

### 🧠 **Advanced Analytics**

- VADER sentiment analysis with compound scoring
- KeyBERT-based keyword extraction
- Aspect-based sentiment correlation
- Multi-dimensional topic modeling
- Statistical summary generation

### 🔗 **LangChain Integration**

- Modular agent architecture
- LLM-driven insight generation
- Prompt engineering for recommendations
- OpenAI GPT-4 and Google Gemini support
- Fallback mechanisms for offline operation

## 🏗️ Architecture

```
┌─────────────────────────────────────────────────────────┐
│         LangChain Orchestrated Pipeline                 │
└─────────────────────────────────────────────────────────┘
                          │
        ┌─────────────────┴─────────────────┐
        │                                   │
   ┌────▼────┐                         ┌───▼────┐
   │ Agent 1 │  Data Ingestion         │ Agent 5│  Topic Modeling
   │ Agent 2 │  Data Cleaning          │ Agent 6│  LLM Insights
   │ Agent 3 │  EDA & Stats            │ Agent 7│  Visualization
   │ Agent 4 │  Sentiment Analysis     │ Agent 8│  Report Generation
   └─────────┘                         └────────┘
        │                                   │
        └───────────────┬───────────────────┘
                        │
                   ┌────▼─────┐
                   │  Output  │
                   │  Report  │
                   │ +Visuals │
                   └──────────┘
```

## 📦 Installation

### Option 1: Using UV (Recommended)

```bash
# Install UV if not already installed
pip install uv

# Sync dependencies
uv sync

# Activate virtual environment
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
```

### Option 2: Using pip

```bash
# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### Option 3: Google Colab

```python
# Install required packages in Colab
!pip install -q langchain langchain-openai langchain-google-genai
!pip install -q pandas matplotlib seaborn wordcloud nltk vaderSentiment
!pip install -q transformers torch keybert bertopic scikit-learn plotly
!pip install -q umap-learn hdbscan openai google-generativeai
```

## 🚀 Usage

### Quick Start

1. **Set your API key**:

```python
import os

# For OpenAI
os.environ["OPENAI_API_KEY"] = "your-openai-api-key"

# OR for Google Gemini
os.environ["GOOGLE_API_KEY"] = "your-google-api-key"
```

2. **Run the notebook**:

Open `iphone_analysis_pipeline.ipynb` and run all cells, or execute the entire pipeline programmatically:

```python
from iphone_analysis_pipeline import ReviewAnalysisPipeline, initialize_llm

# Initialize LLM
llm = initialize_llm()

# Create pipeline
pipeline = ReviewAnalysisPipeline(
    csv_path="iphone_17_pro_max_reviews.csv",
    llm=llm
)

# Execute pipeline
results = pipeline.run_pipeline(save_report=True)
```

3. **View results**:

The pipeline generates:
- **Markdown report**: `iphone_17_analysis_report.md`
- **Inline visualizations** in the notebook
- **Console output** with execution logs

## 🔄 Pipeline Stages

### Stage 1: Data Ingestion
- Loads CSV file
- Validates schema (review_text, stars, date)
- Removes malformed entries
- **Output**: Raw DataFrame

### Stage 2: Data Cleaning
- Normalizes text (whitespace, encoding)
- Extracts numeric ratings from "Rated X.X out of 5"
- Standardizes dates to ISO format
- Removes duplicate reviews
- **Output**: Clean DataFrame

### Stage 3: Exploratory Analysis
- Computes summary statistics (mean, median, mode)
- Generates rating distribution plots
- Creates word frequency analysis
- Produces word clouds
- Plots review timeline
- **Output**: EDA results dict + visualizations

### Stage 4: Sentiment Analysis
- Applies VADER sentiment analyzer
- Assigns sentiment labels (positive/neutral/negative)
- Computes aggregate metrics
- Analyzes sentiment-rating correlation
- **Output**: DataFrame with sentiment scores

### Stage 5: Topic Modeling
- Extracts keywords using KeyBERT
- Identifies product aspects (battery, camera, design, etc.)
- Computes sentiment per topic
- Quantifies aspect frequency
- **Output**: Topics dict + sentiment mapping

### Stage 6: Insight Generation (LLM)
- Prepares context summary
- Generates topic summaries via LLM
- Creates 5 actionable design recommendations
- Justifies with evidence from data
- **Output**: Recommendations list

### Stage 7: Visualization
- Creates comprehensive dashboard
- Sentiment vs. rating scatter plot
- Positive/negative word cloud comparison
- Topic sentiment waterfall chart
- **Output**: Multiple matplotlib/plotly figures

### Stage 8: Report Generation
- Executive summary
- Data overview
- Sentiment analysis results
- Topic analysis breakdown
- Design recommendations
- Limitations and future work
- **Output**: Markdown report file

## 📊 Output

### Generated Files

1. **`iphone_17_analysis_report.md`** - Comprehensive analysis report
2. **Inline visualizations** - All charts displayed in notebook
3. **Console logs** - Execution summary and progress tracking

### Sample Report Sections

- **Executive Summary**: Key findings at a glance
- **Dataset Overview**: Data quality and characteristics
- **Sentiment Analysis**: Polarity distribution and metrics
- **Topic Analysis**: Aspect-level sentiment breakdown
- **Recommendations**: 5 evidence-backed design improvements
- **Limitations**: Methodological constraints and future work

## 📋 Requirements

### Core Dependencies

```
langchain>=0.3.0
langchain-openai>=0.2.0
langchain-google-genai>=2.0.0
pandas>=2.3.3
matplotlib>=3.8.0
seaborn>=0.13.0
wordcloud>=1.9.0
nltk>=3.8.0
vaderSentiment>=3.3.2
keybert>=0.8.0
bertopic>=0.16.0
plotly>=5.18.0
```

### LLM Requirements

**Option 1: OpenAI**
- API key from https://platform.openai.com/
- Model: `gpt-4o-mini` or `gpt-4`

**Option 2: Google Gemini**
- API key from https://ai.google.dev/
- Model: `gemini-1.5-flash` or `gemini-pro`

**Fallback**: Pipeline works without LLM using rule-based recommendations

## 📁 Project Structure

```
review-collector/
├── iphone_analysis_pipeline.ipynb    # Main analysis notebook
├── google_reviews_scraper.ipynb      # Web scraper (optional)
├── iphone_17_pro_max_reviews.csv     # Review dataset (4,700+ reviews)
├── pyproject.toml                     # Dependencies (UV)
├── uv.lock                            # Dependency lock file
├── README.md                          # This file
└── __init__.py                        # Package marker
```

## 🎨 Customization

### Modify Aspect Keywords

Edit the `aspects` dictionary in `TopicModelingAgent`:

```python
aspects = {
    'battery': ['battery', 'charge', 'charging', 'power'],
    'camera': ['camera', 'photo', 'picture', 'video'],
    # Add your custom aspects here
}
```

### Change LLM Model

Modify the `initialize_llm()` function:

```python
llm = ChatOpenAI(model="gpt-4", temperature=0.5)  # Use GPT-4 instead
```

### Adjust Visualization Style

Customize colors, chart types in the `VisualizationAgent` class:

```python
plt.style.use('ggplot')  # Change matplotlib style
sns.set_palette("Set2")   # Change seaborn palette
```

## ⚠️ Limitations

1. **Sample Bias**: Reviews are self-selected; may not represent all customers
2. **Time Period**: Analysis covers recent reviews only
3. **Language**: Limited to English-language reviews
4. **Sentiment Accuracy**: VADER is rule-based; transformers may be more accurate
5. **LLM Costs**: OpenAI API usage incurs costs per request

## 🔮 Future Enhancements

- [ ] Implement transformer-based sentiment models (BERT, RoBERTa)
- [ ] Add multilingual support
- [ ] Comparative analysis with competitor products
- [ ] Real-time scraping integration
- [ ] Interactive Streamlit dashboard
- [ ] Fake review detection
- [ ] Temporal trend analysis over extended periods

## 📜 License

This project is open source for educational and research purposes.

## 🤝 Contributing

Contributions are welcome! Please feel free to submit issues or pull requests.

## 📧 Contact

For questions or collaboration:
- **Project**: LangChain Review Analysis Pipeline
- **Purpose**: Academic evaluation and data science education

---

**Built with**: LangChain • Python • VADER • KeyBERT • OpenAI/Gemini  
**Dataset**: 4,700+ iPhone 17 Pro Max Google Reviews  
**Status**: Production-ready, fully automated ✅

