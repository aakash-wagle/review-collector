# 🍎 iPhone 17 Pro Max Review Analysis Pipeline

**Comprehensive LangChain-Based Analytical System**

A fully automated 8-agent pipeline that extracts, cleans, analyzes, and interprets customer review data to produce evidence-backed design improvement recommendations.

---

## 📋 Overview

This pipeline implements a sophisticated multi-agent system using LangChain to analyze iPhone 17 Pro Max customer reviews. It performs:

- **Data Ingestion & Validation**: Loads and validates review datasets
- **Data Cleaning**: Normalizes text, extracts ratings, removes duplicates
- **Exploratory Data Analysis**: Generates statistics and distributions
- **Sentiment Analysis**: Uses VADER to compute sentiment scores
- **Topic Extraction**: Identifies key product aspects discussed
- **Insight Generation**: Creates actionable design recommendations
- **Visualization**: Produces comprehensive charts and word clouds
- **Report Generation**: Compiles findings into a detailed markdown report

---

## 🚀 Quick Start

### Prerequisites

- Python 3.8+
- Virtual environment (recommended)
- `iphone_17_pro_max_reviews.csv` in the project directory

### Installation

1. **Create and activate virtual environment:**

```bash
python3 -m venv venv_analysis
source venv_analysis/bin/activate  # On Windows: venv_analysis\Scripts\activate
```

2. **Install dependencies:**

```bash
pip install -r requirements.txt
```

### Running the Analysis

**Option 1: Using the main script (Recommended)**

```bash
python run_analysis.py
```

**Option 2: Using the orchestrator directly**

```bash
python pipeline_orchestrator.py iphone_17_pro_max_reviews.csv --output-dir ./analysis_output
```

**Option 3: Using individual agents programmatically**

```python
from pipeline_orchestrator import ReviewAnalysisPipeline

pipeline = ReviewAnalysisPipeline(use_llm=False)
result = pipeline.run_full_pipeline('iphone_17_pro_max_reviews.csv', './output')
```

---

## 📦 Project Structure

```
review-collector/
├── agents/
│   ├── __init__.py
│   ├── data_ingestion_agent.py      # Stage 1: Load data
│   ├── data_cleaning_agent.py       # Stage 2: Clean data
│   ├── eda_agent.py                 # Stage 3: Exploratory analysis
│   ├── sentiment_agent.py           # Stage 4: Sentiment analysis
│   ├── topic_agent.py               # Stage 5: Topic extraction
│   ├── insight_agent.py             # Stage 6: Generate insights
│   ├── visualization_agent.py       # Stage 7: Create visualizations
│   └── report_agent.py              # Stage 8: Generate report
├── pipeline_orchestrator.py          # Main pipeline coordinator
├── run_analysis.py                   # Simple execution script
├── requirements.txt                  # Python dependencies
├── iphone_17_pro_max_reviews.csv    # Input data
└── README_ANALYSIS.md               # This file
```

---

## 🔧 The 8-Agent Pipeline

### 1. Data Ingestion Agent
- **Purpose**: Load and validate CSV data
- **Output**: Validated DataFrame with required columns
- **Validation**: Checks for `review_text`, `stars`, `date` columns

### 2. Data Cleaning Agent
- **Purpose**: Preprocess and normalize data
- **Operations**:
  - Extract numeric ratings from "Rated X.X out of 5" format
  - Parse relative dates to ISO format
  - Remove duplicates and empty reviews
  - Normalize text (whitespace, punctuation)
- **Output**: Clean DataFrame ready for analysis

### 3. Exploratory Data Analysis (EDA) Agent
- **Purpose**: Generate descriptive statistics
- **Metrics**:
  - Rating distribution
  - Average, median, mode ratings
  - Text length statistics
  - Word frequency analysis
  - Temporal trends
- **Output**: Statistical summary and word frequencies

### 4. Sentiment Analysis Agent
- **Purpose**: Compute sentiment polarity
- **Method**: VADER (Valence Aware Dictionary and sEntiment Reasoner)
- **Output**: 
  - Compound sentiment scores (-1 to +1)
  - Sentiment labels (positive/neutral/negative)
  - Sentiment distribution statistics

### 5. Topic Extraction Agent
- **Purpose**: Identify product aspects discussed
- **Aspects Tracked**:
  - Battery life
  - Camera quality
  - Performance/speed
  - Design/aesthetics
  - Display quality
  - Build quality
  - Price/value
  - Features/software
  - Size/weight
  - Setup process
  - Thermal management
  - Upgrade experience
- **Output**: Aspect frequency, sentiment correlation, pain points, strengths

### 6. Insight Generation Agent
- **Purpose**: Generate actionable recommendations
- **Method**: Rule-based reasoning (LLM-enhanced optional)
- **Output**: 
  - 3-5 prioritized design recommendations
  - Evidence from data
  - Expected impact statements

### 7. Visualization Agent
- **Purpose**: Create comprehensive visualizations
- **Outputs**:
  - Rating distribution charts (bar + pie)
  - Sentiment distribution visualizations
  - Topic-sentiment heatmap
  - Word clouds (all, positive, negative reviews)
  - Temporal trends (if date data available)
- **Format**: High-resolution PNG files (300 DPI)

### 8. Report Generation Agent
- **Purpose**: Compile comprehensive markdown report
- **Sections**:
  - Executive summary
  - Dataset overview
  - Sentiment analysis results
  - Product aspect analysis
  - Design recommendations
  - Methodology
  - Limitations and future work
- **Output**: `analysis_report.md`

---

## 📊 Output Files

After running the pipeline, you'll find these files in the output directory:

### Reports
- **analysis_report.md**: Comprehensive markdown report with all findings

### Visualizations
- **rating_distribution.png**: Bar and pie charts of rating distribution
- **sentiment_distribution.png**: Sentiment analysis visualizations
- **topic_sentiment_heatmap.png**: Product aspects with sentiment coloring
- **wordcloud_all.png**: Word cloud from all reviews
- **wordcloud_positive.png**: Word cloud from positive reviews
- **wordcloud_negative.png**: Word cloud from negative reviews

---

## 🧪 Example Usage

### Basic Analysis

```python
from pipeline_orchestrator import ReviewAnalysisPipeline

# Initialize pipeline
pipeline = ReviewAnalysisPipeline(use_llm=False)

# Run analysis
result = pipeline.run_full_pipeline(
    csv_path='iphone_17_pro_max_reviews.csv',
    output_dir='./my_analysis'
)

# Access results
if result['status'] == 'success':
    print("Analysis complete!")
    print(f"Report: {result['output_directory']}/analysis_report.md")
```

### With LLM Enhancement

```python
import os
from pipeline_orchestrator import ReviewAnalysisPipeline

# Set OpenAI API key
os.environ['OPENAI_API_KEY'] = 'your-api-key-here'

# Initialize with LLM
pipeline = ReviewAnalysisPipeline(use_llm=True)

# Run analysis (insights will be LLM-enhanced)
result = pipeline.run_full_pipeline(
    'iphone_17_pro_max_reviews.csv',
    './llm_analysis'
)
```

### Accessing Individual Agent Results

```python
pipeline = ReviewAnalysisPipeline()
result = pipeline.run_full_pipeline('data.csv', './output')

# Access processed data
df = pipeline.get_data()

# Access individual stage results
results = pipeline.get_results()
eda_results = results['eda']
sentiment_results = results['sentiment']
topic_results = results['topics']
insights = results['insights']
```

---

## 🔬 Methodology

### Sentiment Analysis
- **Tool**: VADER (Valence Aware Dictionary and sEntiment Reasoner)
- **Rationale**: Optimized for social media and short-form text
- **Scoring**: Compound score from -1 (most negative) to +1 (most positive)
- **Classification**:
  - Positive: compound ≥ 0.05
  - Negative: compound ≤ -0.05
  - Neutral: -0.05 < compound < 0.05

### Topic Extraction
- **Method**: Keyword-based pattern matching
- **Approach**: Predefined aspect dictionaries with semantic keywords
- **Sentiment Correlation**: Cross-reference aspect mentions with sentiment scores
- **Pain Point Detection**: Aspects with >5% mention rate and <-0.1 avg sentiment
- **Strength Detection**: Aspects with >5% mention rate and >0.3 avg sentiment

### Recommendation Generation
- **Primary**: Rule-based logic using statistical thresholds
- **Optional**: LLM-enhanced reasoning with GPT-3.5/4
- **Prioritization**: Based on mention frequency and sentiment intensity
- **Format**: Title, Evidence, Action, Impact, Priority

---

## 📈 Performance

**Typical Execution Time** (on standard hardware):
- 1,000 reviews: ~30 seconds
- 5,000 reviews: ~60 seconds
- 10,000 reviews: ~2 minutes

**Memory Usage**:
- Peak: ~500MB for 10,000 reviews
- Average: ~200MB for 5,000 reviews

---

## 🎯 Use Cases

- **Product Development**: Identify features to improve/add
- **Customer Experience**: Understand pain points and satisfaction drivers
- **Competitive Analysis**: Compare sentiment across product versions
- **Marketing**: Highlight strengths in campaigns
- **Quality Assurance**: Track emerging issues from user feedback
- **Academic Research**: Sentiment analysis and NLP research

---

## 🔄 Extending the Pipeline

### Adding a New Agent

1. Create agent file in `agents/` directory:
```python
class MyCustomAgent:
    def perform_analysis(self, df):
        # Your logic here
        return {'status': 'success', 'results': {...}}
```

2. Import in `pipeline_orchestrator.py`:
```python
from agents.my_custom_agent import MyCustomAgent
```

3. Add to pipeline initialization:
```python
self.custom_agent = MyCustomAgent()
```

4. Add execution stage in `run_full_pipeline()`:
```python
custom_result = self.custom_agent.perform_analysis(self.data)
self.results['custom'] = custom_result
```

### Adding New Product Aspects

Edit `agents/topic_agent.py`:
```python
self.aspect_keywords = {
    ...
    'new_aspect': ['keyword1', 'keyword2', 'keyword3'],
}
```

### Customizing Visualizations

Modify `agents/visualization_agent.py` to add new plot types or customize existing ones.

---

## 🐛 Troubleshooting

### Common Issues

**1. ModuleNotFoundError: No module named 'agents'**
- Ensure you're running from the project root directory
- Check that `agents/__init__.py` exists

**2. FileNotFoundError: 'iphone_17_pro_max_reviews.csv'**
- Verify CSV file is in the current directory
- Use absolute path: `pipeline.run_full_pipeline('/full/path/to/data.csv', ...)`

**3. WordCloud images not generated**
- Install wordcloud: `pip install wordcloud`
- If installation fails, the pipeline will skip word clouds

**4. Memory issues with large datasets**
- Process data in chunks
- Reduce visualization resolution
- Close figures after saving: `plt.close('all')`

---

## 📚 Dependencies

### Core
- pandas >= 2.2.0
- numpy >= 1.26.3

### LangChain
- langchain >= 0.1.7
- langchain-community >= 0.0.20
- langchain-openai >= 0.0.5 (optional, for LLM features)

### NLP & Sentiment
- nltk >= 3.8.1
- vaderSentiment >= 3.3.2
- transformers >= 4.37.2 (optional)
- torch >= 2.2.0 (optional)

### Visualization
- matplotlib >= 3.8.2
- seaborn >= 0.13.2
- wordcloud >= 1.9.3

### ML & Processing
- scikit-learn >= 1.4.0

---

## 📝 Citation

If you use this pipeline in research or publications, please cite:

```bibtex
@software{iphone_review_analysis_2025,
  title = {iPhone Review Analysis Pipeline: LangChain-Based Sentiment and Topic Extraction},
  author = {Your Name},
  year = {2025},
  version = {1.0},
  url = {https://github.com/yourusername/review-collector}
}
```

---

## 🤝 Contributing

Contributions are welcome! Areas for improvement:

- Advanced topic modeling (BERTopic, LDA)
- Multi-language support
- Real-time streaming analysis
- Interactive dashboards (Streamlit/Dash)
- Comparative analysis across products
- Temporal trend forecasting

---

## 📄 License

MIT License - See LICENSE file for details

---

## 👥 Support

For issues, questions, or feature requests:
- Open an issue on GitHub
- Email: your.email@example.com

---

**Built with ❤️ using LangChain, VADER, and Python**

*Last updated: October 2025*

