# iPhone 17 Pro Max Reviews – Multilingual Analysis

Comprehensive analysis pipeline for Google reviews of the iPhone 17 Pro Max, including multilingual translation, sentiment analysis, aspect detection, topic modeling, and evidence-backed design recommendations.

## Project Overview

**Objective**: Clean, translate, and analyze Google reviews to produce:
- Exploratory Data Analysis (EDA)
- Sentiment analysis
- Topic/aspect identification
- Temporal trends
- 3-5 evidence-backed design recommendations
- Comparison of deterministic baselines vs. agentic (LangChain/LLM) approaches

**Dataset**: ~8,000 Google reviews with fields:
- `review_text`: User review text (multilingual)
- `stars`: Rating string (e.g., "Rated 4.0 out of 5")
- `date`: Review date (e.g., "Review submitted 3 weeks ago")

## Pipeline Stages

### S1: Ingest & Canonicalize
- Parse star ratings from various string formats
- Normalize to 5-star scale
- Parse relative dates to absolute timestamps
- Remove duplicates
- **Output**: `data/processed/01_ingested.parquet`

### S2: Language Detection & Translation
- Detect language using `langdetect`
- Translate non-English reviews to English
- Cache translations for efficiency
- Optional back-translation QA
- **Output**: `data/processed/02_translated.parquet`

### S3: Text Normalization & Feature Engineering
- Clean text (demojize, expand contractions, lowercase)
- Lemmatization with spaCy
- Extract features (word count, exclamation marks, etc.)
- **Output**: `data/processed/03_features.parquet`

### S4: Deterministic Baselines
- **EDA**: Rating trends over time, distribution analysis
- **Sentiment**: VADER sentiment analysis
- **Topics**: TF-IDF + KMeans clustering (k optimized by silhouette score)
- **Aspects**: Dictionary-based detection for 9 key aspects
- **Output**: `data/processed/04_baselines.parquet` + figures + tables

### S5: LangChain Agentic Layer (Optional)
- **Aspect Tagging**: LLM-based structured JSON extraction
- **Topic Labeling**: Human-readable labels for clusters
- **Insight Generation**: Automated bullet points and recommendations
- **Output**: `data/processed/05_llm_aspects.parquet`, `data/processed/05_llm_topics.parquet`, `outputs/llm/insights.md`

### S6: Evaluation & Reconciliation
- Compare LLM vs dictionary aspect detection
- Compute precision, recall, F1 scores
- **Output**: `data/processed/06_evaluation.parquet`

### S7: Final EDA & Storytelling Assets
- Generate all required visualizations:
  - Ratings over time with confidence intervals
  - Share of translated reviews over time
  - Aspect negative share comparison (LLM vs Dictionary)
  - Topic clusters with labels
  - Keyword spikes (Battery, Overheating, Camera)
- **Output**: `outputs/figures/*.png`, `outputs/tables/*.csv`

### S8: Evidence-Backed Recommendations
- Generate 3-5 actionable design recommendations
- Each includes: Finding, Evidence, Recommendation, Metric Target
- **Output**: `outputs/recommendations.md`

### S9: Two-Page Report
- Compile comprehensive PDF report
- Includes: Summary, Methods, Findings, Visualizations, Recommendations
- **Output**: `REPORT_2PAGES.pdf`

## Installation

### Prerequisites
- Python 3.11+
- (Optional) OpenAI API key for agentic layer (S5)

### Setup

1. **Create virtual environment**:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

2. **Install dependencies**:
```bash
pip install -r requirements.txt
```

3. **Download spaCy model**:
```bash
python -m spacy download en_core_web_sm
```

4. **Configure API keys** (optional, for S5):
Create a `.env` file:
```
OPENAI_API_KEY=your-api-key-here
```

## Usage

### Run Full Pipeline
```bash
python run_pipeline.py
```

This will execute all stages (S1-S9) in sequence.

### Run Individual Stages
```bash
# Example: Run only stage 1
python scripts/s1_ingest.py

# Example: Run only stage 4
python scripts/s4_baselines.py
```

### Skip Agentic Layer
To skip the LLM-based analysis (S5), set in `config.yaml`:
```yaml
agentic:
  enabled: false
```

## Outputs

After running the pipeline, you'll find:

### Main Deliverable
- **`REPORT_2PAGES.pdf`**: Comprehensive 2-page report

### Supporting Outputs
- **`outputs/figures/`**: All visualizations (PNG format)
  - `ratings_over_time_with_CI.png`
  - `share_translated_over_time.png`
  - `aspect_negative_share_llm_vs_dict.png`
  - `topic_clusters_with_llm_labels.png`
  - `keyword_spikes_battery_overheat_camera.png`

- **`outputs/tables/`**: Summary tables (CSV format)
  - `aspects_summary_dict.csv`
  - `aspects_summary_llm_vs_dict.csv`
  - `topics_summary_labeled.csv`
  - `evaluation_metrics.csv`

- **`outputs/recommendations.md`**: Evidence-backed design recommendations

- **`outputs/llm/insights.md`**: LLM-generated insights (if S5 enabled)

- **`outputs/run_logs.txt`**: Execution logs

### Processed Data
- **`data/processed/`**: Intermediate parquet files from each stage

## Configuration

Edit `config.yaml` to customize:

- **Seeds**: For reproducibility
- **Star parsing**: Regex patterns for rating extraction
- **Translation**: Model selection, caching, QA settings
- **Normalization**: Text cleaning options
- **Sentiment**: VADER thresholds
- **Topics**: KMeans parameters, TF-IDF settings
- **Aspects**: Catalog of aspects to detect
- **Agentic**: LLM provider, cost controls
- **Evaluation**: Metrics and audit settings

## Key Features

✅ **Multilingual Support**: Automatic detection and translation of non-English reviews

✅ **Robust Parsing**: Handles various star rating and date formats

✅ **Comprehensive Analysis**: Sentiment, topics, aspects, trends

✅ **Dual Approach**: Deterministic baselines + LLM-enhanced insights

✅ **Evaluation**: Quantitative comparison of detection methods

✅ **Actionable Output**: Evidence-backed recommendations with metric targets

✅ **Reproducible**: Seeded randomness, cached translations, logged execution

## Quality Gates

The pipeline includes automatic validation:
- ✓ >= 99% rows have valid star ratings after parsing
- ✓ 100% rows have valid dates
- ✓ Duplicate removal tracked
- ✓ All non-English texts translated
- ✓ No empty cleaned text
- ✓ Feature variance checks

## Cost Controls

For the agentic layer (S5):
- Translation caching to minimize API calls
- Stratified sampling for LLM analysis
- Configurable max API call limits
- Near-duplicate text deduplication

## Dependencies

Key packages:
- **Data**: pandas, numpy, pyarrow
- **NLP**: spacy, nltk, langdetect, transformers
- **Sentiment**: vaderSentiment
- **ML**: scikit-learn
- **Agentic**: langchain, langchain-openai, openai
- **Viz**: matplotlib, seaborn, plotly
- **Report**: reportlab

See `requirements.txt` for complete list with versions.

## Limitations

- Translation quality not validated at scale
- Manual aspect audit limited
- Temporal trends based on relative date approximations
- LLM tagging applied to sample only (cost constraints)

## Next Steps

- Expand LLM tagging to full dataset
- Implement active learning for aspect detection
- A/B testing of design changes based on recommendations
- Continuous monitoring dashboard for real-time tracking

## Project Structure

```
.
├── config.yaml                 # Configuration file
├── requirements.txt            # Python dependencies
├── run_pipeline.py            # Main pipeline orchestrator
├── README.md                  # This file
├── REPORT_2PAGES.pdf         # Final report (generated)
│
├── data/
│   ├── raw/
│   │   └── reviews.csv       # Raw input data
│   └── processed/            # Intermediate outputs (*.parquet)
│
├── outputs/
│   ├── figures/              # Visualizations (*.png)
│   ├── tables/               # Summary tables (*.csv)
│   ├── llm/                  # LLM insights (*.md)
│   ├── recommendations.md    # Final recommendations
│   └── run_logs.txt         # Execution logs
│
├── cache/                    # Translation and LLM caches (*.sqlite)
│
└── scripts/
    ├── utils.py             # Utility functions
    ├── s1_ingest.py         # Stage 1: Ingest
    ├── s2_translate.py      # Stage 2: Translation
    ├── s3_features.py       # Stage 3: Features
    ├── s4_baselines.py      # Stage 4: Baselines
    ├── s5_agentic.py        # Stage 5: Agentic
    ├── s6_evaluate.py       # Stage 6: Evaluation
    ├── s7_storytelling.py   # Stage 7: Storytelling
    ├── s8_recommendations.py # Stage 8: Recommendations
    └── s9_report.py         # Stage 9: Report
```

## License

This project is for academic/research purposes.

## Contact

For questions or issues, please refer to the project documentation or course materials.

---

**Generated by**: Automated Review Analysis Pipeline  
**Version**: 1.0  
**Date**: October 2024

