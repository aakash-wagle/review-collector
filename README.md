# Beats Solo3 Wireless – Multilingual Review Analysis

**Local NLP-Based Review Analysis Pipeline (No LLM/API)**

A complete, production-ready pipeline for analyzing multilingual product reviews using local NLP models, extracting sentiment, identifying key aspects, performing topic modeling, and generating evidence-backed recommendations.

---

## Quick Summary

- **Dataset**: 5,928 Google reviews for Beats Solo3 Wireless headphones
- **Input**: Review text + star ratings (1-5 scale, with flexible parsing)
- **Output**: 12 visualization figures, 11 analysis tables, actionable recommendations
- **Runtime**: ~15-20 minutes (GPU recommended but not required)
- **Methods**: VADER + SiEBERT sentiment, PyABSA aspect extraction, TF-IDF + KMeans topics

---

## Project Structure

```
.
├── config.yaml              # Pipeline configuration
├── requirements.txt         # Python dependencies
├── main.py                 # Main orchestrator (runs S1-S8)
│
├── data/
│   ├── raw/                # Input: reviews.csv
│   └── processed/          # Intermediate parquet files (01-06)
│
├── src/
│   ├── s1_ingest.py        # Data ingestion, language detection, star parsing
│   ├── s2_translate.py     # Conditional translation (M2M100)
│   ├── s3_features.py      # Text normalization & feature engineering
│   ├── s4_baselines.py     # VADER sentiment, topics, dictionary aspects
│   ├── s5_models.py        # SiEBERT sentiment, PyABSA aspect extraction
│   ├── s6_evaluate.py      # Model evaluation & comparison
│   ├── s7_story.py         # All visualizations (12 figures)
│   └── s8_recommendations.py # Generate evidence-backed recommendations
│
├── outputs/
│   ├── figures/            # 12 PNG visualizations
│   ├── tables/             # 11 CSV analysis tables
│   └── recommendations.md  # Final recommendations (3-5 items)
│
└── cache/                  # Model & translation cache (auto-generated)
```

---

## Installation

### Prerequisites
- Python 3.11+
- 8GB+ RAM (16GB recommended for transformer models)
- GPU optional (speeds up processing by 50-70%)

### Setup (5 minutes)

```bash
# 1. Create virtual environment
python -m venv venv

# Activate (Windows PowerShell)
.\venv\Scripts\activate

# Activate (Linux/Mac)
source venv/bin/activate

# 2. Install dependencies
pip install -r requirements.txt

# 3. Download spaCy English model
python -m spacy download en_core_web_sm
```

**Key Dependencies**:
- Core: pandas, numpy, pyyaml, pyarrow
- NLP: spacy, nltk, langdetect, emoji
- Sentiment: vaderSentiment
- ML: torch, transformers, scikit-learn, pyabsa
- Visualization: matplotlib, seaborn

---

## Usage

### Run Complete Pipeline

```bash
python main.py
```

This executes all 8 stages (S1-S8) sequentially and generates all outputs in `outputs/`.

### Run Specific Stages

```bash
# Run stages 1-4 only
python main.py --end-at s4

# Resume from stage 5
python main.py --start-from s5

# Run single stage (e.g., regenerate visualizations)
python main.py --start-from s7 --end-at s7
```

---

## Pipeline Stages

| Stage | Name | Input | Output | Description |
|-------|------|-------|--------|-------------|
| **S1** | Ingest | `reviews.csv` | `01_ingested.parquet` | Load data, parse star ratings (handles "3/5", "three stars", etc.), detect languages, remove duplicates |
| **S2** | Translate | `01_ingested.parquet` | `02_translated.parquet` | Conditional translation of non-English reviews using M2M100 (local model) |
| **S3** | Features | `02_translated.parquet` | `03_features.parquet` | Text normalization (lemmatization, demojization, contraction expansion), feature engineering |
| **S4** | Baselines | `03_features.parquet` | `04_baselines.parquet` | VADER sentiment, TF-IDF + KMeans topic modeling (with semantic labels), dictionary-based aspect tagging |
| **S5** | Models | `03_features.parquet` | `05_models.parquet` | SiEBERT transformer sentiment, PyABSA aspect extraction (DeBERTa-v3 backend) |
| **S6** | Evaluate | `04+05` | `06_evaluation.parquet` | Compare VADER vs SiEBERT vs Stars, PyABSA vs Dictionary aspects |
| **S7** | Storytelling | `06_evaluation.parquet` | 12 figures + tables | Generate all visualizations and analysis tables |
| **S8** | Recommendations | All outputs | `recommendations.md` | Extract evidence and generate 3-5 actionable recommendations |

---

## Output Files

### Figures (12 total in `outputs/figures/`)

1. **`ratings_histogram.png`** - Distribution of star ratings
2. **`language_distribution.png`** - Language breakdown (English/non-English)
3. **`sentiment_confusion_vader_vs_stars.png`** - VADER accuracy vs ground truth
4. **`sentiment_confusion_vader_vs_siebert.png`** - VADER vs SiEBERT comparison
5. **`siebert_vs_stars_agreement.png`** - SiEBERT error analysis (4 subplots)
6. **`aspect_negative_share_pyabsa_vs_dict.png`** - Aspect comparison
7. **`topic_clusters_with_auto_labels.png`** - Topic distribution with semantic labels
8. **`topic_clusters_2d_scatter.png`** - t-SNE 2D projection of review topics
9. **`topic_clusters_2d_pca.png`** - PCA 2D projection of review topics
10. **`keyword_rates_by_rating_sound_comfort_battery.png`** - Keyword trends across ratings
11. **`text_length_distribution.png`** - Review length analysis by star rating
12. **`top_ngrams_negative.png`** - Most frequent words/phrases in negative reviews

### Tables (11 total in `outputs/tables/`)

1. **`language_distribution.csv`** - Language breakdown
2. **`topics_summary.csv`** - Topic clusters with keywords and statistics
3. **`aspects_summary_dict.csv`** - Dictionary-based aspect summary
4. **`aspects_summary_pyabsa.csv`** - PyABSA aspect summary
5. **`aspect_comparison_pyabsa_vs_dict.csv`** - Aspect method comparison
6. **`evaluation_metrics.csv`** - VADER vs SiEBERT performance metrics
7. **`siebert_error_examples.csv`** - SiEBERT misclassification examples
8. **`summary_statistics.csv`** - Overall dataset statistics
9. **`recommendations.csv`** - Structured recommendations data
10-11. Additional intermediate summaries

---

## Key Results & Insights

### Sentiment Analysis Performance

| Metric | VADER | SiEBERT | Winner |
|--------|-------|---------|--------|
| **Accuracy** | 83.81% | **89.42%** | SiEBERT +5.61% |
| **F1-Score** | 82.02% | **87.05%** | SiEBERT +5.03% |

- **SiEBERT** significantly outperforms rule-based VADER
- 97%+ accuracy on extreme ratings (1★ and 5★)
- Limitation: Cannot predict neutral sentiment (0% on 3★ reviews)

### Top Issues (from N-gram Analysis)

**Most common bigrams in negative reviews**:
1. "stopped working" (73 mentions) → Reliability/durability issues
2. "sound quality" (58) → Mixed sentiment
3. "waste money" (27) → Price/value concerns
4. "hurt ears" (23) → Comfort/fit problems
5. "noise cancellation" (21) → Feature expectations

### Topic Distribution

- 8 distinct topics identified via KMeans clustering
- Semantic labels generated from keywords + average star ratings
- Problem topics: "Product Failures/Breaking", "Comfort Issues"
- Strong topics: "Sound Quality Positive", "Battery Life Praise"

---

## Configuration

All pipeline parameters are in `config.yaml`:

```yaml
# Key sections:
- seeds: Reproducibility (numpy, sklearn)
- translation: Enable/disable, thresholds, back-translation QA
- sentiment: VADER thresholds, SiEBERT model
- topics: KMeans k range, TF-IDF parameters
- aspects: Target aspect catalog, PyABSA backend
```

Edit `config.yaml` to:
- Change sentiment model (`siebert/sentiment-roberta-large-english`)
- Adjust topic cluster count (k_min/k_max)
- Modify aspect categories
- Enable/disable translation

---

## Performance & Optimization

### Speed
- **CPU-only**: ~25-30 minutes
- **GPU (CUDA)**: ~15-20 minutes
- Bottlenecks: SiEBERT inference, PyABSA aspect extraction, t-SNE (for 2D viz)

### GPU Usage
- SiEBERT automatically uses GPU if available (batch_size=256)
- PyABSA automatically uses GPU if available
- Translation (M2M100) benefits from GPU but not required

### Memory
- Peak RAM: ~6-8GB (CPU) or ~4-6GB (GPU w/ offloading)
- Intermediate files: ~50MB (parquet compression)

---

## Quality Gates & Validation

✅ **Data Quality**: >99% valid star ratings after parsing  
✅ **Language Coverage**: Detects non-English, translates if >5% of dataset  
✅ **Sentiment Accuracy**: 89.4% agreement with star-based ground truth (SiEBERT)  
✅ **Aspect Extraction**: Dictionary (100% coverage) + PyABSA (transformer-based)  
✅ **Topic Coherence**: Silhouette score used for optimal k selection  

---

## Reproducibility

- All random seeds fixed in `config.yaml` (numpy=42, sklearn=42)
- Deterministic pipeline (given same input + config)
- Model versions pinned in `requirements.txt`
- Outputs timestamped in `outputs/run_logs.txt`

---

## Limitations & Future Work

### Current Limitations
1. **Neutral Sentiment**: SiEBERT cannot predict neutral (binary classifier)
2. **PyABSA Coverage**: Extracts aspects for ~15-20% of reviews (high precision, low recall)
3. **Translation QA**: Back-translation check on sample only (not full dataset)
4. **Topic Labels**: Heuristic-based (not LLM-generated)

### Future Enhancements
- Add 3-class sentiment model with neutral support
- Integrate LLM for topic interpretation (if allowed)
- Implement active learning for aspect extraction
- Add time-series analysis for review trends

---

## Technical Details

### Star Rating Parser
Handles multiple formats:
- Numeric: "4.5 out of 5", "3/5"
- Word-based: "three stars", "five star rating"
- Mixed: "rated 4 stars"

### Text Normalization Pipeline
1. Lowercase conversion
2. Emoji → text (demojization)
3. URL/HTML/handle stripping
4. Contraction expansion ("don't" → "do not")
5. Lemmatization (spaCy)

### Topic Modeling
- **Vectorization**: TF-IDF (1-2 grams, min_df=5, max 10k features)
- **Clustering**: KMeans (k=6-12, silhouette optimization)
- **Labeling**: Semantic rules based on keywords + avg star rating

### Aspect Extraction
- **Dictionary**: Regex + keyword matching (13 aspect categories)
- **PyABSA**: DeBERTa-v3-based ATEPC model (aspect term extraction + polarity classification)

---

## Troubleshooting

### Installation Issues
```bash
# If spaCy model fails to download:
python -m pip install https://github.com/explosion/spacy-models/releases/download/en_core_web_sm-3.8.0/en_core_web_sm-3.8.0.tar.gz

# If PyTorch CPU-only needed:
pip install torch --index-url https://download.pytorch.org/whl/cpu
```

### Runtime Errors
- **Out of Memory**: Reduce batch sizes in `config.yaml` or use GPU
- **PyABSA Fails**: Pipeline continues with fallback (dictionary-only aspects)
- **Missing Files**: Ensure `final_extracted.csv` is in project root or `data/raw/`

---

## Citation & Credits

**Models Used**:
- SiEBERT: `siebert/sentiment-roberta-large-english` (Hugging Face)
- PyABSA: DeBERTa-v3-based ATEPC (v2.x)
- M2M100: `facebook/m2m100_418M` (Meta)
- spaCy: `en_core_web_sm` v3.8

**Libraries**:
- Transformers (Hugging Face)
- PyABSA (Aspect-Based Sentiment Analysis)
- VADER Sentiment (rule-based)
- scikit-learn (clustering, metrics)

---

## Debug

For issues, questions, or improvements:
- Check `outputs/run_logs.txt` for detailed execution logs
- Review `config.yaml` for parameter tuning
- Examine intermediate parquet files in `data/processed/` for debugging

**Project Status**: Production-ready, fully documented, reproducible.

---

**Last Updated**: October 2024  
**Python Version**: 3.11+  
**License**: Academic/Educational Use
