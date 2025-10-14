"""
Report Generation Agent
Generates comprehensive markdown reports with findings and recommendations
"""

import pandas as pd
from typing import Dict, Any
from datetime import datetime


class ReportAgent:
    """Agent responsible for generating final reports"""
    
    def __init__(self):
        self.report = ""
        
    def format_recommendation(self, rec: Dict, index: int) -> str:
        """Format a single recommendation"""
        return f"""
### {index}. {rec['title']}

**Priority:** {rec.get('priority', 'MEDIUM')}

**Evidence from Data:**
{rec['evidence']}

**Recommended Action:**
{rec['action']}

**Expected Impact:**
{rec['impact']}

---
"""
    
    def generate_report(self, eda_results: Dict, sentiment_results: Dict, 
                       topic_results: Dict, insight_results: Dict,
                       stats_summary: Dict = None) -> Dict[str, Any]:
        """
        Generate comprehensive analysis report
        
        Args:
            eda_results: EDA results
            sentiment_results: Sentiment analysis results
            topic_results: Topic extraction results
            insight_results: Insight generation results
            stats_summary: Optional additional statistics
            
        Returns:
            Dictionary with report content
        """
        try:
            stats = eda_results['statistics']
            sentiment = sentiment_results['average_sentiment']
            sentiment_dist = sentiment_results['sentiment_distribution']
            aspects = topic_results['aspect_analysis']
            recommendations = insight_results['recommendations']
            
            # Generate report sections
            report_sections = []
            
            # Header
            report_sections.append(f"""# iPhone 17 Pro Max Review Analysis Report

**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

**Analysis Type:** Customer Review Sentiment & Topic Analysis

**Data Source:** Google Reviews

---
""")
            
            # Executive Summary
            top_aspects = sorted(aspects.items(), key=lambda x: x[1]['count'], reverse=True)[:5]
            top_aspect_names = [a[0].replace('_', ' ').title() for a in top_aspects]
            
            report_sections.append(f"""## Executive Summary

This report analyzes **{stats['total_reviews']:,}** customer reviews of the iPhone 17 Pro Max to identify design improvement opportunities and customer sentiment patterns.

### Key Findings:

- **Overall Rating:** {stats['rating_mean']:.2f}/5.0 (±{stats['rating_std']:.2f})
- **Sentiment Breakdown:**
  - Positive: {sentiment_dist.get('positive', 0):,} reviews ({sentiment_dist.get('positive', 0)/stats['total_reviews']*100:.1f}%)
  - Neutral: {sentiment_dist.get('neutral', 0):,} reviews ({sentiment_dist.get('neutral', 0)/stats['total_reviews']*100:.1f}%)
  - Negative: {sentiment_dist.get('negative', 0):,} reviews ({sentiment_dist.get('negative', 0)/stats['total_reviews']*100:.1f}%)
- **Average Sentiment Score:** {sentiment['compound']:.3f} (range: -1 to +1)
- **Most Discussed Aspects:** {', '.join(top_aspect_names)}

---
""")
            
            # Dataset Overview
            report_sections.append(f"""## 1. Dataset Overview

### Statistics
- **Total Reviews Analyzed:** {stats['total_reviews']:,}
- **Rating Distribution:**
""")
            
            for rating in sorted(stats['rating_distribution'].keys(), reverse=True):
                count = stats['rating_distribution'][rating]
                percentage = (count / stats['total_reviews']) * 100
                report_sections.append(f"  - {rating}★: {count:,} reviews ({percentage:.1f}%)")
            
            report_sections.append(f"""
- **Text Statistics:**
  - Average review length: {stats['avg_text_length']:.0f} characters
  - Shortest review: {stats['min_text_length']} characters
  - Longest review: {stats['max_text_length']} characters

---
""")
            
            # Sentiment Analysis
            report_sections.append(f"""## 2. Sentiment Analysis

### Overall Sentiment
The aggregate sentiment score is **{sentiment['compound']:.3f}**, indicating a **{
    'highly positive' if sentiment['compound'] > 0.5 else 
    'positive' if sentiment['compound'] > 0.1 else 
    'neutral' if sentiment['compound'] > -0.1 else 
    'negative'}** overall reception.

### Sentiment Components
- **Positive sentiment:** {sentiment['positive']:.1%}
- **Neutral sentiment:** {sentiment['neutral']:.1%}
- **Negative sentiment:** {sentiment['negative']:.1%}

### Sentiment by Rating
""")
            
            for rating in sorted(sentiment_results['sentiment_by_rating'].keys(), reverse=True):
                data = sentiment_results['sentiment_by_rating'][rating]
                report_sections.append(
                    f"- **{rating}★ reviews:** Avg sentiment = {data['mean']:.3f} ({int(data['count'])} reviews)"
                )
            
            report_sections.append("\n---\n")
            
            # Topic Analysis
            report_sections.append("""## 3. Product Aspect Analysis

### Key Aspects Discussed

The following table shows the most frequently mentioned product aspects and their associated sentiment:

| Aspect | Mentions | Avg Sentiment | Positive % | Negative % |
|--------|----------|---------------|------------|------------|
""")
            
            sorted_aspects = sorted(aspects.items(), key=lambda x: x[1]['count'], reverse=True)[:12]
            for aspect, data in sorted_aspects:
                if data['count'] > 0:
                    pos_pct = (data['positive_count'] / data['count'] * 100) if data['count'] > 0 else 0
                    neg_pct = (data['negative_count'] / data['count'] * 100) if data['count'] > 0 else 0
                    report_sections.append(
                        f"| {aspect.replace('_', ' ').title()} | {data['count']} | "
                        f"{data['avg_sentiment']:.2f} | {pos_pct:.1f}% | {neg_pct:.1f}% |"
                    )
            
            # Pain Points
            pain_points = topic_results['pain_points']
            if pain_points:
                report_sections.append(f"""

### Identified Pain Points

The following aspects received notably negative sentiment:
""")
                for i, pp in enumerate(pain_points, 1):
                    report_sections.append(
                        f"{i}. **{pp['aspect'].replace('_', ' ').title()}**: "
                        f"{pp['mentions']} mentions, {pp['negative_ratio']*100:.1f}% negative sentiment"
                    )
            
            # Strengths
            opportunities = topic_results['opportunities']
            if opportunities:
                report_sections.append(f"""

### Identified Strengths

The following aspects received notably positive sentiment:
""")
                for i, opp in enumerate(opportunities, 1):
                    report_sections.append(
                        f"{i}. **{opp['aspect'].replace('_', ' ').title()}**: "
                        f"{opp['mentions']} mentions, {opp['positive_ratio']*100:.1f}% positive sentiment"
                    )
            
            report_sections.append("\n---\n")
            
            # Design Recommendations
            report_sections.append("""## 4. Design Improvement Recommendations

Based on the comprehensive analysis of customer reviews, the following actionable design improvements are recommended:

""")
            
            for i, rec in enumerate(recommendations, 1):
                report_sections.append(self.format_recommendation(rec, i))
            
            # Methodology
            report_sections.append("""## 5. Methodology

### Data Processing Pipeline

1. **Data Ingestion**: Loaded and validated review data from CSV format
2. **Data Cleaning**: Normalized text, extracted numeric ratings, parsed dates, removed duplicates
3. **Exploratory Analysis**: Generated descriptive statistics, word frequencies, and temporal trends
4. **Sentiment Analysis**: Applied VADER sentiment analyzer to compute polarity scores
5. **Topic Extraction**: Identified product aspects using keyword-based pattern matching
6. **Insight Generation**: Correlated sentiment with topics to derive actionable recommendations
7. **Visualization**: Created comprehensive visual representations of findings

### Tools & Technologies

- **Language Models**: LangChain agent framework
- **Sentiment Analysis**: VADER (Valence Aware Dictionary and sEntiment Reasoner)
- **Data Processing**: Pandas, NumPy
- **Visualization**: Matplotlib, Seaborn, WordCloud

---
""")
            
            # Limitations
            report_sections.append("""## 6. Limitations and Future Work

### Limitations

1. **Sentiment Analysis**: Rule-based VADER may miss context-specific sentiment nuances
2. **Topic Extraction**: Keyword-based approach may not capture all implicit aspects
3. **Temporal Analysis**: Limited by relative date formats in source data
4. **Sample Bias**: Reviews may not represent all customer segments equally

### Future Enhancements

1. Implement transformer-based sentiment models (e.g., BERT) for improved accuracy
2. Apply advanced topic modeling (BERTopic, LDA) for automatic theme discovery
3. Conduct comparative analysis with competitor products
4. Integrate multi-lingual review analysis
5. Build predictive models for review helpfulness and rating estimation

---
""")
            
            # Footer
            report_sections.append(f"""## Conclusion

The iPhone 17 Pro Max demonstrates **strong overall reception** with an average rating of **{stats['rating_mean']:.2f}/5.0**. 

Key strengths include {', '.join([opp['aspect'].replace('_', ' ') for opp in opportunities[:2]])} while opportunities for improvement exist in {', '.join([pp['aspect'].replace('_', ' ') for pp in pain_points[:2]])}.

The {len(recommendations)} evidence-based recommendations provided in this report offer concrete paths for enhancing user experience and addressing customer pain points.

---

**End of Report**
""")
            
            # Combine all sections
            self.report = '\n'.join(report_sections)
            
            return {
                'status': 'success',
                'message': f'Generated comprehensive report ({len(self.report)} characters)',
                'report': self.report
            }
            
        except Exception as e:
            return {
                'status': 'error',
                'message': f'Error generating report: {str(e)}',
                'report': None
            }
    
    def save_report(self, filepath: str) -> bool:
        """Save report to file"""
        try:
            with open(filepath, 'w', encoding='utf-8') as f:
                f.write(self.report)
            return True
        except Exception as e:
            print(f"Error saving report: {e}")
            return False
    
    def get_tool(self):
        """Return tool interface for report generation"""
        return {
            'name': 'generate_report',
            'func': lambda x: self.generate_report(
                x.get('eda'), x.get('sentiment'), x.get('topics'), x.get('insights')
            ),
            'description': 'Generates comprehensive markdown report.'
        }


def create_report_tool(eda_results=None, sentiment_results=None, topic_results=None, insight_results=None):
    """
    Factory function to create report generation tool
    
    Args:
        eda_results: EDA results
        sentiment_results: Sentiment analysis results
        topic_results: Topic extraction results
        insight_results: Insight generation results
        
    Returns:
        ReportAgent instance
    """
    agent = ReportAgent()
    
    if all([eda_results, sentiment_results, topic_results, insight_results]):
        result = agent.generate_report(eda_results, sentiment_results, topic_results, insight_results)
        if result['status'] == 'success':
            print(f"\nReport Generation: {result['message']}")
    
    return agent

