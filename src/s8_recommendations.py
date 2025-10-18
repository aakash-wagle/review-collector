"""
S8: Evidence-Backed Recommendations (Deterministic)
"""
import pandas as pd
import numpy as np
import yaml
import logging
from pathlib import Path
from typing import List, Dict

logger = logging.getLogger(__name__)


class RecommendationEngine:
    """Generate evidence-backed recommendations."""
    
    def __init__(self, df: pd.DataFrame, dict_aspects: pd.DataFrame, pyabsa_aspects: pd.DataFrame):
        self.df = df
        self.dict_aspects = dict_aspects
        self.pyabsa_aspects = pyabsa_aspects
    
    def rank_problem_aspects(self, top_n: int = 5) -> List[Dict]:
        """Rank aspects by problem severity (negative share + support)."""
        # Combine both methods, weighted by PyABSA
        aspects_ranked = []
        
        for _, row in self.pyabsa_aspects.iterrows():
            aspect = row['aspect']
            
            # Find corresponding dict data
            dict_row = self.dict_aspects[self.dict_aspects['aspect'] == aspect]
            
            pyabsa_neg_share = row['neg_share']
            pyabsa_count = row['count']
            
            dict_neg_share = dict_row['neg_share'].iloc[0] if len(dict_row) > 0 else 0
            dict_count = dict_row['count'].iloc[0] if len(dict_row) > 0 else 0
            
            # Compute severity score: weighted avg of neg_share * support
            severity = (pyabsa_neg_share * 0.6 + dict_neg_share * 0.4) * np.log(pyabsa_count + dict_count + 1)
            
            aspects_ranked.append({
                'aspect': aspect,
                'severity': severity,
                'pyabsa_neg_share': pyabsa_neg_share,
                'dict_neg_share': dict_neg_share,
                'pyabsa_count': pyabsa_count,
                'dict_count': dict_count,
                'avg_stars': row['avg_stars']
            })
        
        # Sort by severity
        aspects_ranked = sorted(aspects_ranked, key=lambda x: x['severity'], reverse=True)
        
        return aspects_ranked[:top_n]
    
    def extract_evidence(self, aspect: str, sentiment: str = 'negative', max_examples: int = 3) -> List[str]:
        """Extract evidence quotes for an aspect."""
        evidence = []
        
        # Find reviews mentioning this aspect with negative sentiment
        for idx, aspects_raw in enumerate(self.df['aspects_pyabsa_raw']):
            if not isinstance(aspects_raw, list):
                continue
            
            for asp in aspects_raw:
                if asp['aspect'] == aspect and asp['sentiment'] == sentiment:
                    # Get the evidence span
                    evidence_text = asp.get('evidence_span', '')
                    if evidence_text and len(evidence_text) > 10:
                        evidence.append(evidence_text.strip())
                    
                    if len(evidence) >= max_examples:
                        break
            
            if len(evidence) >= max_examples:
                break
        
        # Fallback to dictionary-based extraction if PyABSA didn't find enough
        if len(evidence) < max_examples:
            # Find reviews with low stars mentioning this aspect
            aspect_reviews = self.df[
                (self.df['stars_num'] <= 2) & 
                (self.df['aspects_dict'].apply(lambda x: aspect in x if isinstance(x, list) else False))
            ]
            
            for _, row in aspect_reviews.head(max_examples - len(evidence)).iterrows():
                text = row['text_en_clean']
                # Extract first sentence mentioning relevant keywords
                sentences = text.split('.')
                for sent in sentences:
                    if any(keyword in sent.lower() for keyword in self._get_aspect_keywords(aspect)):
                        evidence.append(sent.strip())
                        break
        
        return evidence[:max_examples]
    
    @staticmethod
    def _get_aspect_keywords(aspect: str) -> List[str]:
        """Get keywords for aspect."""
        keyword_map = {
            'Sound/Audio': ['sound', 'audio', 'bass'],
            'Comfort/Fit': ['comfort', 'fit', 'tight', 'hurt', 'ear'],
            'Battery/Longevity': ['battery', 'charge', 'life'],
            'Connectivity/Bluetooth': ['bluetooth', 'connect', 'wireless'],
            'Build/Durability': ['build', 'quality', 'break', 'broken'],
            'Mic/Calls': ['mic', 'call', 'voice'],
            'Noise Isolation': ['noise', 'leak', 'isolation']
        }
        return keyword_map.get(aspect, [aspect.lower()])
    
    def generate_recommendation(self, aspect_data: Dict) -> Dict:
        """Generate a recommendation for an aspect."""
        aspect = aspect_data['aspect']
        neg_share = aspect_data['pyabsa_neg_share']
        count = aspect_data['pyabsa_count']
        avg_stars = aspect_data['avg_stars']
        
        # Extract evidence
        evidence_quotes = self.extract_evidence(aspect, sentiment='negative', max_examples=2)
        
        # Generate finding, recommendation, and metric
        finding = self._generate_finding(aspect, neg_share, count, avg_stars)
        recommendation = self._generate_recommendation_text(aspect, neg_share)
        metric = self._generate_metric_target(aspect, neg_share)
        
        return {
            'aspect': aspect,
            'finding': finding,
            'evidence': evidence_quotes,
            'recommendation': recommendation,
            'metric_target': metric
        }
    
    @staticmethod
    def _generate_finding(aspect: str, neg_share: float, count: int, avg_stars: float) -> str:
        """Generate finding text."""
        return f"{aspect}: {neg_share*100:.1f}% negative sentiment in {count} reviews (avg {avg_stars:.2f} stars)"
    
    @staticmethod
    def _generate_recommendation_text(aspect: str, neg_share: float) -> str:
        """Generate recommendation text based on aspect."""
        recommendations_map = {
            'Comfort/Fit': "Redesign ear padding with softer memory foam and adjustable headband tension to reduce pressure on ears and head. Consider offering multiple size options.",
            'Sound/Audio': "Enhance sound quality through improved driver tuning, focusing on balanced audio profile and clearer mid-range frequencies. Reduce bass distortion at high volumes.",
            'Battery/Longevity': "Optimize power management firmware to extend battery life. Provide clearer battery status indicators and faster charging capabilities.",
            'Connectivity/Bluetooth': "Update Bluetooth firmware to improve connection stability across devices. Enhance pairing process and reduce latency issues.",
            'Build/Durability': "Reinforce hinge and headband structural components with more durable materials. Improve quality control for manufacturing defects.",
            'Mic/Calls': "Upgrade microphone quality with noise-canceling technology. Improve voice pickup sensitivity for clearer call quality.",
            'Noise Isolation': "Improve passive noise isolation through better ear cup seal design. Reduce sound leakage with enhanced acoustic engineering.",
            'Charging/Port': "Switch to USB-C for universal compatibility. Include charging cable and brick in package. Add quick-charge feature.",
            'Price/Value': "Reconsider pricing strategy or bundle with premium accessories to improve perceived value proposition."
        }
        
        return recommendations_map.get(aspect, f"Investigate and address user concerns related to {aspect} through targeted product improvements.")
    
    @staticmethod
    def _generate_metric_target(aspect: str, current_neg_share: float) -> str:
        """Generate metric target."""
        target_neg_share = max(0.05, current_neg_share * 0.5)  # Aim to cut negative share in half
        return f"Reduce negative sentiment from {current_neg_share*100:.1f}% to <{target_neg_share*100:.1f}% within 6 months (measure via post-improvement review sentiment)"


def format_recommendations_markdown(recommendations: List[Dict], output_path: Path):
    """Format recommendations as markdown."""
    md_content = "# Evidence-Backed Recommendations for Beats Solo3 Wireless\n\n"
    md_content += "_Generated from multilingual review analysis using local NLP models (no LLM)_\n\n"
    md_content += "---\n\n"
    
    for i, rec in enumerate(recommendations, 1):
        md_content += f"## Recommendation {i}: {rec['aspect']}\n\n"
        
        md_content += f"### Finding\n{rec['finding']}\n\n"
        
        md_content += f"### Evidence\n"
        for j, quote in enumerate(rec['evidence'], 1):
            md_content += f"{j}. \"{quote[:150]}...\"\n"
        md_content += "\n"
        
        md_content += f"### Recommendation\n{rec['recommendation']}\n\n"
        
        md_content += f"### Metric Target\n{rec['metric_target']}\n\n"
        
        md_content += "---\n\n"
    
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(md_content)
    
    logger.info(f"Recommendations saved to {output_path}")


def run_s8_recommendations(config_path: str = "config.yaml"):
    """
    S8: Recommendations Pipeline
    - Rank problem aspects
    - Extract evidence
    - Generate 3-5 recommendations with metrics
    """
    # Load configuration
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    # Load data
    df = pd.read_parquet(Path(config['data']['processed_dir']) / '06_evaluation.parquet')
    
    outputs_dir = Path(config['data']['outputs_dir'])
    dict_aspects = pd.read_csv(outputs_dir / 'tables' / 'aspects_summary_dict.csv')
    pyabsa_aspects = pd.read_csv(outputs_dir / 'tables' / 'aspects_summary_pyabsa.csv')
    
    logger.info(f"Loaded {len(df)} reviews for recommendation generation")
    
    # Initialize engine
    engine = RecommendationEngine(df, dict_aspects, pyabsa_aspects)
    
    # Rank problem aspects
    logger.info("Ranking problem aspects...")
    top_aspects = engine.rank_problem_aspects(top_n=5)
    
    # Generate recommendations
    logger.info("Generating recommendations...")
    recommendations = []
    for aspect_data in top_aspects:
        rec = engine.generate_recommendation(aspect_data)
        recommendations.append(rec)
        logger.info(f"Generated recommendation for: {rec['aspect']}")
    
    # Save recommendations
    output_path = outputs_dir / 'recommendations.md'
    format_recommendations_markdown(recommendations, output_path)
    
    # Also save as CSV for analysis
    rec_df = pd.DataFrame([
        {
            'aspect': r['aspect'],
            'finding': r['finding'],
            'recommendation': r['recommendation'],
            'metric_target': r['metric_target'],
            'evidence_count': len(r['evidence'])
        }
        for r in recommendations
    ])
    
    csv_path = outputs_dir / 'tables' / 'recommendations.csv'
    rec_df.to_csv(csv_path, index=False)
    logger.info(f"Recommendations CSV saved to {csv_path}")
    
    logger.info(f"\nS8 Summary: Generated {len(recommendations)} evidence-backed recommendations")
    
    return recommendations


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    recommendations = run_s8_recommendations()
    print(f"\nGenerated {len(recommendations)} recommendations")

