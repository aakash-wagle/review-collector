"""
Insight Generation Agent
Generates actionable design recommendations using LLM reasoning
"""

import pandas as pd
from typing import Dict, Any, List


class InsightAgent:
    """Agent responsible for generating actionable insights and recommendations"""
    
    def __init__(self, llm=None, use_llm: bool = False):
        self.llm = llm
        self.insights = {}
        self.use_llm = False  # Disabled for standalone operation
        
    def generate_recommendation_template(self):
        """Create prompt template for generating recommendations"""
        template = """
You are a product design expert analyzing customer reviews for the iPhone 17 Pro Max.

Based on the following analysis data, generate 3-5 concrete, actionable design improvement recommendations.

Review Statistics:
- Total Reviews: {total_reviews}
- Average Rating: {avg_rating}
- Sentiment Distribution: {sentiment_dist}

Key Aspects Analyzed:
{aspect_summary}

Pain Points Identified:
{pain_points}

Positive Aspects:
{opportunities}

For each recommendation:
1. Be specific and actionable
2. Reference supporting evidence from the data
3. Explain the expected impact
4. Consider feasibility

Format each recommendation as:
**Recommendation [N]: [Title]**
- **Evidence**: [What the data shows]
- **Action**: [Specific design change]
- **Expected Impact**: [How this helps users]

Generate recommendations now:
"""
        return template
    
    def generate_rule_based_recommendations(self, topic_results: Dict, sentiment_results: Dict, 
                                           stats: Dict) -> List[Dict[str, str]]:
        """Generate recommendations using rule-based logic"""
        recommendations = []
        
        aspect_analysis = topic_results['aspect_analysis']
        pain_points = topic_results['pain_points']
        opportunities = topic_results['opportunities']
        
        # Rule 1: Address top pain points
        for pp in pain_points[:2]:  # Top 2 pain points
            aspect = pp['aspect']
            aspect_data = aspect_analysis[aspect]
            
            rec_map = {
                'battery': {
                    'title': 'Enhance Battery Life Management',
                    'action': 'Implement adaptive power management algorithms that optimize battery performance during high-intensity tasks like gaming and video recording',
                    'impact': 'Reduce user complaints about battery drain and extend daily usage time'
                },
                'heating': {
                    'title': 'Improve Thermal Management System',
                    'action': 'Expand vapor chamber coverage and optimize thermal dissipation during charging and intensive use',
                    'impact': 'Reduce overheating complaints and improve sustained performance'
                },
                'setup': {
                    'title': 'Streamline Device Setup and Data Transfer',
                    'action': 'Redesign the setup wizard with better error handling, progress indicators, and automated app restoration',
                    'impact': 'Reduce setup frustration and improve first-use experience'
                },
                'price': {
                    'title': 'Introduce More Storage Tiers',
                    'action': 'Offer mid-range storage options (e.g., 384GB) at competitive price points',
                    'impact': 'Improve perceived value and capture price-sensitive customers'
                }
            }
            
            if aspect in rec_map:
                rec = rec_map[aspect]
                recommendations.append({
                    'title': rec['title'],
                    'evidence': f"{aspect_data['count']} reviews mentioned {aspect} with {pp['negative_ratio']*100:.1f}% expressing negative sentiment. Average sentiment: {pp['sentiment']:.2f}",
                    'action': rec['action'],
                    'impact': rec['impact'],
                    'priority': 'HIGH'
                })
        
        # Rule 2: Leverage strengths
        for opp in opportunities[:2]:
            aspect = opp['aspect']
            aspect_data = aspect_analysis[aspect]
            
            strength_map = {
                'camera': {
                    'title': 'Expand Camera Capabilities and Marketing',
                    'action': 'Introduce advanced photography modes and AI-enhanced features; highlight in marketing materials',
                    'impact': 'Capitalize on strong camera satisfaction to differentiate from competitors'
                },
                'performance': {
                    'title': 'Optimize for Performance-Heavy Use Cases',
                    'action': 'Create gaming and professional modes that maximize performance with custom thermal profiles',
                    'impact': 'Strengthen position in pro and gaming markets'
                },
                'design': {
                    'title': 'Expand Color and Finish Options',
                    'action': 'Offer additional bold color options and premium finishes (matte, glossy variants)',
                    'impact': 'Attract design-conscious users and improve visual differentiation'
                }
            }
            
            if aspect in strength_map and len(recommendations) < 5:
                rec = strength_map[aspect]
                recommendations.append({
                    'title': rec['title'],
                    'evidence': f"{aspect_data['count']} reviews mentioned {aspect} with {opp['positive_ratio']*100:.1f}% expressing positive sentiment. Average sentiment: {opp['sentiment']:.2f}",
                    'action': rec['action'],
                    'impact': rec['impact'],
                    'priority': 'MEDIUM'
                })
        
        # Rule 3: Address rating-sentiment mismatch
        if stats['rating_mean'] > 4.0 and sentiment_results['average_sentiment']['compound'] < 0.2:
            recommendations.append({
                'title': 'Improve User Experience Consistency',
                'action': 'Conduct user research to identify gaps between expectations and experience; focus on reducing friction points',
                'evidence': f"High average rating ({stats['rating_mean']:.2f}) but moderate sentiment score ({sentiment_results['average_sentiment']['compound']:.2f}) suggests room for experience improvements",
                'impact': 'Align actual experience with user expectations and reduce mixed reviews',
                'priority': 'MEDIUM'
            })
        
        return recommendations[:5]  # Return top 5
    
    def generate_insights(self, topic_results: Dict, sentiment_results: Dict, 
                         eda_results: Dict) -> Dict[str, Any]:
        """
        Generate actionable insights and recommendations
        
        Args:
            topic_results: Results from topic extraction
            sentiment_results: Results from sentiment analysis
            eda_results: Results from EDA
            
        Returns:
            Dictionary with insights and recommendations
        """
        try:
            stats = eda_results['statistics']
            
            # Generate recommendations
            if self.use_llm:
                # Use LLM for more sophisticated recommendations
                try:
                    prompt = self.generate_recommendation_template()
                    
                    # Prepare input data
                    aspect_summary = "\n".join([
                        f"- {aspect}: {data['count']} mentions, avg sentiment {data['avg_sentiment']:.2f}"
                        for aspect, data in sorted(
                            topic_results['aspect_analysis'].items(),
                            key=lambda x: x[1]['count'],
                            reverse=True
                        )[:10]
                    ])
                    
                    pain_points_str = "\n".join([
                        f"- {pp['aspect']}: {pp['mentions']} mentions, {pp['negative_ratio']*100:.1f}% negative"
                        for pp in topic_results['pain_points']
                    ])
                    
                    opportunities_str = "\n".join([
                        f"- {opp['aspect']}: {opp['mentions']} mentions, {opp['positive_ratio']*100:.1f}% positive"
                        for opp in topic_results['opportunities']
                    ])
                    
                    formatted_prompt = prompt.format(
                        total_reviews=stats['total_reviews'],
                        avg_rating=stats['rating_mean'],
                        sentiment_dist=sentiment_results['sentiment_distribution'],
                        aspect_summary=aspect_summary,
                        pain_points=pain_points_str,
                        opportunities=opportunities_str
                    )
                    
                    llm_response = self.llm.invoke(formatted_prompt)
                    llm_recommendations = llm_response.content
                except Exception as e:
                    print(f"LLM generation failed, falling back to rule-based: {e}")
                    llm_recommendations = None
            else:
                llm_recommendations = None
            
            # Generate rule-based recommendations (always as backup)
            rule_based_recommendations = self.generate_rule_based_recommendations(
                topic_results, sentiment_results, stats
            )
            
            # Summary statistics
            summary = {
                'total_reviews_analyzed': stats['total_reviews'],
                'average_rating': stats['rating_mean'],
                'overall_sentiment': sentiment_results['average_sentiment']['compound'],
                'key_strengths': [opp['aspect'] for opp in topic_results['opportunities'][:3]],
                'key_weaknesses': [pp['aspect'] for pp in topic_results['pain_points'][:3]],
                'sentiment_breakdown': sentiment_results['sentiment_distribution']
            }
            
            self.insights = {
                'summary': summary,
                'recommendations': rule_based_recommendations,
                'llm_recommendations': llm_recommendations
            }
            
            return {
                'status': 'success',
                'message': f'Generated {len(rule_based_recommendations)} actionable recommendations',
                'results': self.insights
            }
            
        except Exception as e:
            return {
                'status': 'error',
                'message': f'Error generating insights: {str(e)}',
                'results': None
            }
    
    def get_tool(self):
        """Return tool interface for insight generation"""
        return {
            'name': 'generate_insights',
            'func': lambda x: self.generate_insights(x.get('topics'), x.get('sentiment'), x.get('eda')),
            'description': 'Generates actionable insights and design recommendations.'
        }


def create_insight_tool(topic_results=None, sentiment_results=None, eda_results=None, use_llm=False):
    """
    Factory function to create insight generation tool
    
    Args:
        topic_results: Topic extraction results
        sentiment_results: Sentiment analysis results
        eda_results: EDA results
        use_llm: Whether to use LLM for generation
        
    Returns:
        InsightAgent instance
    """
    agent = InsightAgent(use_llm=use_llm)
    
    if all([topic_results, sentiment_results, eda_results]):
        result = agent.generate_insights(topic_results, sentiment_results, eda_results)
        if result['status'] == 'success':
            print(f"Insight Generation: {result['message']}")
    
    return agent

