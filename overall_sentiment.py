"""
Overall sentiment scoring and distribution analysis
"""

def compute_overall_score(ratings):
    """
    Compute overall score from ratings list
    
    Args:
        ratings: List of numerical ratings
        
    Returns:
        dict with score metrics
    """
    if not ratings:
        return {}
    
    ratings = [r for r in ratings if r and str(r).replace('.', '').isdigit()]
    
    if not ratings:
        return {}
    
    avg_rating = sum(ratings) / len(ratings)
    percentage = (avg_rating / 5.0) * 100
    
    return {
        'out_of_5': round(avg_rating, 2),
        'percentage': round(percentage, 1),
        'total_count': len(ratings),
        'min': min(ratings),
        'max': max(ratings)
    }


def compute_sentiment_distribution(sentiments):
    """
    Compute distribution of sentiments
    
    Args:
        sentiments: List of sentiment labels
        
    Returns:
        dict with sentiment counts and percentages
    """
    if not sentiments:
        return {}
    
    total = len(sentiments)
    distribution = {}
    
    for sentiment in set(sentiments):
        count = sentiments.count(sentiment)
        percentage = round((count / total) * 100, 1)
        distribution[sentiment] = {
            'count': count,
            'percentage': percentage
        }
    
    return dict(sorted(distribution.items(), key=lambda x: x[1]['count'], reverse=True))


def generate_summary_metrics(aspect_df):
    """
    Generate summary metrics from aspect analysis
    
    Args:
        aspect_df: DataFrame with aspect sentiment results
        
    Returns:
        dict with summary statistics
    """
    if aspect_df.empty:
        return {
            'total_mentions': 0,
            'average_positive_ratio': 0,
            'best_aspect': {'name': 'N/A', 'score': 0},
            'worst_aspect': {'name': 'N/A', 'score': 0}
        }
    
    total_mentions = aspect_df['mention_count'].sum() if 'mention_count' in aspect_df.columns else 0
    
    if 'positive_ratio' in aspect_df.columns:
        avg_positive = aspect_df['positive_ratio'].mean()
    else:
        avg_positive = 0
    
    # Find best and worst aspects
    if 'positive_ratio' in aspect_df.columns:
        best_idx = aspect_df['positive_ratio'].idxmax()
        worst_idx = aspect_df['positive_ratio'].idxmin()
        
        best_aspect = {
            'name': aspect_df.loc[best_idx, 'aspect'] if 'aspect' in aspect_df.columns else 'N/A',
            'score': round(aspect_df.loc[best_idx, 'positive_ratio'], 1)
        }
        worst_aspect = {
            'name': aspect_df.loc[worst_idx, 'aspect'] if 'aspect' in aspect_df.columns else 'N/A',
            'score': round(aspect_df.loc[worst_idx, 'positive_ratio'], 1)
        }
    else:
        best_aspect = {'name': 'N/A', 'score': 0}
        worst_aspect = {'name': 'N/A', 'score': 0}
    
    return {
        'total_mentions': int(total_mentions),
        'average_positive_ratio': round(avg_positive, 1),
        'best_aspect': best_aspect,
        'worst_aspect': worst_aspect
    }