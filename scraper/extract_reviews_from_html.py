import pandas as pd
from bs4 import BeautifulSoup
import re
import os

# THIS FILE WAS WRITTEN AS A FAIL SAFE TO EXTRACT THE REVIEWS FROM THE HTML DUMP IF SCRAPER IS INTERRUPTED
# ONE SUCH HTML DUMP IS test.html

def extract_reviews_from_html(html_file_path):
    with open(html_file_path, 'r', encoding='utf-8') as file:
        html_content = file.read()
    
    soup = BeautifulSoup(html_content, 'html.parser')
    
    reviews = soup.find_all('div', {'jsname': 'UtCV2e', 'class': 'HDso9d'})
    
    extracted_data = []
    
    for review in reviews:
        try:
            rating_elem = review.find('span', class_='yi40Hd YrbPuc')
            rating_raw = rating_elem.text.strip() if rating_elem else 'N/A'
            
            if rating_raw != 'N/A':
                try:
                    rating_num = float(rating_raw)
                    rating = f"Rated {rating_num} out of 5"
                except ValueError:
                    rating = 'N/A'
            else:
                rating = 'N/A'
            
            date_elem = review.find('span', class_='C8sUec')
            date_text = 'N/A'
            if date_elem and 'aria-label' in date_elem.attrs:
                date_text = date_elem['aria-label'].replace('Review submitted ', '')
            
            full_review = review.find('div', id=re.compile('.*-full'))
            if full_review:
                review_text_elem = full_review.find('p', class_='YdBXLd')
            else:
                short_review = review.find('div', id=re.compile('.*-short'))
                review_text_elem = short_review.find('p', class_='YdBXLd') if short_review else None
            
            review_text = review_text_elem.text.strip() if review_text_elem else 'N/A'
            
            extracted_data.append({
                'review_text': review_text,
                'stars': rating,
                'date': date_text
            })
            
        except Exception as e:
            print(f"Error extracting review: {e}")
            continue
    
    df = pd.DataFrame(extracted_data)
    
    return df


def clean_and_process_data(df):
    df['rating_numeric'] = pd.to_numeric(df['rating'], errors='coerce')
    
    df['review_text'] = df['review_text'].str.replace('\n', ' ').str.strip()
    df['review_text'] = df['review_text'].str.replace(r'\s+', ' ', regex=True)
    
    df['title'] = df['title'].str.replace('\n', ' ').str.strip()
    df['title'] = df['title'].str.replace(r'\s+', ' ', regex=True)
    
    df['word_count'] = df['review_text'].str.split().str.len()
    
    df['char_count'] = df['review_text'].str.len()
    
    return df


def main():
    
    html_file = 'test.html'
    
    if not os.path.exists(html_file):
        print(f"Error: HTML file not found at {html_file}")
        return
    
    print("Extracting reviews from HTML...")
    reviews_df = extract_reviews_from_html(html_file)
    
    if reviews_df.empty:
        print("No reviews found in the HTML file.")
        return
    
    print("Cleaning and processing data...")
    # This line is commented out to match next stage's expected input shape, i.e., column names anda data format
    # reviews_df = clean_and_process_data(reviews_df)
    
    print(f"\nExtracted {len(reviews_df)} reviews")
    print("\nFirst few reviews:")
    print(reviews_df.head())
    
    output_file = '../data/raw/reviews.csv'
    reviews_df.to_csv(output_file, index=False, encoding='utf-8')
    print(f"\nReviews saved to: {output_file}")
    
    print("\n--- Statistics ---")
    print(f"Total reviews: {len(reviews_df)}")
    print(f"\nStars distribution:")
    print(reviews_df['stars'].value_counts().sort_index())
    print(f"\nDate distribution:")
    print(reviews_df['date'].value_counts())
    
    print("\n--- Sample Review Text ---")
    for i, row in reviews_df.head(3).iterrows():
        print(f"\nStars: {row['stars']} | Date: {row['date']}")
        print(f"Text: {row['review_text'][:200]}...")


if __name__ == "__main__":
    main()
