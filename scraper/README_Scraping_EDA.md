# Review Extraction Tool

This tool extracts review data from HTML dumps of Google review pages for Beats Solo3 headphones.

## Features

- Extracts reviewer names, ratings, dates, titles, and full review text
- Handles both short and full review text versions
- Cleans and processes the data
- Exports to CSV format
- Provides statistics and sample output

## Usage

1. **Install dependencies** (already done):
   ```bash
   uv add pandas beautifulsoup4
   ```

2. **Run the extraction script**:
   ```bash
   uv run extract_reviews.py
   ```

## Output

The script generates `reviews.csv` with the following columns:

- `reviewer_name`: Name of the reviewer
- `rating`: Star rating (1-5)
- `date`: When the review was posted (e.g., "10 months ago")
- `title`: Review title
- `review_text`: Full review text
- `source`: Where the review was posted (e.g., "Reviewed on walmart.com")
- `rating_numeric`: Numeric rating for analysis
- `word_count`: Number of words in the review
- `char_count`: Number of characters in the review

## Results Summary

From the test HTML file:
- **Total reviews extracted**: 10
- **Rating distribution**: 
  - 5 stars: 5 reviews
  - 4 stars: 2 reviews  
  - 3 stars: 2 reviews
  - 1 star: 1 review
- **Date range**: 8-11 months ago
- **Sources**: Mostly Walmart.com, some Influenster.com

## Technical Details

The script uses BeautifulSoup to parse HTML and extract data from specific CSS selectors:
- Review containers: `div[jsname="UtCV2e"][class="HDso9d"]`
- Reviewer names: `div.q7ToWe`
- Ratings: `span.yi40Hd.YrbPuc`
- Dates: `span.C8sUec` (from aria-label)
- Review text: `p.YdBXLd` (from full or short review sections)
