"""
Google Reviews Scraper
Scrapes Google reviews for products using Selenium WebDriver.
Saves reviews incrementally every 1000 reviews to avoid data loss.
"""

import time
import random
import csv
import pandas as pd
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.chrome.options import Options
from selenium.common.exceptions import TimeoutException, NoSuchElementException
import warnings
warnings.filterwarnings('ignore')


# Configuration
PRODUCT = "Beats Solo3 On-Ear Wireless Headphones"  # Change this to scrape different products
SAVE_INTERVAL = 1000  # Save to CSV every N reviews
MAX_BUTTON_CLICKS = 600  # Number of times to click "More Reviews" before scraping
CLICK_TIMEOUT_MIN = 1000  # milliseconds - faster for just clicking
CLICK_TIMEOUT_MAX = 5000  # milliseconds - faster for just clicking


def setup_driver():
    """Set up Chrome WebDriver with appropriate options"""
    chrome_options = Options()
    chrome_options.add_argument("--no-sandbox")
    chrome_options.add_argument("--disable-dev-shm-usage")
    chrome_options.add_argument("--disable-blink-features=AutomationControlled")
    chrome_options.add_experimental_option("excludeSwitches", ["enable-automation"])
    chrome_options.add_experimental_option('useAutomationExtension', False)
    
    # Add user agent to appear more like a real browser
    chrome_options.add_argument("user-agent=Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36")
    
    # Window size
    chrome_options.add_argument("--window-size=1920,1080")
    
    # Headless mode - runs without opening browser window
    chrome_options.add_argument("--headless")
    chrome_options.add_argument("--disable-gpu")  # Sometimes needed for headless
    
    driver = webdriver.Chrome(options=chrome_options)
    driver.execute_script("Object.defineProperty(navigator, 'webdriver', {get: () => undefined})")
    
    return driver


def search_product(driver, product_name):
    """Open Google and search for the product"""
    # Navigate to Google
    driver.get("https://www.google.com")
    
    # Wait for search box and enter product name
    search_box = WebDriverWait(driver, 10).until(
        EC.presence_of_element_located((By.NAME, "q"))
    )
    
    search_box.clear()
    search_box.send_keys(product_name + " google reviews")
    search_box.submit()
    
    print(f"Searching for: {product_name}")
    time.sleep(3)
    return driver


def find_and_click_more_reviews(driver):
    """Find and click the 'More reviews' button in the User Reviews section"""
    try:
        # Wait a bit for the page to load completely
        time.sleep(3)
        
        # Look for various possible selectors for the "More reviews" button
        possible_selectors = [
            "//span[contains(text(), 'More reviews')]",
            "//button[contains(text(), 'More reviews')]",
            "//div[contains(text(), 'More reviews')]",
            "//a[contains(text(), 'More reviews')]",
            "//span[contains(text(), 'View all reviews')]",
            "//button[contains(text(), 'View all reviews')]"
        ]
        
        for selector in possible_selectors:
            try:
                more_reviews_button = WebDriverWait(driver, 5).until(
                    EC.element_to_be_clickable((By.XPATH, selector))
                )
                driver.execute_script("arguments[0].click();", more_reviews_button)
                print("Successfully clicked 'More reviews' button")
                return True
            except TimeoutException:
                continue
        
        print("Could not find 'More reviews' button")
        return False
        
    except Exception as e:
        print(f"Error clicking 'More reviews' button: {e}")
        return False


def scrape_reviews_from_page(driver):
    """Scrape reviews from the current page"""
    reviews = []
    
    try:
        # Wait for reviews to load
        time.sleep(2)
        
        # Use the specific XPath structure
        # Look for review containers using the jsname attribute
        review_selectors = [
            "//div[@jsname='I1taC']",
            "//div[contains(@class, 'TqzWd')]",
            "//div[contains(@data-ved, '')]"
        ]
        
        review_elements = []
        for selector in review_selectors:
            try:
                elements = driver.find_elements(By.XPATH, selector)
                if elements:
                    review_elements = elements
                    print(f"Found {len(elements)} reviews using selector: {selector}")
                    break
            except:
                continue
        
        if not review_elements:
            print("No review elements found")
            return reviews
        
        for i, review_element in enumerate(review_elements):
            try:
                review_data = {}
                
                # Extract review text - look for the full review text in the YdBXLd class
                text_selectors = [
                    ".//p[@class='YdBXLd']",
                    ".//p[contains(@class, 'YdBXLd')]",
                    ".//div[@jsname='QFmKA']//p",
                    ".//div[@jsname='an9Zef']//p"
                ]
                
                review_text = ""
                for text_selector in text_selectors:
                    try:
                        text_element = review_element.find_element(By.XPATH, text_selector)
                        review_text = text_element.text.strip()
                        if review_text:
                            break
                    except:
                        continue
                
                # Extract star rating - look for the aria-label with rating info
                star_selectors = [
                    ".//span[@class='z3HNkc']",
                    ".//span[contains(@aria-label, 'Rated')]",
                    ".//span[contains(@aria-label, 'out of 5')]"
                ]
                
                stars = ""
                for star_selector in star_selectors:
                    try:
                        star_element = review_element.find_element(By.XPATH, star_selector)
                        stars = star_element.get_attribute('aria-label') or star_element.text.strip()
                        if stars:
                            break
                    except:
                        continue
                
                # Extract date - look for the date span
                date_selectors = [
                    ".//span[@class='C8sUec']",
                    ".//span[contains(@aria-label, 'Review submitted')]",
                    ".//span[contains(@aria-label, 'ago')]"
                ]
                
                date = ""
                for date_selector in date_selectors:
                    try:
                        date_element = review_element.find_element(By.XPATH, date_selector)
                        date = date_element.get_attribute('aria-label') or date_element.text.strip()
                        if date:
                            break
                    except:
                        continue
                
                # Only add if we have at least the review text
                if review_text:
                    review_data = {
                        'review_text': review_text,
                        'stars': stars,
                        'date': date
                    }
                    reviews.append(review_data)
                
            except Exception as e:
                print(f"Error scraping review {i+1}: {e}")
                continue
        
        print(f"Successfully scraped {len(reviews)} reviews from current page")
        
    except Exception as e:
        print(f"Error in scrape_reviews_from_page: {e}")
    
    return reviews


def save_reviews_to_csv(reviews, filename, mode='w'):
    """Save scraped reviews to CSV file"""
    if not reviews:
        print("No reviews to save")
        return
    
    df = pd.DataFrame(reviews)
    
    # Write header only if creating new file
    header = mode == 'w'
    df.to_csv(filename, mode=mode, index=False, encoding='utf-8', header=header)
    
    print(f"Saved {len(reviews)} reviews to {filename} (mode: {mode})")


def check_driver_alive(driver):
    """Check if the driver session is still alive"""
    try:
        driver.current_url
        return True
    except:
        return False


def click_more_reviews_multiple_times(driver, max_clicks):
    """Click 'More Reviews' button multiple times to load all reviews first"""
    print(f"\n{'='*60}")
    print(f"PHASE 1: Loading all reviews")
    print(f"Will click 'More Reviews' button up to {max_clicks} times")
    print(f"{'='*60}\n")
    
    clicks_count = 0
    consecutive_failures = 0
    max_consecutive_failures = 3
    
    for click_num in range(1, max_clicks + 1):
        # Check if driver is still alive
        if not check_driver_alive(driver):
            print("\nBrowser session is no longer valid. Stopping clicks.")
            break
        
        print(f"\nClick attempt {click_num}/{max_clicks}")
        
        try:
            # Scroll down to ensure the "More Reviews" button is in viewport
            driver.execute_script("window.scrollTo(0, document.body.scrollHeight);")
            time.sleep(1)
            
            # Use the specific XPath structure
            pagination_selectors = [
                "/html/body/div[19]/div[2]/div[2]/div/div[2]/div[2]/button",
                "//div[2]/div[2]/button",
                "//button[contains(@class, '')]",
                "//span[contains(text(), 'More reviews')]",
                "//button[contains(text(), 'More reviews')]",
                "//div[contains(text(), 'More reviews')]",
                "//a[contains(text(), 'More reviews')]",
                "//span[contains(text(), 'Show more')]",
                "//button[contains(text(), 'Show more')]"
            ]
            
            button_found = False
            for i, selector in enumerate(pagination_selectors):
                try:
                    next_button = WebDriverWait(driver, 2).until(
                        EC.element_to_be_clickable((By.XPATH, selector))
                    )
                    
                    # Scroll to the button to ensure it's visible
                    driver.execute_script("arguments[0].scrollIntoView(true);", next_button)
                    time.sleep(0.5)
                    
                    driver.execute_script("arguments[0].click();", next_button)
                    clicks_count += 1
                    consecutive_failures = 0  # Reset failure counter
                    button_found = True
                    print(f"   Successfully clicked button (Total clicks: {clicks_count})")
                    break
                except (TimeoutException, Exception):
                    continue
            
            if not button_found:
                consecutive_failures += 1
                print(f"   Button not found (Failure {consecutive_failures}/{max_consecutive_failures})")
                
                if consecutive_failures >= max_consecutive_failures:
                    print(f"\nNo more reviews to load after {clicks_count} clicks")
                    break
                
                # Try scrolling more
                driver.execute_script("window.scrollTo(0, document.body.scrollHeight);")
                time.sleep(2)
                continue
            
            # Random timeout between clicks (1-5 seconds)
            timeout = random.randint(CLICK_TIMEOUT_MIN, CLICK_TIMEOUT_MAX) / 1000
            print(f"   Waiting {timeout:.1f}s before next click...")
            time.sleep(timeout)
            
        except Exception as e:
            error_msg = str(e)
            print(f"   Error: {error_msg}")
            
            # Check if it's a session error
            if "invalid session id" in error_msg.lower() or "session" in error_msg.lower():
                print("Browser session is invalid. Stopping clicks.")
                break
            
            consecutive_failures += 1
            if consecutive_failures >= max_consecutive_failures:
                break
    
    print(f"\n{'='*60}")
    print(f"PHASE 1 COMPLETE: Clicked button {clicks_count} times")
    print(f"{'='*60}\n")
    
    return clicks_count


def scrape_all_reviews(driver, product_name):
    """Scrape all available reviews after loading them all"""
    all_reviews = []
    seen_reviews = set()  # Track unique reviews to avoid duplicates
    last_save_count = 0
    
    # Generate filename with relative path to data/raw
    filename = "../data/raw/reviews.csv"
    
    print(f"\n{'='*60}")
    print(f"Starting to scrape reviews for: {product_name}")
    print(f"Output file: {filename}")
    print(f"Will save every {SAVE_INTERVAL} reviews")
    print(f"{'='*60}\n")
    
    # PHASE 1: Click "More Reviews" button many times to load all reviews
    total_clicks = click_more_reviews_multiple_times(driver, MAX_BUTTON_CLICKS)
    
    # PHASE 2: Scrape all loaded reviews in one go
    print(f"\n{'='*60}")
    print(f"PHASE 2: Scraping all loaded reviews")
    print(f"{'='*60}\n")
    
    # Check if driver is still alive
    if not check_driver_alive(driver):
        print("\nBrowser session is no longer valid. Cannot scrape.")
        return all_reviews, filename
    
    # Scrape all reviews from the page
    page_reviews = scrape_reviews_from_page(driver)
    
    # Filter out duplicates by checking review text
    for review in page_reviews:
        review_text = review.get('review_text', '')
        if review_text and review_text not in seen_reviews:
            seen_reviews.add(review_text)
            all_reviews.append(review)
    
    print(f"\nStatistics:")
    print(f"   - Total reviews scraped: {len(page_reviews)}")
    print(f"   - Unique reviews: {len(all_reviews)}")
    print(f"   - Duplicates removed: {len(page_reviews) - len(all_reviews)}")
    
    # Save reviews in batches
    if len(all_reviews) > 0:
        print(f"\nSaving reviews to CSV...")
        
        # Save in chunks of SAVE_INTERVAL
        for i in range(0, len(all_reviews), SAVE_INTERVAL):
            chunk = all_reviews[i:i + SAVE_INTERVAL]
            mode = 'w' if i == 0 else 'a'
            save_reviews_to_csv(chunk, filename, mode=mode)
            print(f"   Saved chunk {i//SAVE_INTERVAL + 1}: {len(chunk)} reviews")
    
    print(f"\n{'='*60}")
    print(f"SCRAPING COMPLETE!")
    print(f"{'='*60}")
    print(f"Button clicks in Phase 1: {total_clicks}")
    print(f"Total unique reviews: {len(all_reviews)}")
    print(f"Saved to: {filename}")
    print(f"{'='*60}\n")
    
    return all_reviews, filename


def main():
    """Main function to run the scraper"""
    driver = None
    
    try:
        # Step 1: Initialize driver
        print("Initializing Selenium WebDriver...")
        driver = setup_driver()
        print("Selenium WebDriver initialized successfully!\n")
        
        # Step 2 & 3: Search for product
        driver = search_product(driver, PRODUCT)
        
        # Step 4: Click initial "More reviews" button
        more_reviews_clicked = find_and_click_more_reviews(driver)
        if not more_reviews_clicked:
            print("Warning: Could not find initial 'More reviews' button")
            print("Attempting to scrape visible reviews anyway...\n")
        
        # Step 5-7: Scrape all reviews with pagination
        all_reviews, filename = scrape_all_reviews(driver, PRODUCT)
        
        # Display summary
        if all_reviews:
            df = pd.DataFrame(all_reviews)
            print("\nFirst 5 reviews:")
            print(df.head())
            print(f"\nReview statistics:")
            print(f"   - Total reviews: {len(df)}")
            print(f"   - Columns: {list(df.columns)}")
        
    except KeyboardInterrupt:
        print("\n\nScraping interrupted by user (Ctrl+C)")
        print("Saving collected reviews before exiting...")
        
    except Exception as e:
        print(f"\nError occurred: {e}")
        import traceback
        traceback.print_exc()
        
    finally:
        # Clean up - Close the browser
        if driver:
            try:
                driver.quit()
                print("\nBrowser closed successfully")
            except Exception as e:
                print(f"Error closing browser: {e}")


if __name__ == "__main__":
    main()

