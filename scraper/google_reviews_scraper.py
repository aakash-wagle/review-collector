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
RANDOM_TIMEOUT_MIN = 3000  # milliseconds
RANDOM_TIMEOUT_MAX = 10000  # milliseconds


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
    
    # Uncomment the line below if you want to run in headless mode
    # chrome_options.add_argument("--headless")
    
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


def scrape_all_reviews(driver, product_name):
    """Scrape all available reviews with no page limit"""
    all_reviews = []
    seen_reviews = set()  # Track unique reviews to avoid duplicates
    page_count = 0
    consecutive_failures = 0
    max_consecutive_failures = 3
    last_save_count = 0
    no_new_reviews_count = 0  # Track pages with no new reviews
    max_no_new_reviews = 3  # Stop after 3 pages with no new reviews
    
    # Generate filename with relative path to data/raw
    filename = "../data/raw/reviews.csv"
    
    print(f"\n{'='*60}")
    print(f"Starting to scrape reviews for: {product_name}")
    print(f"Output file: {filename}")
    print(f"Will save every {SAVE_INTERVAL} reviews")
    print(f"{'='*60}\n")
    
    while True:  # No page limit
        # Check if driver is still alive
        if not check_driver_alive(driver):
            print("\nBrowser session is no longer valid. Stopping scraper.")
            break
        
        page_count += 1
        print(f"\n{'='*60}")
        print(f"Scraping page {page_count}")
        print(f"{'='*60}")
        
        # Scrape reviews from current page
        page_reviews = scrape_reviews_from_page(driver)
        
        # Filter out duplicates by checking review text
        new_reviews_count = 0
        for review in page_reviews:
            review_text = review.get('review_text', '')
            if review_text and review_text not in seen_reviews:
                seen_reviews.add(review_text)
                all_reviews.append(review)
                new_reviews_count += 1
        
        print(f"\nStatistics:")
        print(f"   - New reviews from this page: {new_reviews_count}")
        print(f"   - Total unique reviews collected: {len(all_reviews)}")
        
        # Track consecutive pages with no new reviews
        if new_reviews_count == 0:
            no_new_reviews_count += 1
            print(f"No new reviews found ({no_new_reviews_count}/{max_no_new_reviews} consecutive)")
            if no_new_reviews_count >= max_no_new_reviews:
                print(f"\nStopping: No new reviews for {max_no_new_reviews} consecutive pages")
                break
        else:
            no_new_reviews_count = 0  # Reset counter
        
        # Save incrementally every SAVE_INTERVAL reviews
        if len(all_reviews) >= last_save_count + SAVE_INTERVAL:
            reviews_to_save = all_reviews[last_save_count:]
            mode = 'w' if last_save_count == 0 else 'a'
            save_reviews_to_csv(reviews_to_save, filename, mode=mode)
            last_save_count = len(all_reviews)
            print(f"Checkpoint: Saved {len(all_reviews)} total reviews to CSV")
        
        # Random timeout to avoid rate limiting
        timeout = random.randint(RANDOM_TIMEOUT_MIN, RANDOM_TIMEOUT_MAX) / 1000
        print(f"\nWaiting {timeout:.1f} seconds to avoid rate limiting...")
        time.sleep(timeout)
        
        # Try to find and click "More Reviews" for next page
        try:
            # Check driver is still alive before attempting pagination
            if not check_driver_alive(driver):
                print("\nBrowser session died. Stopping scraper.")
                break
            
            # Scroll down to ensure the "More Reviews" button is in viewport
            print("\nLooking for pagination button...")
            driver.execute_script("window.scrollTo(0, document.body.scrollHeight);")
            time.sleep(2)
            
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
                "//button[contains(text(), 'Show more')]",
                "//div[contains(text(), 'Show more')]",
                "//a[contains(text(), 'Show more')]",
                "//span[contains(text(), 'Load more')]",
                "//button[contains(text(), 'Load more')]"
            ]
            
            next_page_found = False
            for i, selector in enumerate(pagination_selectors):
                try:
                    next_button = WebDriverWait(driver, 3).until(
                        EC.element_to_be_clickable((By.XPATH, selector))
                    )
                    
                    # Scroll to the button to ensure it's visible
                    driver.execute_script("arguments[0].scrollIntoView(true);", next_button)
                    time.sleep(1)
                    
                    driver.execute_script("arguments[0].click();", next_button)
                    print("Successfully clicked pagination button")
                    next_page_found = True
                    consecutive_failures = 0  # Reset failure counter
                    break
                except TimeoutException:
                    continue
                except Exception as e:
                    continue
            
            if not next_page_found:
                consecutive_failures += 1
                print(f"No pagination button found (attempt {consecutive_failures}/{max_consecutive_failures})")
                
                if consecutive_failures >= max_consecutive_failures:
                    print("\nReached maximum consecutive failures, stopping pagination")
                    break
                else:
                    # Try scrolling more and waiting longer
                    print("Trying to scroll more and wait longer...")
                    driver.execute_script("window.scrollTo(0, document.body.scrollHeight);")
                    time.sleep(5)
                    
                    # Try one more time with longer wait
                    for selector in pagination_selectors[:3]:
                        try:
                            next_button = WebDriverWait(driver, 5).until(
                                EC.element_to_be_clickable((By.XPATH, selector))
                            )
                            driver.execute_script("arguments[0].scrollIntoView(true);", next_button)
                            time.sleep(1)
                            driver.execute_script("arguments[0].click();", next_button)
                            print("Successfully clicked pagination button after extended wait")
                            next_page_found = True
                            consecutive_failures = 0
                            break
                        except:
                            continue
                    
                    if not next_page_found:
                        print("Still no pagination button found after extended wait")
                        continue
                
        except Exception as e:
            error_msg = str(e)
            print(f"Error finding next page: {error_msg}")
            
            # Check if it's a session error
            if "invalid session id" in error_msg.lower() or "session" in error_msg.lower():
                print("Browser session is invalid. Stopping scraper gracefully.")
                break
            
            consecutive_failures += 1
            if consecutive_failures >= max_consecutive_failures:
                break
    
    # Final save of any remaining reviews
    if len(all_reviews) > last_save_count:
        reviews_to_save = all_reviews[last_save_count:]
        mode = 'a' if last_save_count > 0 else 'w'
        save_reviews_to_csv(reviews_to_save, filename, mode=mode)
    
    print(f"\n{'='*60}")
    print(f"SCRAPING COMPLETE!")
    print(f"{'='*60}")
    print(f"Total pages scraped: {page_count}")
    print(f"Total unique reviews: {len(all_reviews)}")
    print(f"Saved to: {filename}")
    
    # Provide reason for stopping
    if no_new_reviews_count >= max_no_new_reviews:
        print(f"\nStopped because: No new unique reviews found for {max_no_new_reviews} consecutive pages")
        print(f"   (This usually means all available reviews have been scraped)")
    elif consecutive_failures >= max_consecutive_failures:
        print(f"\nStopped because: Could not find pagination button for {max_consecutive_failures} consecutive attempts")
    elif not check_driver_alive(driver):
        print(f"\nStopped because: Browser session became invalid")
        print(f"   (This is normal - Google may have detected automation after ~100-150 reviews)")
    
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

