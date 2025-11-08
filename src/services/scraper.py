from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.common.exceptions import TimeoutException, NoSuchElementException
from bs4 import BeautifulSoup
import json
import time
from typing import List, Dict

class PrepackagedSolutionsScraper:
    def __init__(self):
        self.base_url = "https://www.shl.com/products/product-catalog/"
        
    def scrape_prepackaged_solutions(self) -> List[Dict]:
        """
        Scrape Pre-packaged Job Solutions (~12 pages, ~132+ products)
        """
        # Setup Chrome driver
        options = webdriver.ChromeOptions()
        options.add_argument('--headless')
        options.add_argument('--no-sandbox')
        options.add_argument('--disable-dev-shm-usage')
        options.add_argument('--disable-blink-features=AutomationControlled')
        options.add_argument('user-agent=Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36')
        
        driver = webdriver.Chrome(options=options)
        
        try:
            driver.get(self.base_url)
            print(f"Loading page: {self.base_url}")
            time.sleep(5)
            
            print("\n" + "="*70)
            print("SCRAPING PRE-PACKAGED JOB SOLUTIONS")
            print("="*70)
            
            all_assessments = []
            page_num = 1
            max_pages = 15  # Safety limit
            consecutive_empty = 0
            
            while page_num <= max_pages:
                print(f"\nScraping Page {page_num}...")
                
                try:
                    time.sleep(2)
                    
                    # Get current page HTML
                    soup = BeautifulSoup(driver.page_source, 'html.parser')
                    
                    # Find the FIRST table (Pre-packaged solutions)
                    tables = soup.find_all('table')
                    if len(tables) == 0:
                        print("No tables found")
                        break
                    
                    table = tables[0]  # First table is pre-packaged
                    
                    # Process table
                    assessments = self._process_table(table, page_num)
                    
                    if len(assessments) == 0:
                        consecutive_empty += 1
                        print(f"No products found on page {page_num}")
                        if consecutive_empty >= 3:
                            print("3 consecutive empty pages. Stopping.")
                            break
                    else:
                        consecutive_empty = 0
                        all_assessments.extend(assessments)
                        print(f"Found {len(assessments)} products (Total: {len(all_assessments)})")
                    
                    # Try to click Next button
                    if not self._click_next_button(driver):
                        print(f"No more pages. Finished at page {page_num}")
                        break
                    
                    page_num += 1
                    
                except Exception as e:
                    print(f"Error on page {page_num}: {e}")
                    break
            
            print(f"\n{'='*70}")
            print(f"COMPLETED: Pre-packaged Job Solutions")
            print(f"   Total Products: {len(all_assessments)}")
            print(f"   Pages Scraped: {page_num}")
            print(f"{'='*70}")
            
            return all_assessments
            
        except Exception as e:
            print(f"Error during scraping: {e}")
            import traceback
            traceback.print_exc()
            return []
        finally:
            driver.quit()
    
    def _click_next_button(self, driver) -> bool:
        """Find and click the Next button"""
        try:
            # Wait a moment for pagination to be ready
            time.sleep(1)
            
            # Strategy 1: Find "Next" text link
            next_buttons = driver.find_elements(By.XPATH, "//a[contains(text(), 'Next')]")
            
            if next_buttons:
                next_button = next_buttons[0]
                
                # Check if disabled
                classes = next_button.get_attribute('class') or ''
                if 'disabled' in classes.lower():
                    return False
                
                # Scroll and click
                driver.execute_script("arguments[0].scrollIntoView({block: 'center'});", next_button)
                time.sleep(0.5)
                driver.execute_script("arguments[0].click();", next_button)
                return True
            
            # Strategy 2: Find next page number
            pagination = driver.find_elements(By.CLASS_NAME, "pagination")
            if pagination:
                active = pagination[0].find_elements(By.CLASS_NAME, "active")
                if active:
                    try:
                        next_link = active[0].find_element(By.XPATH, "following-sibling::*[1]//a")
                        driver.execute_script("arguments[0].click();", next_link)
                        return True
                    except:
                        pass
            
            return False
            
        except Exception as e:
            print(f"   Error clicking next: {e}")
            return False
    
    def _process_table(self, table, page_num: int) -> List[Dict]:
        """Process table and extract products"""
        assessments = []
        rows = table.find_all('tr')[1:]  # Skip header
        
        for row in rows:
            cols = row.find_all('td')
            if len(cols) >= 4:
                product_link = cols[0].find('a')
                product_name = product_link.text.strip() if product_link else cols[0].text.strip()
                product_url = product_link.get('href', '') if product_link else ''
                
                if not product_name:
                    continue
                
                remote_testing = '✓' if (cols[1].text.strip() or cols[1].find('img') or cols[1].find('svg')) else ''
                adaptive_irt = '✓' if (cols[2].text.strip() or cols[2].find('img') or cols[2].find('svg')) else ''
                test_type = cols[3].text.strip()
                
                assessment = {
                    'category': 'Pre-packaged Job Solutions',
                    'name': product_name,
                    'url': product_url if product_url.startswith('http') else f"https://www.shl.com{product_url}",
                    'remote_testing': remote_testing,
                    'adaptive_irt': adaptive_irt,
                    'test_type': test_type,
                    'page_number': page_num,
                    'source': 'SHL Catalog'
                }
                assessments.append(assessment)
        
        return assessments
    
    def save_to_json(self, assessments: List[Dict], filepath: str):
        """Save to JSON"""
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(assessments, f, indent=2, ensure_ascii=False)
        print(f"Saved to {filepath}")
    
    def save_to_csv(self, assessments: List[Dict], filepath: str):
        """Save to CSV"""
        import csv
        if not assessments:
            return
        with open(filepath, 'w', newline='', encoding='utf-8') as f:
            fieldnames = ['category', 'name', 'url', 'remote_testing', 'adaptive_irt', 'test_type', 'page_number', 'source']
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(assessments)
        print(f"Saved to {filepath}")


if __name__ == "__main__":
    print("Starting Pre-packaged Solutions Scraper")
    
    scraper = PrepackagedSolutionsScraper()
    assessments = scraper.scrape_prepackaged_solutions()
    
    if assessments:
        scraper.save_to_json(assessments, 'prepackaged_solutions.json')
        scraper.save_to_csv(assessments, 'prepackaged_solutions.csv')
        print(f"\nSUCCESS: Scraped {len(assessments)} pre-packaged solutions")
    else:
        print("\nFAILED: No data scraped")
