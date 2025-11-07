from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.common.exceptions import TimeoutException, NoSuchElementException
from bs4 import BeautifulSoup
import json
import time
from typing import List, Dict

class IndividualTestSolutionsScraper:
    def __init__(self):
        self.base_url = "https://www.shl.com/products/product-catalog/"
        
    def scrape_individual_solutions(self) -> List[Dict]:
        """
        Scrape Individual Test Solutions (~32 pages, ~132+ products)
        """
        # Setup Chrome driver
        options = webdriver.ChromeOptions()
        # Remove headless to see what's happening
        # options.add_argument('--headless')
        options.add_argument('--no-sandbox')
        options.add_argument('--disable-dev-shm-usage')
        options.add_argument('--disable-blink-features=AutomationControlled')
        options.add_argument('user-agent=Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36')
        options.add_argument('--start-maximized')
        
        driver = webdriver.Chrome(options=options)
        
        try:
            driver.get(self.base_url)
            print(f"Loading page: {self.base_url}")
            time.sleep(5)
            
            # Scroll down to make sure both tables are loaded
            driver.execute_script("window.scrollTo(0, document.body.scrollHeight/2);")
            time.sleep(2)
            
            print("\n" + "="*70)
            print("SCRAPING INDIVIDUAL TEST SOLUTIONS")
            print("="*70)
            
            all_assessments = []
            page_num = 1
            max_pages = 35  # Safety limit
            consecutive_empty = 0
            
            while page_num <= max_pages:
                print(f"\nScraping Page {page_num}...")
                
                try:
                    time.sleep(2)
                    
                    # Get current page HTML
                    soup = BeautifulSoup(driver.page_source, 'html.parser')
                    
                    # Find ALL tables
                    tables = soup.find_all('table')
                    print(f"   Found {len(tables)} table(s) on page")
                    
                    # Find the Individual Test Solutions table
                    # It's the SECOND table or the one with technical product names
                    target_table = None
                    table_index = -1
                    
                    for idx, table in enumerate(tables):
                        rows = table.find_all('tr')[1:]  # Skip header
                        if rows:
                            first_row = rows[0]
                            cols = first_row.find_all('td')
                            if cols and cols[0].find('a'):
                                text = cols[0].get_text().strip()
                                # Individual tests have technical names with (New) or specific patterns
                                if any(indicator in text for indicator in 
                                      ['(New)', '.NET', 'Adobe', 'AI Skills', 'Android', 
                                       'Angular', 'Apache', 'Aeronautical', 'Aerospace', 'Agile']):
                                    target_table = table
                                    table_index = idx
                                    print(f"   Found Individual Test Solutions table at index {idx}")
                                    print(f"   First product: {text}")
                                    break
                    
                    if target_table is None:
                        # Fallback: Use second table if two tables exist
                        if len(tables) >= 2:
                            target_table = tables[1]
                            table_index = 1
                            print(f"   Using second table as fallback (index {table_index})")
                        elif len(tables) == 1:
                            target_table = tables[0]
                            table_index = 0
                            print(f"   Only one table found, using it (index {table_index})")
                        else:
                            print("No tables found")
                            break
                    
                    # Process table
                    assessments = self._process_table(target_table, page_num)
                    
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
                    
                    # CRITICAL: Click Next button for the INDIVIDUAL TEST table (second pagination)
                    if not self._click_next_button_for_individual_table(driver, table_index):
                        print(f"No more pages. Finished at page {page_num}")
                        break
                    
                    page_num += 1
                    
                except Exception as e:
                    print(f"Error on page {page_num}: {e}")
                    import traceback
                    traceback.print_exc()
                    break
            
            print(f"\n{'='*70}")
            print(f"COMPLETED: Individual Test Solutions")
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
    
    def _click_next_button_for_individual_table(self, driver, table_index: int) -> bool:
        """
        Find and click the Next button SPECIFICALLY for the Individual Test Solutions table
        This ensures we don't click the Pre-packaged table's Next button
        """
        try:
            time.sleep(1)
            
            # Find all pagination containers on the page
            paginations = driver.find_elements(By.CLASS_NAME, "pagination")
            print(f"   Found {len(paginations)} pagination container(s)")
            
            # If we have multiple paginations, we need to click the correct one
            # The pagination index matches the table index
            if len(paginations) > 1:
                # We have multiple tables with pagination
                # Use the SECOND pagination (index 1) for Individual Tests
                target_pagination_index = 1 if table_index == 1 else table_index
                
                if target_pagination_index < len(paginations):
                    pagination = paginations[target_pagination_index]
                    print(f"   Using pagination at index {target_pagination_index} for Individual Tests")
                    
                    # Find Next button within THIS specific pagination
                    next_buttons = pagination.find_elements(By.XPATH, ".//a[contains(text(), 'Next') or contains(text(), '›')]")
                    
                    if next_buttons:
                        next_button = next_buttons[0]
                        
                        # Check if disabled
                        classes = next_button.get_attribute('class') or ''
                        if 'disabled' in classes.lower():
                            print(f"   Next button is disabled (last page)")
                            return False
                        
                        # Scroll the pagination into view (not the button itself)
                        driver.execute_script("arguments[0].scrollIntoView({block: 'center'});", pagination)
                        time.sleep(0.5)
                        
                        # Click using JavaScript
                        print(f"   Clicking Next button for Individual Test Solutions")
                        driver.execute_script("arguments[0].click();", next_button)
                        time.sleep(2)
                        return True
                    else:
                        print(f"   No Next button found in pagination {target_pagination_index}")
                        return False
                else:
                    print(f"   Pagination index {target_pagination_index} out of range")
                    return False
                    
            elif len(paginations) == 1:
                # Only one table visible, use its pagination
                pagination = paginations[0]
                print(f"   Only one pagination found, using it")
                
                next_buttons = pagination.find_elements(By.XPATH, ".//a[contains(text(), 'Next') or contains(text(), '›')]")
                
                if next_buttons:
                    next_button = next_buttons[0]
                    
                    classes = next_button.get_attribute('class') or ''
                    if 'disabled' in classes.lower():
                        print(f"   Next button is disabled")
                        return False
                    
                    driver.execute_script("arguments[0].scrollIntoView({block: 'center'});", pagination)
                    time.sleep(0.5)
                    driver.execute_script("arguments[0].click();", next_button)
                    time.sleep(2)
                    return True
            
            print("   No suitable pagination found")
            return False
            
        except Exception as e:
            print(f"   Error clicking next: {e}")
            import traceback
            traceback.print_exc()
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
                    'category': 'Individual Test Solutions',
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
    print("Starting Individual Test Solutions Scraper")
    
    scraper = IndividualTestSolutionsScraper()
    assessments = scraper.scrape_individual_solutions()
    
    if assessments:
        scraper.save_to_json(assessments, 'individual_solutions.json')
        scraper.save_to_csv(assessments, 'individual_solutions.csv')
        print(f"\nSUCCESS: Scraped {len(assessments)} individual test solutions")
    else:
        print("\nFAILED: No data scraped")
