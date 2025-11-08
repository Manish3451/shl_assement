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
        self.base_url = "https://www.shl.com/products/product-catalog/?start=12&type=1"
        
    def scrape_individual_solutions(self) -> List[Dict]:
        """
        Scrape Individual Test Solutions (~32 pages, ~132+ products)
        WITH descriptions from each product page
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
            print("SCRAPING INDIVIDUAL TEST SOLUTIONS WITH DESCRIPTIONS")
            print("="*70)
            
            all_assessments = []
            page_num = 1
            max_pages = 35  # Safety limit
            consecutive_empty = 0
            
            while page_num <= max_pages:
                print(f"\n{'='*70}")
                print(f"PAGE {page_num}")
                print(f"{'='*70}")
                
                try:
                    time.sleep(2)
                    
                    # Get current page HTML
                    soup = BeautifulSoup(driver.page_source, 'html.parser')
                    
                    # Find ALL tables
                    tables = soup.find_all('table')
                    print(f"   Found {len(tables)} table(s) on page")
                    
                    # Find the Individual Test Solutions table
                    target_table = None
                    table_index = -1
                    
                    for idx, table in enumerate(tables):
                        rows = table.find_all('tr')[1:]  # Skip header
                        if rows:
                            first_row = rows[0]
                            cols = first_row.find_all('td')
                            if cols and cols[0].find('a'):
                                text = cols[0].get_text().strip()
                                # Individual tests have technical names
                                if any(indicator in text for indicator in 
                                      ['(New)', '.NET', 'Adobe', 'AI Skills', 'Android', 
                                       'Angular', 'Apache', 'Aeronautical', 'Aerospace', 'Agile']):
                                    target_table = table
                                    table_index = idx
                                    print(f"   ✓ Found Individual Test Solutions table at index {idx}")
                                    print(f"   ✓ First product: {text}")
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
                            print("   ✗ No tables found")
                            break
                    
                    # Process table WITH descriptions
                    assessments = self._process_table(target_table, page_num, driver)
                    
                    if len(assessments) == 0:
                        consecutive_empty += 1
                        print(f"   ✗ No products found on page {page_num}")
                        if consecutive_empty >= 3:
                            print("   ✗ 3 consecutive empty pages. Stopping.")
                            break
                    else:
                        consecutive_empty = 0
                        all_assessments.extend(assessments)
                        print(f"\n   ✓ Page {page_num} Complete: {len(assessments)} products")
                        print(f"   ✓ Total Collected: {len(all_assessments)} products")
                    
                    # Click Next button
                    if not self._click_next_button_for_individual_table(driver, table_index):
                        print(f"\n   ✓ No more pages. Finished at page {page_num}")
                        break
                    
                    page_num += 1
                    
                except Exception as e:
                    print(f"   ✗ Error on page {page_num}: {e}")
                    import traceback
                    traceback.print_exc()
                    break
            
            print(f"\n{'='*70}")
            print(f"SCRAPING COMPLETED")
            print(f"{'='*70}")
            print(f"   Total Products: {len(all_assessments)}")
            print(f"   Pages Scraped: {page_num}")
            print(f"   Products with Descriptions: {sum(1 for a in all_assessments if a.get('description'))}")
            print(f"{'='*70}\n")
            
            return all_assessments
            
        except Exception as e:
            print(f"✗ Error during scraping: {e}")
            import traceback
            traceback.print_exc()
            return []
        finally:
            driver.quit()
    
    def _scrape_product_description(self, driver, product_url: str, product_name: str) -> str:
        """
        Navigate to product page, extract description, then go back
        """
        try:
            # Save current URL to return later
            current_url = driver.current_url
            
            # Go to product page
            driver.get(product_url)
            time.sleep(2)
            
            # Parse the product page
            soup = BeautifulSoup(driver.page_source, 'html.parser')
            
            # Try multiple selectors for description
            description = ""
            
            # Strategy 1: Look for common description containers
            selectors = [
                {'class_': 'product-description'},
                {'class_': 'description'},
                {'class_': 'product-detail'},
                {'class_': 'product-content'},
                {'id': 'description'},
                {'class_': 'content-block'},
                {'class_': 'overview'},
                {'class_': 'summary'},
            ]
            
            for selector in selectors:
                elem = soup.find('div', selector)
                if elem:
                    description = elem.get_text(strip=True, separator=' ')
                    if len(description) > 50:  # Meaningful content
                        break
            
            # Strategy 2: Look for main content area
            if not description or len(description) < 50:
                main_content = soup.find('main') or soup.find('article')
                if main_content:
                    paragraphs = main_content.find_all('p')
                    description = ' '.join([p.get_text(strip=True) for p in paragraphs[:5]])
            
            # Strategy 3: Fallback - Get first few paragraphs
            if not description or len(description) < 50:
                paragraphs = soup.find_all('p')
                description = ' '.join([p.get_text(strip=True) for p in paragraphs[:5]])
            
            # Clean up description
            description = ' '.join(description.split())  # Normalize whitespace
            
            # Go back to catalog page
            driver.get(current_url)
            time.sleep(1.5)
            
            if description and len(description) > 20:
                print(f"         ✓ Got description ({len(description)} chars)")
                return description[:2000]  # Limit to 2000 chars
            else:
                print(f"         ✗ No description found")
                return ""
            
        except Exception as e:
            print(f"         ✗ Error getting description: {e}")
            # Try to go back to catalog
            try:
                driver.back()
                time.sleep(1)
            except:
                pass
            return ""
    
    def _process_table(self, table, page_num: int, driver) -> List[Dict]:
        """Process table and extract products WITH descriptions"""
        assessments = []
        rows = table.find_all('tr')[1:]  # Skip header
        
        print(f"\n   Processing {len(rows)} products on page {page_num}...")
        
        for idx, row in enumerate(rows, 1):
            cols = row.find_all('td')
            if len(cols) >= 4:
                product_link = cols[0].find('a')
                product_name = product_link.text.strip() if product_link else cols[0].text.strip()
                product_url = product_link.get('href', '') if product_link else ''
                
                if not product_name or not product_url:
                    continue
                
                full_url = product_url if product_url.startswith('http') else f"https://www.shl.com{product_url}"
                
                # Fetch description from product page
                print(f"\n   [{idx}/{len(rows)}] {product_name}")
                print(f"      URL: {full_url}")
                description = self._scrape_product_description(driver, full_url, product_name)
                
                remote_testing = 'Yes' if (cols[1].text.strip() or cols[1].find('img') or cols[1].find('svg')) else 'No'
                adaptive_irt = 'Yes' if (cols[2].text.strip() or cols[2].find('img') or cols[2].find('svg')) else 'No'
                test_type = cols[3].text.strip()
                
                assessment = {
                    'category': 'Individual Test Solutions',
                    'name': product_name,
                    'url': full_url,
                    'description': description,
                    'remote_testing': remote_testing,
                    'adaptive_irt': adaptive_irt,
                    'test_type': test_type,
                    'page_number': page_num,
                    'source': 'SHL Catalog'
                }
                assessments.append(assessment)
        
        return assessments
    
    def _click_next_button_for_individual_table(self, driver, table_index: int) -> bool:
        """
        Find and click the Next button for the Individual Test Solutions table
        """
        try:
            time.sleep(1)
            
            # Find all pagination containers
            paginations = driver.find_elements(By.CLASS_NAME, "pagination")
            
            if len(paginations) > 1:
                # Use the SECOND pagination for Individual Tests
                target_pagination_index = 1 if table_index == 1 else table_index
                
                if target_pagination_index < len(paginations):
                    pagination = paginations[target_pagination_index]
                    
                    # Find Next button
                    next_buttons = pagination.find_elements(By.XPATH, ".//a[contains(text(), 'Next') or contains(text(), '›')]")
                    
                    if next_buttons:
                        next_button = next_buttons[0]
                        
                        # Check if disabled
                        classes = next_button.get_attribute('class') or ''
                        if 'disabled' in classes.lower():
                            return False
                        
                        # Scroll and click
                        driver.execute_script("arguments[0].scrollIntoView({block: 'center'});", pagination)
                        time.sleep(0.5)
                        driver.execute_script("arguments[0].click();", next_button)
                        time.sleep(2)
                        return True
                    else:
                        return False
                else:
                    return False
                    
            elif len(paginations) == 1:
                # Only one table visible
                pagination = paginations[0]
                next_buttons = pagination.find_elements(By.XPATH, ".//a[contains(text(), 'Next') or contains(text(), '›')]")
                
                if next_buttons:
                    next_button = next_buttons[0]
                    classes = next_button.get_attribute('class') or ''
                    if 'disabled' in classes.lower():
                        return False
                    
                    driver.execute_script("arguments[0].scrollIntoView({block: 'center'});", pagination)
                    time.sleep(0.5)
                    driver.execute_script("arguments[0].click();", next_button)
                    time.sleep(2)
                    return True
            
            return False
            
        except Exception as e:
            print(f"   ✗ Error clicking next: {e}")
            return False
    
    def save_to_json(self, assessments: List[Dict], filepath: str):
        """Save to JSON"""
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(assessments, f, indent=2, ensure_ascii=False)
        print(f"\n✓ Saved {len(assessments)} assessments to {filepath}")
    
    def save_to_csv(self, assessments: List[Dict], filepath: str):
        """Save to CSV"""
        import csv
        if not assessments:
            return
        with open(filepath, 'w', newline='', encoding='utf-8') as f:
            fieldnames = ['category', 'name', 'url', 'description', 'remote_testing', 'adaptive_irt', 'test_type', 'page_number', 'source']
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(assessments)
        print(f"✓ Saved {len(assessments)} assessments to {filepath}")


if __name__ == "__main__":
    print("\n" + "="*70)
    print("SHL INDIVIDUAL TEST SOLUTIONS SCRAPER")
    print("Fetching descriptions from each product page")
    print("="*70 + "\n")
    
    scraper = IndividualTestSolutionsScraper()
    assessments = scraper.scrape_individual_solutions()
    
    if assessments:
        scraper.save_to_json(assessments, 'individual_solutions.json')
        scraper.save_to_csv(assessments, 'individual_solutions.csv')
        
        # Print summary
        with_desc = sum(1 for a in assessments if a.get('description'))
        print(f"\n{'='*70}")
        print(f"SUCCESS!")
        print(f"{'='*70}")
        print(f"   Total Products: {len(assessments)}")
        print(f"   With Descriptions: {with_desc}")
        print(f"   Without Descriptions: {len(assessments) - with_desc}")
        print(f"{'='*70}\n")
    else:
        print("\n✗ FAILED: No data scraped\n")