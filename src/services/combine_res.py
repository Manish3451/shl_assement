import json
import csv
from typing import List, Dict

def combine_json_files(file1: str, file2: str, output: str):
    """Combine two JSON files"""
    try:
        with open(file1, 'r', encoding='utf-8') as f:
            data1 = json.load(f)
        
        with open(file2, 'r', encoding='utf-8') as f:
            data2 = json.load(f)
        
        combined = data1 + data2
        
        with open(output, 'w', encoding='utf-8') as f:
            json.dump(combined, f, indent=2, ensure_ascii=False)
        
        print(f"✅ Combined {len(data1)} + {len(data2)} = {len(combined)} products")
        print(f"💾 Saved to {output}")
        return combined
    except Exception as e:
        print(f"❌ Error combining JSON: {e}")
        return []

def combine_csv_files(file1: str, file2: str, output: str):
    """Combine two CSV files"""
    try:
        with open(file1, 'r', encoding='utf-8') as f1, \
             open(file2, 'r', encoding='utf-8') as f2, \
             open(output, 'w', newline='', encoding='utf-8') as fout:
            
            reader1 = csv.DictReader(f1)
            reader2 = csv.DictReader(f2)
            
            # Get fieldnames from first file
            fieldnames = reader1.fieldnames
            
            writer = csv.DictWriter(fout, fieldnames=fieldnames)
            writer.writeheader()
            
            count = 0
            for row in reader1:
                writer.writerow(row)
                count += 1
            
            for row in reader2:
                writer.writerow(row)
                count += 1
            
            print(f"✅ Combined {count} products into CSV")
            print(f"💾 Saved to {output}")
    except Exception as e:
        print(f"❌ Error combining CSV: {e}")

if __name__ == "__main__":
    print("🔗 Combining scraping results...\n")
    
    # Combine JSON
    combine_json_files(
        'prepackaged_solutions.json',
        'individual_solutions.json',
        'shl_all_assessments.json'
    )
    
    print()
    
    # Combine CSV
    combine_csv_files(
        'prepackaged_solutions.csv',
        'individual_solutions.csv',
        'shl_all_assessments.csv'
    )