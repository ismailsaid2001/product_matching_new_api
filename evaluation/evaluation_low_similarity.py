#!/usr/bin/env python3
"""
Evaluation script for low similarity product descriptions.
Tests the system on 200 random samples without ground truth.
"""

import pandas as pd
import requests
import json
import time
from datetime import datetime
import random
import sys
import os

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

def load_sample_data(csv_path: str, sample_size: int = 200):
    """Load and sample data from CSV file."""
    print(f"Loading data from: {csv_path}")
    
    # Try different encodings
    for encoding in ['utf-8', 'latin-1', 'cp1252']:
        try:
            df = pd.read_csv(csv_path, encoding=encoding)
            print(f"Successfully loaded with {encoding} encoding")
            break
        except UnicodeDecodeError:
            continue
    else:
        raise ValueError("Could not read CSV with any common encoding")
    
    print(f"Total products in file: {len(df)}")
    print(f"Columns: {list(df.columns)}")
    
    # Sample random products
    if len(df) > sample_size:
        df_sample = df.sample(n=sample_size, random_state=42)
        print(f"Sampled {sample_size} random products")
    else:
        df_sample = df.copy()
        print(f"Using all {len(df)} products (less than requested sample size)")
    
    return df_sample

def classify_product(description: str, api_url: str = "http://localhost:8000/classify"):
    """Classify a single product using the API."""
    try:
        payload = {
            "designation": description,
            "product_id": f"eval_{random.randint(1000, 9999)}"
        }
        
        response = requests.post(
            api_url,
            json=payload,
            timeout=30
        )
        response.raise_for_status()
        return response.json()
    
    except requests.exceptions.RequestException as e:
        print(f"Error calling API: {e}")
        return None
    except Exception as e:
        print(f"Unexpected error: {e}")
        return None

def run_evaluation(csv_path: str, sample_size: int = 200):
    """Run evaluation on the dataset."""
    print("="*80)
    print("🚀 EVALUATION ON LOW SIMILARITY PRODUCTS")
    print("="*80)
    
    # Load data
    df_sample = load_sample_data(csv_path, sample_size)
    
    # Find description column (common names)
    description_col = None
    possible_cols = ['description', 'designation', 'libelle', 'produit', 'product', 'name']
    
    for col in possible_cols:
        if col.lower() in [c.lower() for c in df_sample.columns]:
            description_col = col
            break
    
    if not description_col:
        print("Available columns:", list(df_sample.columns))
        description_col = input("Enter the column name containing product descriptions: ")
    
    print(f"Using column '{description_col}' for descriptions")
    
    # Prepare results
    results = []
    total_cost = 0.0
    total_time = 0.0
    
    print(f"\n🔄 Processing {len(df_sample)} products...")
    
    # Process each product
    current_count = 0
    for idx, row in df_sample.iterrows():
        description = str(row[description_col]).strip()
        
        if not description or description.lower() in ['nan', 'null', '']:
            continue
            
        current_count += 1
        print(f"[{current_count}/{len(df_sample)}] Processing: {description[:50]}...")
        
        start_time = time.time()
        result = classify_product(description)
        processing_time = (time.time() - start_time) * 1000
        
        if result:
            # Handle None values for cost_usd
            cost_value = result.get('cost_usd', 0.0)
            if cost_value is not None:
                total_cost += cost_value
            total_time += processing_time
            
            # Store result
            results.append({
                'index': current_count,
                'description': description,
                'final_label': result.get('final_label', ''),
                'confidence': result.get('confidence', 0.0),
                'database_confidence': result.get('database_confidence'),
                'database_prediction': result.get('database_prediction'),
                't5_confidence': result.get('t5_confidence'),
                't5_prediction': result.get('t5_prediction'),
                'path_taken': str(result.get('path_taken', [])),
                'processing_time_ms': processing_time,
                'cost_usd': cost_value if cost_value is not None else 0.0
            })
        else:
            print(f"  ❌ Failed to process: {description[:30]}...")
            results.append({
                'index': current_count,
                'description': description,
                'final_label': 'ERROR',
                'confidence': 0.0,
                'database_confidence': None,
                'database_prediction': None,
                't5_confidence': None,
                't5_prediction': None,
                'path_taken': 'ERROR',
                'processing_time_ms': processing_time,
                'cost_usd': 0.0
            })
        
        # Brief pause to avoid overwhelming the system
        time.sleep(0.1)
    
    # Save results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = f"evaluation_low_similarity_{timestamp}.csv"
    
    df_results = pd.DataFrame(results)
    df_results.to_csv(output_file, index=False, encoding='utf-8')
    
    print(f"\n💾 Results saved to: {output_file}")
    
    # Generate summary
    generate_summary(df_results, total_cost, total_time)
    
    return df_results

def generate_summary(df_results: pd.DataFrame, total_cost: float, total_time: float):
    """Generate evaluation summary."""
    print("\n" + "="*80)
    print("📊 EVALUATION SUMMARY")
    print("="*80)
    
    total_products = len(df_results)
    successful_products = len(df_results[df_results['final_label'] != 'ERROR'])
    
    print(f"Total products processed: {total_products}")
    print(f"Successful classifications: {successful_products}")
    print(f"Failed classifications: {total_products - successful_products}")
    print(f"Success rate: {(successful_products/total_products)*100:.1f}%")
    
    if successful_products > 0:
        # Path analysis
        path_counts = df_results['path_taken'].value_counts()
        print(f"\n🛤️ Classification paths:")
        for path, count in path_counts.items():
            if path != 'ERROR':
                percentage = (count / successful_products) * 100
                print(f"  {path}: {count} ({percentage:.1f}%)")
        
        # Performance metrics
        avg_time = total_time / successful_products
        print(f"\n⚡ Performance:")
        print(f"  Total processing time: {total_time/1000:.1f}s")
        print(f"  Average time per product: {avg_time:.1f}ms")
        print(f"  Total cost: ${total_cost:.4f}")
        print(f"  Average cost per product: ${total_cost/successful_products:.6f}")
        
        # Confidence analysis
        valid_confidence = df_results[df_results['confidence'] > 0]['confidence']
        if len(valid_confidence) > 0:
            print(f"\n🎯 Confidence distribution:")
            print(f"  Mean confidence: {valid_confidence.mean():.3f}")
            print(f"  Median confidence: {valid_confidence.median():.3f}")
            print(f"  Min confidence: {valid_confidence.min():.3f}")
            print(f"  Max confidence: {valid_confidence.max():.3f}")
        
        # Show sample results
        print(f"\n📝 Sample results:")
        sample_results = df_results[df_results['final_label'] != 'ERROR'].head(10)
        for _, row in sample_results.iterrows():
            desc = row['description'][:40] + "..." if len(row['description']) > 40 else row['description']
            print(f"  '{desc}' → '{row['final_label']}' (conf: {row['confidence']:.2f})")

def main():
    """Main execution function."""
    # Default path to the uploaded file
    csv_path = "data/low_similarity.csv"
    
    # Check if file exists
    if not os.path.exists(csv_path):
        print(f"❌ File not found: {csv_path}")
        csv_path = input("Enter the full path to your CSV file: ")
        
        if not os.path.exists(csv_path):
            print(f"❌ File still not found: {csv_path}")
            return
    
    # Run evaluation
    sample_size = 200
    print(f"🎯 Starting evaluation with {sample_size} random samples...")
    
    try:
        df_results = run_evaluation(csv_path, sample_size)
        print("\n✅ Evaluation completed successfully!")
        
    except KeyboardInterrupt:
        print("\n❌ Evaluation interrupted by user")
    except Exception as e:
        print(f"\n❌ Evaluation failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()