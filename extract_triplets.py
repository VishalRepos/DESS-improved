#!/usr/bin/env python3
"""
Extract JSON triplets from training output logs.

Usage:
    python extract_triplets.py
    
Input: OutputLogs/output.txt
Output: OutputTriplets/triplets.jsonl
"""

import json
import os
import re

def extract_triplets(input_file, output_file):
    """Extract JSON lines from output.txt and save to triplets.jsonl"""
    
    # Create output directory if it doesn't exist
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    
    triplet_count = 0
    
    with open(input_file, 'r', encoding='utf-8') as infile, \
         open(output_file, 'w', encoding='utf-8') as outfile:
        
        for line in infile:
            line = line.strip()
            
            # Check if line starts with { and looks like JSON
            if line.startswith('{') and '"ID"' in line and '"Text"' in line:
                try:
                    # Validate it's proper JSON
                    json_obj = json.loads(line)
                    
                    # Write to output file
                    outfile.write(line + '\n')
                    triplet_count += 1
                    
                except json.JSONDecodeError:
                    # Skip invalid JSON lines
                    continue
    
    return triplet_count


if __name__ == "__main__":
    # Define paths
    input_file = "OutputLogs/output.txt"
    output_file = "OutputTriplets/triplets.jsonl"
    
    # Check if input file exists
    if not os.path.exists(input_file):
        print(f"❌ Error: Input file not found: {input_file}")
        print(f"Please ensure the file exists in the OutputLogs folder.")
        exit(1)
    
    print(f"📖 Reading from: {input_file}")
    print(f"💾 Writing to: {output_file}")
    print()
    
    # Extract triplets
    count = extract_triplets(input_file, output_file)
    
    print(f"✅ Extraction complete!")
    print(f"📊 Total triplets extracted: {count}")
    print(f"📁 Output saved to: {output_file}")
    
    # Show first few lines as preview
    if count > 0:
        print(f"\n📋 Preview (first 3 lines):")
        with open(output_file, 'r', encoding='utf-8') as f:
            for i, line in enumerate(f):
                if i >= 3:
                    break
                data = json.loads(line)
                print(f"  {i+1}. ID: {data['ID']}, Triplets: {len(data['Quadruplet'])}")
