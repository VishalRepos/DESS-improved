#!/usr/bin/env python3
"""
Convert old format output.txt to JSON triplets.

Usage:
    python convert_old_format.py
    
Input: Results/OutputLogs/output.txt (old format)
Output: Results/OutputTriplets/triplets.jsonl (JSON format)
"""

import json
import os
import re

def parse_old_format(input_file, output_file):
    """Parse old format and convert to JSON"""
    
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    
    triplet_count = 0
    current_sentence = None
    current_text = None
    current_triplets = []
    in_triplets_section = False
    
    with open(input_file, 'r', encoding='utf-8') as infile, \
         open(output_file, 'w', encoding='utf-8') as outfile:
        
        for line in infile:
            line = line.rstrip()
            
            # Detect sentence start
            if line.startswith("SENTENCE "):
                # Save previous sentence if exists
                if current_sentence is not None and current_text is not None:
                    output = {
                        "ID": f"sentence_{current_sentence}",
                        "Text": current_text,
                        "Quadruplet": current_triplets if current_triplets else [{"Aspect": "NULL", "Opinion": "NULL", "Sentiment": "NULL"}]
                    }
                    outfile.write(json.dumps(output, ensure_ascii=False) + '\n')
                    triplet_count += 1
                
                # Start new sentence
                match = re.search(r'SENTENCE (\d+):', line)
                if match:
                    current_sentence = match.group(1)
                    current_triplets = []
                    current_text = None
                    in_triplets_section = False
            
            # Extract text
            elif line.startswith("Text:"):
                current_text = line.replace("Text:", "").strip()
            
            # Detect triplets section
            elif "PREDICTED TRIPLETS" in line:
                in_triplets_section = True
            
            # Parse triplet lines
            elif in_triplets_section and line.strip().startswith(tuple(str(i) for i in range(1, 100))):
                # This is a triplet entry like "1. Aspect: ..."
                aspect = None
                opinion = None
                sentiment = None
                
                # Look for Aspect in this line
                if "Aspect:" in line:
                    aspect_match = re.search(r"Aspect: '([^']+)'", line)
                    if aspect_match:
                        aspect = aspect_match.group(1)
            
            # Parse Opinion line (indented)
            elif in_triplets_section and "Opinion:" in line:
                opinion_match = re.search(r"Opinion: '([^']+)'", line)
                if opinion_match:
                    opinion = opinion_match.group(1)
            
            # Parse Sentiment line (indented)
            elif in_triplets_section and "Sentiment:" in line:
                sentiment_match = re.search(r"Sentiment: (\w+)", line)
                if sentiment_match:
                    sentiment = sentiment_match.group(1)
                    
                    # Complete triplet
                    if aspect and opinion and sentiment:
                        current_triplets.append({
                            "Aspect": aspect,
                            "Opinion": opinion,
                            "Sentiment": sentiment
                        })
                        aspect = None
                        opinion = None
                        sentiment = None
            
            # End of sentence block
            elif line.startswith("="*80):
                in_triplets_section = False
        
        # Save last sentence
        if current_sentence is not None and current_text is not None:
            output = {
                "ID": f"sentence_{current_sentence}",
                "Text": current_text,
                "Quadruplet": current_triplets if current_triplets else [{"Aspect": "NULL", "Opinion": "NULL", "Sentiment": "NULL"}]
            }
            outfile.write(json.dumps(output, ensure_ascii=False) + '\n')
            triplet_count += 1
    
    return triplet_count


if __name__ == "__main__":
    input_file = "Results/OutputLogs/output.txt"
    output_file = "Results/OutputTriplets/triplets.jsonl"
    
    if not os.path.exists(input_file):
        print(f"❌ Error: Input file not found: {input_file}")
        exit(1)
    
    print(f"📖 Reading from: {input_file}")
    print(f"💾 Writing to: {output_file}")
    print(f"🔄 Converting old format to JSON...")
    print()
    
    count = parse_old_format(input_file, output_file)
    
    print(f"✅ Conversion complete!")
    print(f"📊 Total sentences converted: {count}")
    print(f"📁 Output saved to: {output_file}")
    
    # Show preview
    if count > 0:
        print(f"\n📋 Preview (first 3 lines):")
        with open(output_file, 'r', encoding='utf-8') as f:
            for i, line in enumerate(f):
                if i >= 3:
                    break
                data = json.loads(line)
                print(f"  {i+1}. ID: {data['ID']}, Text: {data['Text'][:50]}..., Triplets: {len(data['Quadruplet'])}")
