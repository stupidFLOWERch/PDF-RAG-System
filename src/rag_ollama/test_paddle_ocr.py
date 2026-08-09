"""
PaddleOCR Document Parser - Test Script
Tests both PP-StructureV3 and PaddleOCR-VL-1.6
"""

import os

# Disable OneDNN (avoids compatibility issues on Windows)
os.environ["FLAGS_use_mkldnn"] = "0"
os.environ["FLAGS_use_onednn"] = "0"

import json
import numpy as np
from typing import List, Dict

# Import PaddleOCR loader
from paddle_loader import PaddleDocLoader, extract_with_paddle


def convert_numpy(obj):
    """Convert numpy types to Python native types for JSON serialization."""
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, np.float32):
        return float(obj)
    elif isinstance(obj, np.float64):
        return float(obj)
    elif isinstance(obj, np.int32):
        return int(obj)
    elif isinstance(obj, np.int64):
        return int(obj)
    elif isinstance(obj, dict):
        return {k: convert_numpy(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [convert_numpy(item) for item in obj]
    return obj


def print_results(elements: List[Dict], sections: List[Dict], document_title: str):
    """Print extraction results in a readable format."""
    print("\n" + "=" * 80)
    print("EXTRACTION RESULTS")
    print("=" * 80)
    
    # Statistics
    type_counts = {}
    for elem in elements:
        type_counts[elem['type']] = type_counts.get(elem['type'], 0) + 1
    
    print(f"\nStatistics:")
    print(f"  Total elements: {len(elements)}")
    print(f"  Total sections: {len(sections)}")
    print(f"  Element type distribution:")
    for elem_type, count in sorted(type_counts.items()):
        print(f"    - {elem_type}: {count}")
    
    if document_title:
        print(f"\nDocument Title: {document_title}")
    
    # Sections
    print(f"\nDocument Sections ({len(sections)} sections):")
    print("-" * 80)
    
    for i, section in enumerate(sections):
        heading = section['heading'][:60]
        content_preview = section['content'][:100].replace('\n', ' ') if section['content'] else '(empty)'
        word_count = len(section['content'].split()) if section['content'] else 0
        
        print(f"\nSection {i+1}:")
        print(f"  Heading: {heading}{'...' if len(section['heading']) > 60 else ''}")
        print(f"  Page: {section['page']}")
        print(f"  Word count: {word_count}")
        print(f"  Content preview: {content_preview}{'...' if len(content_preview) > 100 else ''}")
    
    # Element details
    print(f"\n\nElement Details (first 20):")
    print("-" * 80)
    print(f"  {'#':3} | {'Page':4} | {'Type':22} | {'Text':50}")
    print("-" * 80)
    
    for i, elem in enumerate(elements[:20]):
        text_preview = elem['text'][:50].replace('\n', ' ')
        print(f"  {i+1:3d} | {elem['page']:4} | {elem['type']:22} | {text_preview:50}")
    
    if len(elements) > 20:
        print(f"  ... and {len(elements) - 20} more elements")


def main():
    """Main test function."""
    pdf_path = r"C:\Users\User\Desktop\pdf-rag-system\uploads\plant-hunt-info.pdf"
    
    print("=" * 80)
    print("PaddleOCR Document Parser (VL-1.6 / PP-StructureV3)")
    print("=" * 80)
    print(f"PDF: {pdf_path}\n")
    
    # Check if file exists
    if not os.path.exists(pdf_path):
        print(f"ERROR: File not found - {pdf_path}")
        return
    
    # Use extract_with_paddle (default uses VL-1.6)
    print("\n" + "=" * 80)
    print("EXTRACTING WITH PaddleOCR")
    print("=" * 80)
    
    try:
        # use_vl=True for VL-1.6, use_vl=False for PP-StructureV3
        sections, document_title = extract_with_paddle(pdf_path, use_gpu=False, use_vl=True)
    except Exception as e:
        print(f"VL extraction failed: {e}")
        print("Falling back to PP-StructureV3...")
        sections, document_title = extract_with_paddle(pdf_path, use_gpu=False, use_vl=False)
    
    if not sections:
        print("No sections extracted")
        return
    
    # Build element list for display
    elements = []
    for section in sections:
        # Reconstruct element info from sections
        elements.append({
            'page': section.get('page', 1),
            'type': 'heading',
            'text': section.get('heading', ''),
            'confidence': 1.0
        })
        if section.get('content'):
            elements.append({
                'page': section.get('page', 1),
                'type': 'text',
                'text': section.get('content', '')[:100],
                'confidence': 1.0
            })
    
    # Print results
    print_results(elements, sections, document_title)
    
    # Save JSON
    output_file = os.path.splitext(pdf_path)[0] + "_layout.json"
    try:
        serializable_data = {
            'document_title': document_title,
            'total_sections': len(sections),
            'sections': convert_numpy(sections)
        }
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(serializable_data, f, ensure_ascii=False, indent=2)
        print(f"\nResults saved to: {output_file}")
    except Exception as e:
        print(f"Save JSON failed: {e}")
    
    print("\n" + "=" * 80)
    print("Test Complete")
    print("=" * 80)
    
    return sections, document_title


if __name__ == "__main__":
    main()