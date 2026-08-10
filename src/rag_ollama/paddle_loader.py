"""
PaddleOCR-based document structure extractor.
Supports both PP-StructureV3 and PaddleOCR-VL-1.6
"""

import os
import re
from typing import List, Dict, Optional, Tuple

# Disable OneDNN (avoids compatibility issues on Windows)
os.environ['FLAGS_use_mkldnn'] = '0'
os.environ['FLAGS_use_onednn'] = '0'


def clean_text(text: str) -> str:
    """
    Clean text extracted from scanned PDFs (OCR results).
    Removes emojis, stray characters, and normalizes whitespace.
    """
    if not text:
        return text
    
    # Remove emojis
    emoji_pattern = re.compile(
        "["
        "\U0001F600-\U0001F64F"
        "\U0001F300-\U0001F5FF"
        "\U0001F680-\U0001F6FF"
        "\U0001F700-\U0001F77F"
        "\U0001F780-\U0001F7FF"
        "\U0001F800-\U0001F8FF"
        "\U0001F900-\U0001F9FF"
        "\U0001FA00-\U0001FA6F"
        "\U0001FA70-\U0001FAFF"
        "\U00002702-\U000027B0"
        "\U000024C2-\U0001F251"
        "]+",
        flags=re.UNICODE
    )
    text = emoji_pattern.sub('', text)
    
    # Clean up common OCR artifacts
    text = re.sub(r'^[^\w\s]{1,3}', '', text)      # Remove leading garbage chars
    text = re.sub(r'^[\d]{1,2}', '', text)         # Remove leading numbers
    text = re.sub(r'[^\w\s\.\,\!\?\-\:\;\(\)"\']+$', '', text)  # Remove trailing garbage
    
    # Normalize whitespace
    text = re.sub(r'\s+', ' ', text).strip()
    return text


class PaddleDocLoader:
    """Load and extract document structure using PaddleOCR."""
    
    def __init__(
        self,
        use_gpu: bool = False,
        lang: str = 'en',
        use_vl: bool = True,
    ):
        """
        Initialize the PaddleOCR document loader.
        
        Args:
            use_gpu: Whether to use GPU for inference
            lang: Language code ('en', 'ch', 'korean', 'japan')
            use_vl: Use PaddleOCR-VL-1.6 (if False, falls back to PP-StructureV3)
        """
        self.lang = lang
        self.use_gpu = use_gpu
        self.use_vl = use_vl
        self.parser = None
        
        print(f"🔄 Initializing PaddleOCR Parser (GPU: {use_gpu}, Lang: {lang})...")
        
        if use_vl:
            self._init_vl_parser()
        else:
            self._init_structure_parser()
    
    def _init_vl_parser(self):
        """Initialize PaddleOCR-VL-1.6 parser. Falls back to PP-StructureV3 on failure."""
        try:
            from paddleocr import PaddleOCRVL
            
            print("   Using PaddleOCR-VL-1.6...")
            
            init_methods = [
                lambda: PaddleOCRVL(pipeline_version="v1.6", device='gpu' if self.use_gpu else 'cpu'),
                lambda: PaddleOCRVL(pipeline_version="v1.6"),
                lambda: PaddleOCRVL(pipeline_version="v1.5"),
                lambda: PaddleOCRVL(),
            ]
            
            for init_func in init_methods:
                try:
                    self.parser = init_func()
                    print("   ✅ PaddleOCR-VL initialized successfully")
                    return
                except Exception as e:
                    print(f"   ⚠️ VL init attempt failed: {e}")
                    continue
            
            # 所有 VL 尝试都失败 → 回退到 PP-StructureV3
            print("   ⚠️ All VL initialization attempts failed")
            print("   🔄 Falling back to PP-StructureV3...")
            self.use_vl = False
            self._init_structure_parser()
            
        except ImportError as e:
            print(f"   ⚠️ PaddleOCRVL not available: {e}")
            print("   🔄 Falling back to PP-StructureV3...")
            self.use_vl = False
            self._init_structure_parser()
        except Exception as e:
            print(f"   ❌ VL initialization failed: {e}")
            print("   🔄 Falling back to PP-StructureV3...")
            self.use_vl = False
            self._init_structure_parser()
    
    def _init_structure_parser(self):
        """Initialize PP-StructureV3 parser."""
        try:
            from paddleocr import PPStructureV3
            
            print("   Using PP-StructureV3...")
            
            init_methods = [
                lambda: PPStructureV3(device='gpu' if self.use_gpu else 'cpu'),
                lambda: PPStructureV3(device='gpu' if self.use_gpu else 'cpu', show_log=False),
                lambda: PPStructureV3(show_log=False),
                lambda: PPStructureV3(),
            ]
            
            for init_func in init_methods:
                try:
                    self.parser = init_func()
                    print("   ✅ PP-StructureV3 initialized successfully")
                    return
                except Exception as e:
                    continue
            
            raise RuntimeError("All Structure initialization attempts failed")
            
        except Exception as e:
            print(f"   ❌ Structure parser initialization failed: {e}")
            raise
    
    def extract_structure(self, pdf_path: str) -> List[Dict]:
        """
        Extract document structure from PDF.
        Assumes the PDF is already detected as scanned (OCR required).
        
        Args:
            pdf_path: Path to the PDF file
            
        Returns:
            List of extracted elements with page, type, text, bbox, and confidence
        """
        if not os.path.exists(pdf_path):
            raise FileNotFoundError(f"PDF file not found: {pdf_path}")
        
        print(f"📄 PDF type: Scanned (OCR required)")
        
        if self.use_vl:
            return self._extract_with_vl(pdf_path)
        else:
            return self._extract_with_structure(pdf_path)

    def _extract_with_vl(self, pdf_path: str) -> List[Dict]:
        """Extract using PaddleOCR-VL."""
        print("📄 Using PaddleOCR-VL for document parsing...")
        
        try:
            result = list(self.parser.predict(pdf_path))
        except Exception as e:
            print(f"   ❌ VL prediction failed: {e}")
            return []
        
        all_elements = []
        
        for idx, res in enumerate(result):
            print(f"   Processing page {idx + 1}")
            
            try:
                if hasattr(res, 'save_to_markdown'):
                    import tempfile
                    with tempfile.NamedTemporaryFile(mode='w+', suffix='.md', delete=False) as f:
                        temp_path = f.name
                    
                    res.save_to_markdown(temp_path)
                    
                    with open(temp_path, 'r', encoding='utf-8') as f:
                        md_text = f.read()
                    
                    os.unlink(temp_path)
                    
                    if md_text and len(md_text) > 10:
                        print(f"   ✅ Page {idx + 1} markdown: {len(md_text)} chars")
                        page_elements = self._parse_markdown(md_text, page_num=idx + 1)
                        all_elements.extend(page_elements)
                        print(f"   ✅ Extracted {len(page_elements)} elements from page {idx + 1}")
            except Exception as e:
                print(f"   save_to_markdown error for page {idx + 1}: {e}")
        
        print(f"   Total elements across all pages: {len(all_elements)}")
        return all_elements
    
    def _extract_with_structure(self, pdf_path: str) -> List[Dict]:
        """Extract using PP-StructureV3."""
        print("📄 Using PP-StructureV3 for layout detection...")
        
        try:
            result = list(self.parser.predict(pdf_path))
        except Exception as e:
            print(f"   ❌ Structure prediction failed: {e}")
            return []
        
        elements = []
        
        for page_idx, page in enumerate(result):
            parsing_list = getattr(page, 'parsing_res_list', [])
            if not parsing_list and isinstance(page, dict):
                parsing_list = page.get('parsing_res_list', [])
            
            if parsing_list:
                for block in parsing_list:
                    text = getattr(block, 'content', '')
                    if not text and isinstance(block, dict):
                        text = block.get('content', '')
                    
                    if text and text.strip():
                        label = getattr(block, 'label', 'text')
                        if not label and isinstance(block, dict):
                            label = block.get('label', 'text')
                        
                        bbox = getattr(block, 'bbox', [])
                        if not bbox and isinstance(block, dict):
                            bbox = block.get('bbox', [])
                        
                        elements.append({
                            'page': page_idx + 1,
                            'type': label,
                            'text': clean_text(text.strip()),
                            'bbox': bbox,
                            'confidence': 1.0
                        })
                continue
            
            # Fallback: extract from OCR results
            ocr_res = getattr(page, 'overall_ocr_res', {})
            if not ocr_res and isinstance(page, dict):
                ocr_res = page.get('overall_ocr_res', {})
            
            rec_texts = getattr(ocr_res, 'rec_texts', [])
            if not rec_texts and isinstance(ocr_res, dict):
                rec_texts = ocr_res.get('rec_texts', [])
            
            for text in rec_texts:
                if text and text.strip():
                    elements.append({
                        'page': page_idx + 1,
                        'type': 'text',
                        'text': clean_text(text.strip()),
                        'bbox': [],
                        'confidence': 1.0
                    })
        
        print(f"✅ Extracted {len(elements)} elements")
        return elements
    
    def _parse_markdown(self, md_text: str, page_num: int = 1) -> List[Dict]:
        """
        Parse Markdown text into structured elements.
        
        Args:
            md_text: Markdown formatted text
            page_num: Page number for all extracted elements
            
        Returns:
            List of parsed elements (heading/text)
        """
        if not md_text:
            return []
        
        md_text = md_text.replace('\\n', '\n').replace('\\"', '"').replace('\\t', ' ')
        
        elements = []
        lines = md_text.split('\n')
        
        current_heading = None
        current_content = []
        
        for line in lines:
            line = line.strip()
            if not line:
                continue
            
            # Skip HTML tables
            if line.startswith('<table') or line.startswith('</table>') or 'td' in line:
                continue
            
            # Detect headings (# ## ###)
            if line.startswith('#'):
                # Save previous heading's content
                if current_heading is not None and current_content:
                    elements.append({
                        'page': page_num,
                        'type': 'text',
                        'text': clean_text(' '.join(current_content)),
                        'bbox': [],
                        'confidence': 1.0
                    })
                    current_content = []
                
                heading_text = line.lstrip('#').strip()
                if heading_text:
                    current_heading = heading_text
                    elements.append({
                        'page': page_num,
                        'type': 'heading',
                        'text': clean_text(heading_text),
                        'bbox': [],
                        'confidence': 1.0
                    })
                continue
            
            # Regular text - belongs to current heading
            if current_heading is not None:
                current_content.append(line)
            else:
                # No heading, treat as standalone text
                elements.append({
                    'page': page_num,
                    'type': 'text',
                    'text': clean_text(line),
                    'bbox': [],
                    'confidence': 1.0
                })
        
        # Handle remaining content
        if current_heading is not None and current_content:
            elements.append({
                'page': page_num,
                'type': 'text',
                'text': clean_text(' '.join(current_content)),
                'bbox': [],
                'confidence': 1.0
            })
        
        return elements
    
    def create_sections(self, pdf_path: str) -> Tuple[List[Dict], str]:
        """
        Group elements into sections by heading.
        
        Args:
            pdf_path: Path to the PDF file
            
        Returns:
            Tuple of (sections, document_title)
        """
        elements = self.extract_structure(pdf_path)
        
        if not elements:
            print("⚠️  No elements extracted")
            return [], ""
        
        sections = []
        current_section = None
        document_title = None
        
        for elem in elements:
            if elem['type'] == 'heading':
                # Save previous section
                if current_section:
                    sections.append(current_section)
                
                # First heading becomes document title
                if document_title is None:
                    document_title = elem['text']
                
                # Start new section
                current_section = {
                    'heading': elem['text'],
                    'page': elem.get('page', 1),
                    'content': ''
                }
            else:
                # Add text to current section
                if current_section:
                    current_section['content'] += elem['text'] + ' '
                else:
                    # No heading yet, create default section
                    current_section = {
                        'heading': 'Document Start',
                        'page': elem.get('page', 1),
                        'content': elem['text'] + ' '
                    }
        
        # Save the last section
        if current_section:
            sections.append(current_section)
        
        print(f"✅ Created {len(sections)} sections")
        return sections, document_title or ''


def extract_with_paddle(pdf_path: str, use_gpu: bool = False, use_vl: bool = True) -> Tuple[List[Dict], str]:
    """
    Convenience function to extract sections from a scanned PDF using PaddleOCR.
    
    Args:
        pdf_path: Path to the PDF file
        use_gpu: Whether to use GPU for inference
        use_vl: Use PaddleOCR-VL-1.6 (if False, falls back to PP-StructureV3)
        
    Returns:
        Tuple of (sections, document_title)
    """
    loader = PaddleDocLoader(use_gpu=use_gpu, use_vl=use_vl)
    return loader.create_sections(pdf_path)


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) < 2:
        print("Usage: python paddle_loader.py <pdf_path>")
        sys.exit(1)
    
    pdf_path = sys.argv[1]
    
    loader = PaddleDocLoader(use_vl=True)
    sections, title = loader.create_sections(pdf_path)
    
    print(f"\n📌 Document Title: {title}")
    print(f"📊 Found {len(sections)} sections")
    
    for i, section in enumerate(sections[:3]):
        print(f"\nSection {i+1}: {section['heading'][:50]}...")
        content_preview = section['content'][:100]
        if content_preview:
            print(f"  Content: {content_preview}...")