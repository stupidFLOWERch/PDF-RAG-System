"""
PaddleOCR-based document structure extractor.

Supports both PP-StructureV3 and PaddleOCR-VL-1.6.
HTML/table cleaning is handled by the chunker.
"""

import os
import re
from typing import List, Dict, Tuple


# Disable OneDNN (avoids compatibility issues on Windows)
os.environ["FLAGS_use_mkldnn"] = "0"
os.environ["FLAGS_use_onednn"] = "0"


# ============================================================
# Regex
# ============================================================

_EMOJI_PATTERN = re.compile(
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
    flags=re.UNICODE,
)


# Short form-field labels that VL often emits as markdown headings.
_FIELD_LABEL_RE = re.compile(
    r"^(TO|FROM|BILL TO|SHIP TO|SOLD TO|"
    r"INVOICE\s*(NUMBER|NO\.?|#)|"
    r"(ISSUE|DUE|INVOICE)\s*DATE|"
    r"CURRENCY|DATE|P\.?O\.?\s*(NUMBER|NO\.?|#)?|"
    r"DELIVERY ADDRESS|NOTES?|"
    r"PAGE\s+\d+\s+OF\s+\d+)$",
    re.IGNORECASE,
)


# Supported document titles
_DOC_TITLE_RE = re.compile(
    r"^(TAX\s+)?"
    r"(INVOICE|PAYSLIP|PAY\s*SLIP|MEMO|RECEIPT|"
    r"STATEMENT|REPORT|SALARY|BILL)$",
    re.IGNORECASE,
)


# ============================================================
# Basic text cleaning
# ============================================================

def clean_text(text: str) -> str:
    """
    Basic OCR text cleanup.

    NOTE:
    HTML/table processing is intentionally NOT done here.
    The chunker handles HTML/table cleaning later.
    """

    if not text:
        return text

    # Remove emojis
    text = _EMOJI_PATTERN.sub("", text)

    # Normalize whitespace
    text = re.sub(r"\s+", " ", text).strip()

    return text


# ============================================================
# Generic object attribute helper
# ============================================================

def _get_attr(obj, key, default=None):
    if obj is None:
        return default

    if isinstance(obj, dict):
        return obj.get(key, default)

    return getattr(obj, key, default)


# ============================================================
# Heading detection
# ============================================================

def _is_field_label(text: str) -> bool:
    """
    Detect field labels such as:

        TO
        FROM
        INVOICE NUMBER
        DATE
        CURRENCY
        BILL TO
    """

    label = re.sub(r"[#*:]+", "", text or "").strip()

    return (
        bool(label)
        and bool(_FIELD_LABEL_RE.match(label))
    )


def _is_document_heading(text: str) -> bool:
    """
    Detect actual document titles such as:

        INVOICE
        MEMO
        RECEIPT
        STATEMENT
    """

    heading = (text or "").lstrip("#").strip()

    if not heading:
        return False

    # Do not treat fields as document headings
    if _is_field_label(heading):
        return False

    # Document title should normally not contain numbers
    if re.search(r"\d", heading):
        return False

    # Document title should normally not contain colon
    if ":" in heading:
        return False

    return bool(_DOC_TITLE_RE.match(heading))


# ============================================================
# PaddleDocLoader
# ============================================================

class PaddleDocLoader:
    """
    Load and extract document structure using PaddleOCR.

    PaddleOCR-VL is the preferred parser.

    HTML/table conversion is NOT performed here.
    The downstream chunker handles that.
    """

    def __init__(
        self,
        use_gpu: bool = False,
        lang: str = "en",
        use_vl: bool = True,
    ):
        """
        Args:
            use_gpu:
                Whether to use GPU for inference.

            lang:
                Language code.

            use_vl:
                Use PaddleOCR-VL-1.6.
                If False, use PP-StructureV3.
        """

        self.lang = lang
        self.use_gpu = use_gpu
        self.use_vl = use_vl
        self.parser = None

        print(
            f"🔄 Initializing PaddleOCR Parser "
            f"(GPU: {use_gpu}, Lang: {lang})..."
        )

        if use_vl:
            self._init_vl_parser()
        else:
            self._init_structure_parser()

    # ========================================================
    # Initialize PaddleOCR-VL
    # ========================================================

    def _init_vl_parser(self):
        """
        Initialize PaddleOCR-VL-1.6.

        Falls back to PP-StructureV3 if unavailable.
        """

        try:
            from paddleocr import PaddleOCRVL

            print("   Using PaddleOCR-VL-1.6...")

            init_methods = [
                lambda: PaddleOCRVL(
                    pipeline_version="v1.6",
                    device="gpu" if self.use_gpu else "cpu",
                ),
                lambda: PaddleOCRVL(
                    pipeline_version="v1.6"
                ),
                lambda: PaddleOCRVL(
                    pipeline_version="v1.5"
                ),
                lambda: PaddleOCRVL(),
            ]

            for init_func in init_methods:

                try:
                    self.parser = init_func()

                    print(
                        "   ✅ PaddleOCR-VL initialized successfully"
                    )

                    return

                except Exception as e:

                    print(
                        f"   ⚠️ VL init attempt failed: {e}"
                    )

            # All VL attempts failed
            print(
                "   ⚠️ All VL initialization attempts failed"
            )

            print(
                "   🔄 Falling back to PP-StructureV3..."
            )

            self.use_vl = False

            self._init_structure_parser()

        except ImportError as e:

            print(
                f"   ⚠️ PaddleOCRVL not available: {e}"
            )

            print(
                "   🔄 Falling back to PP-StructureV3..."
            )

            self.use_vl = False

            self._init_structure_parser()

        except Exception as e:

            print(
                f"   ❌ VL initialization failed: {e}"
            )

            print(
                "   🔄 Falling back to PP-StructureV3..."
            )

            self.use_vl = False

            self._init_structure_parser()

    # ========================================================
    # Initialize PP-StructureV3
    # ========================================================

    def _init_structure_parser(self):
        """Initialize PP-StructureV3 parser."""

        try:
            from paddleocr import PPStructureV3

            print("   Using PP-StructureV3...")

            init_methods = [
                lambda: PPStructureV3(
                    device="gpu" if self.use_gpu else "cpu"
                ),
                lambda: PPStructureV3(
                    device="gpu" if self.use_gpu else "cpu",
                    show_log=False,
                ),
                lambda: PPStructureV3(
                    show_log=False
                ),
                lambda: PPStructureV3(),
            ]

            for init_func in init_methods:

                try:
                    self.parser = init_func()

                    print(
                        "   ✅ PP-StructureV3 initialized successfully"
                    )

                    return

                except Exception:
                    continue

            raise RuntimeError(
                "All Structure initialization attempts failed"
            )

        except Exception as e:

            print(
                f"   ❌ Structure parser initialization failed: {e}"
            )

            raise

    # ========================================================
    # Main extraction entry
    # ========================================================

    def extract_structure(
        self,
        pdf_path: str
    ) -> List[Dict]:
        """
        Extract document structure from PDF.

        The PDF is assumed to be scanned,
        therefore OCR is required.
        """

        if not os.path.exists(pdf_path):
            raise FileNotFoundError(
                f"PDF file not found: {pdf_path}"
            )

        print(
            "📄 PDF type: Scanned (OCR required)"
        )

        if self.use_vl:
            return self._extract_with_vl(pdf_path)

        return self._extract_with_structure(pdf_path)

    # ========================================================
    # PaddleOCR-VL extraction
    # ========================================================

    def _extract_with_vl(
        self,
        pdf_path: str
    ) -> List[Dict]:
        """
        Extract using PaddleOCR-VL.

        Pipeline:

            PDF
             ↓
        PaddleOCR-VL
             ↓
        Markdown
             ↓
        Parse Markdown
             ↓
        Structured elements

        No separate OCR text recovery is performed.
        """

        print(
            "📄 Using PaddleOCR-VL for document parsing..."
        )

        # ----------------------------------------------------
        # Run PaddleOCR-VL
        # ----------------------------------------------------

        try:
            result = list(
                self.parser.predict(pdf_path)
            )

        except Exception as e:

            print(
                f"   ❌ VL prediction failed: {e}"
            )

            return []

        # ----------------------------------------------------
        # Process pages
        # ----------------------------------------------------

        all_elements = []

        for idx, res in enumerate(result):

            page_num = idx + 1

            print(
                f"   Processing page {page_num}"
            )

            try:

                # ------------------------------------------------
                # 1. Get markdown
                # ------------------------------------------------

                md_text = self._markdown_from_vl_result(res)

                if not md_text:

                    print(
                        f"   ⚠️ No markdown found "
                        f"for page {page_num}"
                    )

                    continue

                # ------------------------------------------------
                # 2. Parse markdown
                # ------------------------------------------------

                page_elements = self._parse_markdown(
                    md_text,
                    page_num=page_num,
                )

                # ------------------------------------------------
                # 3. Store elements
                # ------------------------------------------------

                if page_elements:

                    print(
                        f"   ✅ Page {page_num} markdown: "
                        f"{len(md_text)} chars"
                    )

                    print(
                        f"   📝 RAW MARKDOWN "
                        f"(page {page_num}):"
                    )

                    print("-" * 80)

                    print(
                        md_text[:1000]
                    )

                    print("-" * 80)

                    all_elements.extend(
                        page_elements
                    )

                    print(
                        f"   ✅ Extracted "
                        f"{len(page_elements)} elements "
                        f"from page {page_num}"
                    )

                else:

                    print(
                        f"   ⚠️ No elements extracted "
                        f"from page {page_num}"
                    )

            except Exception as e:

                print(
                    f"   ❌ VL extract error "
                    f"for page {page_num}: {e}"
                )

        print(
            f"   Total elements across all pages: "
            f"{len(all_elements)}"
        )

        return all_elements

    # ========================================================
    # PP-StructureV3 extraction
    # ========================================================

    def _extract_with_structure(
        self,
        pdf_path: str
    ) -> List[Dict]:
        """
        Extract using PP-StructureV3.

        No additional OCR recovery is performed.
        """

        print(
            "📄 Using PP-StructureV3 for layout detection..."
        )

        try:

            result = list(
                self.parser.predict(pdf_path)
            )

        except Exception as e:

            print(
                f"   ❌ Structure prediction failed: {e}"
            )

            return []

        elements = []

        for page_idx, page in enumerate(result):

            page_num = page_idx + 1

            page_elements = []

            parsing_list = (
                _get_attr(
                    page,
                    "parsing_res_list"
                )
                or []
            )

            for block in parsing_list:

                text = _get_attr(
                    block,
                    "content",
                    ""
                )

                if not text:
                    continue

                if not str(text).strip():
                    continue

                label = (
                    _get_attr(
                        block,
                        "label",
                        "text"
                    )
                    or "text"
                )

                bbox = (
                    _get_attr(
                        block,
                        "bbox",
                        []
                    )
                    or []
                )

                element_type = (
                    "heading"
                    if label in (
                        "doc_title",
                        "paragraph_title",
                        "title",
                    )
                    else label
                )

                page_elements.append(
                    {
                        "page": page_num,
                        "type": element_type,
                        "text": clean_text(
                            str(text).strip()
                        ),
                        "bbox": bbox,
                        "confidence": 1.0,
                    }
                )

            elements.extend(page_elements)

        print(
            f"✅ Extracted {len(elements)} elements"
        )

        return elements

    # ========================================================
    # Get Markdown from PaddleOCR-VL result
    # ========================================================

    def _markdown_from_vl_result(
        self,
        res
    ) -> str:
        """
        Read markdown from PaddleOCR-VL result.

        Supports:

        1. res.markdown as string
        2. res.markdown as dict
        3. save_to_markdown()
        """

        markdown = _get_attr(
            res,
            "markdown"
        )

        # ----------------------------------------------------
        # Case 1: markdown is string
        # ----------------------------------------------------

        if (
            isinstance(markdown, str)
            and markdown.strip()
        ):

            return markdown

        # ----------------------------------------------------
        # Case 2: markdown is dict
        # ----------------------------------------------------

        if isinstance(markdown, dict):

            for key in (
                "markdown",
                "text",
                "markdown_texts",
            ):

                value = markdown.get(key)

                if (
                    isinstance(value, str)
                    and value.strip()
                ):

                    return value

                if isinstance(value, list):

                    joined = "\n".join(
                        str(item)
                        for item in value
                        if item
                    )

                    if joined.strip():

                        return joined

        # ----------------------------------------------------
        # Case 3: save_to_markdown()
        # ----------------------------------------------------

        if not hasattr(
            res,
            "save_to_markdown"
        ):

            return ""

        import tempfile

        with tempfile.TemporaryDirectory() as tmpdir:

            res.save_to_markdown(tmpdir)

            parts = []

            for root, _, files in os.walk(
                tmpdir
            ):

                for name in sorted(files):

                    if not name.lower().endswith(
                        ".md"
                    ):

                        continue

                    path = os.path.join(
                        root,
                        name
                    )

                    with open(
                        path,
                        "r",
                        encoding="utf-8"
                    ) as handle:

                        parts.append(
                            handle.read()
                        )

            return "\n\n".join(parts)

    # ========================================================
    # Parse Markdown
    # ========================================================

    def _parse_markdown(
        self,
        md_text: str,
        page_num: int = 1
    ) -> List[Dict]:
        """
        Parse PaddleOCR-VL Markdown into
        structured elements.

        IMPORTANT:

        This function does NOT remove HTML.

        HTML/table processing is left to
        the chunker.
        """

        if not md_text:
            return []

        elements = []

        seen_heading = False

        # ----------------------------------------------------
        # Process lines
        # ----------------------------------------------------

        for line in md_text.split("\n"):

            line = line.strip()

            if not line:
                continue

            # Skip markdown separators
            if re.match(
                r"^[-*_`]{3,}$",
                line
            ):
                continue

            # ------------------------------------------------
            # Markdown heading
            # ------------------------------------------------

            if line.startswith("#"):

                heading_text = (
                    line.lstrip("#").strip()
                )

                if (
                    heading_text
                    and _is_document_heading(
                        heading_text
                    )
                    and not seen_heading
                ):

                    seen_heading = True

                    elements.append(
                        {
                            "page": page_num,
                            "type": "heading",
                            "text": clean_text(
                                heading_text
                            ),
                            "bbox": [],
                            "confidence": 1.0,
                        }
                    )

                    continue

                # Not a document title
                line = heading_text

            # ------------------------------------------------
            # Plain-text document heading
            # ------------------------------------------------

            if (
                _is_document_heading(line)
                and not seen_heading
            ):

                seen_heading = True

                elements.append(
                    {
                        "page": page_num,
                        "type": "heading",
                        "text": clean_text(line),
                        "bbox": [],
                        "confidence": 1.0,
                    }
                )

                continue

            # ------------------------------------------------
            # Normal text / HTML / table content
            # ------------------------------------------------

            cleaned = clean_text(line)

            if cleaned:

                elements.append(
                    {
                        "page": page_num,
                        "type": "text",
                        "text": cleaned,
                        "bbox": [],
                        "confidence": 1.0,
                    }
                )

        return elements

    # ========================================================
    # Create sections
    # ========================================================

    def create_sections(
        self,
        pdf_path: str
    ) -> Tuple[List[Dict], str]:
        """
        Group extracted elements into sections
        based on document headings.
        """

        elements = self.extract_structure(
            pdf_path
        )

        if not elements:

            print(
                "⚠️ No elements extracted"
            )

            return [], ""

        sections = []

        current_section = None

        document_title = None

        # ----------------------------------------------------
        # Group elements
        # ----------------------------------------------------

        for elem in elements:

            if elem["type"] == "heading":

                # Save previous section
                if current_section:

                    sections.append(
                        current_section
                    )

                # First heading = document title
                if document_title is None:

                    document_title = (
                        elem["text"]
                    )

                current_section = {
                    "heading": elem["text"],
                    "page": elem.get(
                        "page",
                        1
                    ),
                    "content": "",
                }

            else:

                if current_section:

                    current_section[
                        "content"
                    ] += (
                        elem["text"] + " "
                    )

                else:

                    current_section = {
                        "heading": "Document Start",
                        "page": elem.get(
                            "page",
                            1
                        ),
                        "content": (
                            elem["text"] + " "
                        ),
                    }

        # ----------------------------------------------------
        # Save final section
        # ----------------------------------------------------

        if current_section:

            sections.append(
                current_section
            )

        print(
            f"✅ Created {len(sections)} sections"
        )

        return (
            sections,
            document_title or ""
        )


# ============================================================
# Convenience function
# ============================================================

def extract_with_paddle(
    pdf_path: str,
    use_gpu: bool = False,
    use_vl: bool = True
) -> Tuple[List[Dict], str]:
    """
    Convenience function for extracting
    sections from a scanned PDF.
    """

    loader = PaddleDocLoader(
        use_gpu=use_gpu,
        use_vl=use_vl
    )

    return loader.create_sections(
        pdf_path
    )


# ============================================================
# Main
# ============================================================

if __name__ == "__main__":

    import sys

    if len(sys.argv) < 2:

        print(
            "Usage: python paddle_loader.py <pdf_path>"
        )

        sys.exit(1)

    pdf_path = sys.argv[1]

    loader = PaddleDocLoader(
        use_vl=True
    )

    sections, title = (
        loader.create_sections(
            pdf_path
        )
    )

    print(
        f"\n📌 Document Title: {title}"
    )

    print(
        f"📊 Found {len(sections)} sections"
    )

    for i, section in enumerate(
        sections[:3]
    ):

        print(
            f"\nSection {i + 1}: "
            f"{section['heading'][:50]}..."
        )

        content_preview = (
            section["content"][:100]
        )

        if content_preview:

            print(
                f"  Content: "
                f"{content_preview}..."
            )