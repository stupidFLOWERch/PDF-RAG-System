import pymupdf
import re


# ============================================================
# Text Cleaning
# ============================================================

def fix_ligatures(text):
    """
    Replace common PDF ligatures and special typography
    characters with standard ASCII equivalents.
    """

    replacements = {
        "ﬁ": "fi",
        "ﬂ": "fl",
        "ﬃ": "ffi",
        "ﬄ": "ffl",
        "ﬀ": "ff",
        "ﬅ": "ft",
        "ﬆ": "st",
        "Œ": "OE",
        "œ": "oe",
        "Æ": "AE",
        "æ": "ae",
        "–": "-",
        "—": "-",
    }

    for old, new in replacements.items():
        text = text.replace(old, new)

    return text


def clean_text(text):
    """
    Clean extracted PDF text.
    """

    # Fix ligatures
    text = fix_ligatures(text)

    # Remove repeated separators
    text = re.sub(r"[`\-_,]{5,}", "", text)

    # Normalize copyright symbols
    text = re.sub(r"\([cC]\)", "(c)", text)

    # Remove empty citation brackets
    text = re.sub(r"\[\s*\]", "", text)

    # Normalize whitespace
    text = re.sub(r"\s+", " ", text)

    # Remove duplicated page markers
    text = re.sub(
        r"===== Page \d+ =====",
        "",
        text
    )

    # Remove unsupported Unicode characters
    text = re.sub(
        r"[^\x00-\x7F\u2000-\u206F\u20A0-\u20CF]",
        "",
        text
    )

    return text.strip()


# ============================================================
# Bounding Box Utilities
# ============================================================

def bbox_area(bbox):
    """
    Calculate the area of a bounding box.

    bbox format:
        (x0, y0, x1, y1)
    """

    width = max(0, bbox[2] - bbox[0])
    height = max(0, bbox[3] - bbox[1])

    return width * height


def bbox_overlap_ratio(bbox1, bbox2):
    """
    Calculate how much of bbox1 overlaps bbox2.

    Returns:
        0.0 ~ 1.0
    """

    x0 = max(bbox1[0], bbox2[0])
    y0 = max(bbox1[1], bbox2[1])
    x1 = min(bbox1[2], bbox2[2])
    y1 = min(bbox1[3], bbox2[3])

    if x1 <= x0 or y1 <= y0:
        return 0.0

    intersection_area = (x1 - x0) * (y1 - y0)

    area1 = bbox_area(bbox1)

    if area1 <= 0:
        return 0.0

    return intersection_area / area1


def bbox_overlaps(bbox1, bbox2, threshold=0.5):
    """
    Return True when at least `threshold` of bbox1
    overlaps bbox2.

    This is intentionally not a strict containment check.

    PDF text line bounding boxes can extend slightly outside
    the detected table bounding box because of font metrics
    and rendering coordinates.
    """

    return bbox_overlap_ratio(bbox1, bbox2) >= threshold


# ============================================================
# Table Utilities
# ============================================================

def extract_table_data(table):
    """
    Extract structured table data from a PyMuPDF Table object.

    Returns:
        List[List[str]]
    """

    try:
        raw_data = table.extract()
    except Exception as exc:
        print(f"⚠️ Failed to extract table data: {exc}")
        return []

    cleaned_data = []

    for row in raw_data:

        cleaned_row = []

        for cell in row:

            if cell is None:
                cell = ""

            cell = clean_text(str(cell))

            cleaned_row.append(cell)

        # Keep the row even when some cells are empty.
        # Completely empty rows are ignored.
        if any(cell.strip() for cell in cleaned_row):
            cleaned_data.append(cleaned_row)

    return cleaned_data


def table_data_to_text(table_data):
    """
    Convert structured table data into a simple text representation.

    Example:

        [
            ["Hello", "World"],
            ["Hello", "World"]
        ]

    becomes:

        Hello | World
        Hello | World
    """

    lines = []

    for row in table_data:

        values = []

        for cell in row:
            values.append(clean_text(cell))

        if any(values):
            lines.append(" | ".join(values))

    return "\n".join(lines)


def table_data_to_markdown(table_data):
    """
    Convert structured table data to Markdown.

    The first row is treated as the header.

    Example:

        [
            ["Name", "Age"],
            ["Alice", "20"],
            ["Bob", "25"]
        ]

    becomes:

        | Name | Age |
        | --- | --- |
        | Alice | 20 |
        | Bob | 25 |
    """

    if not table_data:
        return ""

    # Remove completely empty rows
    rows = [
        row
        for row in table_data
        if any(str(cell).strip() for cell in row)
    ]

    if not rows:
        return ""

    # Determine maximum number of columns
    num_columns = max(len(row) for row in rows)

    normalized_rows = []

    for row in rows:

        normalized = list(row)

        while len(normalized) < num_columns:
            normalized.append("")

        normalized_rows.append(normalized)

    header = normalized_rows[0]

    output = []

    output.append(
        "| " + " | ".join(header) + " |"
    )

    output.append(
        "| " + " | ".join(["---"] * num_columns) + " |"
    )

    for row in normalized_rows[1:]:

        output.append(
            "| " + " | ".join(row) + " |"
        )

    return "\n".join(output)


# ============================================================
# PDF Extraction
# ============================================================

def extract_lines(pdf_path, table_overlap_threshold=0.5):
    """
    Extract normal text lines and tables from a PDF.

    Design:
    - Normal text line -> one element
    - One detected table -> one table element
    - Text lines belonging to a table are NOT added as normal text
    - Table structure is preserved in `table_data`
    """

    doc = pymupdf.open(pdf_path)
    elements = []

    for page_number, page in enumerate(doc, start=1):

        # print(f"\n📄 Processing page {page_number}")

        # ============================================================
        # 1. Detect tables
        # ============================================================

        table_finder = page.find_tables()
        tables = table_finder.tables

        table_infos = []

        for table_id, table in enumerate(tables):

            try:
                table_data = extract_table_data(table)

                if not table_data:
                    continue

                table_text = table_data_to_text(table_data)

                table_info = {
                    "table_id": table_id,
                    "bbox": table.bbox,
                    "data": table_data,
                    "text": table_text,
                }

                table_infos.append(table_info)

                print(f"📊 Table {table_id}:")
                print(f"   bbox: {table.bbox}")
                print(f"   data: {table_data}")

            except Exception as e:
                print(
                    f"⚠️ Failed to extract table {table_id}: {e}"
                )

        # ============================================================
        # 2. Extract normal text lines
        # ============================================================

        data = page.get_text("dict")

        for block_id, block in enumerate(data.get("blocks", [])):

            # Skip image blocks / non-text blocks
            if "lines" not in block:
                continue

            for line in block["lines"]:

                spans = line.get("spans", [])

                if not spans:
                    continue

                full_text = "".join(
                    span.get("text", "")
                    for span in spans
                ).strip()

                if not full_text:
                    continue

                # ----------------------------------------------------
                # Get line bbox
                # ----------------------------------------------------

                line_bbox = line.get("bbox")

                if not line_bbox:
                    continue

                # ----------------------------------------------------
                # Check whether this line belongs to a table
                # ----------------------------------------------------

                belongs_to_table = False

                for table_info in table_infos:

                    if bbox_overlaps(
                        line_bbox,
                        table_info["bbox"],
                        threshold=table_overlap_threshold
                    ):
                        belongs_to_table = True

                        print(
                            f"   ↳ Table line skipped: "
                            f"'{full_text}'"
                        )

                        break

                # IMPORTANT:
                # Table text is NOT added as normal text.
                if belongs_to_table:
                    continue

                # ----------------------------------------------------
                # Collect font information
                # ----------------------------------------------------

                fonts = [
                    span.get("font", "")
                    for span in spans
                ]

                sizes = [
                    span.get("size", 0)
                    for span in spans
                ]

                # ----------------------------------------------------
                # Create normal text element
                # ----------------------------------------------------

                element = {
                    "page": page_number,
                    "block": block_id,
                    "text": full_text,
                    "font": fonts,
                    "size": sizes,
                    "spans": spans,
                    "bbox": line_bbox,

                    "is_table": False,
                    "table_id": None,
                    "table_bbox": None,
                    "table_data": None,
                }

                elements.append(element)

        # ============================================================
        # 3. Add each table as ONE element
        # ============================================================

        for table_info in table_infos:

            table_element = {
                "page": page_number,
                "block": None,

                # Keep text version for debugging / fallback
                "text": table_info["text"],

                # Tables do not have normal font information
                "font": [],
                "size": [],
                "spans": [],

                "bbox": table_info["bbox"],

                "is_table": True,

                "table_id": table_info["table_id"],
                "table_bbox": table_info["bbox"],

                # IMPORTANT:
                # Preserve the actual table structure
                "table_data": table_info["data"],
            }

            elements.append(table_element)

    doc.close()

    # ================================================================
    # 4. Restore reading order
    # ================================================================

    elements.sort(
        key=lambda e: (
            e["page"],
            e["bbox"][1] if e.get("bbox") else 0,
            e["bbox"][0] if e.get("bbox") else 0
        )
    )

    # print(f"\n✅ Total elements extracted: {len(elements)}")

    return elements

# ============================================================
# Merge Text Lines
# ============================================================

def merge_lines(elements):
    """
    Merge consecutive normal text elements that belong to the same
    block and have the same font and font size.

    Table elements are preserved as individual elements and are
    never merged with normal text or other tables.
    """

    merged = []

    for element in elements:

        if not merged:
            merged.append(element)
            continue

        previous = merged[-1]

        # ----------------------------------------------------
        # Tables are always kept as individual elements
        # ----------------------------------------------------
        if (
            element.get("is_table", False)
            or previous.get("is_table", False)
        ):
            merged.append(element)
            continue

        # ----------------------------------------------------
        # Check same page and block
        # ----------------------------------------------------
        same_block = (
            element["page"] == previous["page"]
            and element["block"] == previous["block"]
        )

        # ----------------------------------------------------
        # Check same font
        # ----------------------------------------------------
        same_font = (
            element["font"] == previous["font"]
        )

        # ----------------------------------------------------
        # Check same font size
        # ----------------------------------------------------
        same_size = (
            element["size"] == previous["size"]
        )

        # ----------------------------------------------------
        # Merge compatible text elements
        # ----------------------------------------------------
        if same_block and same_font and same_size:

            previous["text"] += (
                " " + element["text"]
            )

            # Update bounding box
            previous["bbox"] = (
                previous["bbox"][0],
                previous["bbox"][1],
                element["bbox"][2],
                element["bbox"][3],
            )

        else:
            merged.append(element)

    return merged

# ============================================================
# Document Font Statistics
# ============================================================

def calculate_document_avg_size(elements):
    """
    Calculate average font size across the document.

    Table elements are ignored because table font sizes can
    distort heading detection.
    """

    sizes = []

    for element in elements:

        # Table text should not influence heading detection
        if element.get("is_table", False):
            continue

        for size in element.get("size", []):

            if size:
                sizes.append(size)

    if not sizes:
        return 0.0

    return sum(sizes) / len(sizes)


# ============================================================
# Bold Ratio
# ============================================================

def get_bold_ratio(element):
    """
    Calculate the ratio of characters using a bold font.
    """

    total_chars = 0
    bold_chars = 0

    for span in element.get("spans", []):

        text = span.get("text", "")

        is_bold = (
            "bold" in span.get("font", "").lower()
        )

        total_chars += len(text)

        if is_bold:
            bold_chars += len(text)

    if total_chars == 0:
        return 0.0

    return bold_chars / total_chars


# ============================================================
# Line Completion
# ============================================================

def is_line_complete(
    element,
    next_element,
    document_avg_size
):
    """
    Check whether the current line is likely complete.
    """

    text = element["text"].strip()

    # 1. Lines ending with punctuation
    if re.search(
        r"[\.\?\!\:；。？！」』]$",
        text
    ):
        return True

    # 2. Lines ending with conjunctions/prepositions
    words = text.split()

    last_word = (
        words[-1].rstrip(",;:-")
        if words
        else ""
    )

    incomplete_words = {
        "a", "an", "the", "and", "or", "but",
        "for", "nor", "on", "at", "to", "by",
        "in", "of", "with", "without", "from",
        "up", "down", "off", "over", "under",
        "about"
    }

    if last_word.lower() in incomplete_words:
        return False

    # 3. Comma / semicolon
    if re.search(
        r"[,;，、；]$",
        text
    ):
        return False

    # 4. Hyphen
    if text.endswith("-"):
        return False

    # 5. No next line
    if next_element is None:
        return True

    # 6. Different page
    if element["page"] != next_element["page"]:
        return True

    next_text = next_element["text"].strip()

    # 7. Lowercase continuation
    if next_text and next_text[0].islower():
        return False

    # 8. Uppercase = likely new sentence / heading
    if next_text and next_text[0].isupper():
        return True

    return True


# ============================================================
# Heading Score
# ============================================================

def get_heading_score(
    element,
    previous_element,
    next_element,
    document_avg_size,
):
    """
    Compute a heuristic score indicating how likely a text
    element is to be a heading.

    Tables are explicitly excluded.
    """

    # --------------------------------------------------------
    # IMPORTANT: Table can NEVER be a heading
    # --------------------------------------------------------

    if element.get("is_table", False):
        return -999

    score = 0

    text = element["text"].strip()

    if not text:
        return 0

    sizes = element.get("size", [])

    if not sizes:
        return 0

    avg_size = sum(sizes) / len(sizes)

    bold_ratio = get_bold_ratio(element)

    word_count = len(text.split())

    # --------------------------------------------------------
    # Exclude ISO / IEC identifiers
    # --------------------------------------------------------

    if re.search(
        r"^ISO/IEC\s+\d{4,5}:\d{4}\s*\(E\)(?:\s+.*)?$",
        text,
        re.I
    ):
        return 0

    # --------------------------------------------------------
    # Exclude emails / domains
    # --------------------------------------------------------

    if (
        "@" in text
        or re.search(
            r"\.(com|edu|my|org|net|gov)\b",
            text,
            re.I
        )
    ):
        return 0

    # --------------------------------------------------------
    # Exclude URLs
    # --------------------------------------------------------

    if re.search(
        r"https?://|www\.",
        text,
        re.I
    ):
        return 0

    # --------------------------------------------------------
    # Exclude numbers only
    # --------------------------------------------------------

    if re.match(
        r"^[\d\.]+$",
        text
    ):
        return 0

    # --------------------------------------------------------
    # Exclude bullet list items
    # --------------------------------------------------------

    if re.match(
        r"^\s*[•\-]\s+",
        text
    ):
        return 0

    # --------------------------------------------------------
    # Exclude descriptive body text
    # --------------------------------------------------------

    if re.search(
        r"\b(also|commonly|scientifically)\s+known\s+as\b",
        text,
        re.I
    ):
        return 0

    # --------------------------------------------------------
    # Exclude list-style text separated by hyphens
    # --------------------------------------------------------

    if " - " in text or " – " in text:

        parts = re.split(
            r"\s*[-–—]\s*",
            text
        )

        if len(parts) >= 2:

            first_part = parts[0].strip()

            if len(first_part.split()) <= 3:

                if not re.match(
                    r"^(Part|Chapter|Section)\s+\w+",
                    first_part,
                    re.I
                ):
                    return 0

    # --------------------------------------------------------
    # Exclude very long text
    # --------------------------------------------------------

    if len(text) > 120:

        has_title_features = (
            ":"
            in text
            or (
                text[0].isupper()
                and len(text.split()) <= 15
            )
        )

        if not has_title_features:
            return 0

    # --------------------------------------------------------
    # Exclude complete sentences
    # --------------------------------------------------------

    if text.endswith(".") and len(text) > 30:

        has_title_features = (
            ":"
            in text
            or (
                text[0].isupper()
                and len(text.split()) <= 12
            )
        )

        if not has_title_features:
            return 0

    # --------------------------------------------------------
    # Exclude long questions
    # --------------------------------------------------------

    if "?" in text and len(text) > 30:
        return 0

    # ========================================================
    # Deduction rules
    # ========================================================

    research_verbs = (
        r"\b(showed|demonstrated|found|revealed|"
        r"indicated|suggests|highlights|emphasizes|"
        r"illustrates|introduces|presents|proposes|"
        r"provides|shows|discusses|explores|"
        r"investigates|examines|describes|reports|"
        r"identifies|argues|claims|states|notes|"
        r"observes)\b"
    )

    if re.search(
        research_verbs,
        text,
        re.I
    ):
        score -= 3

    # Starts with This / The / These / Those
    if re.match(
        r"^(This|The|These|Those)\s+",
        text,
        re.I
    ):

        if re.search(
            r"\b(is|are|was|were|has|have|"
            r"includes|contains|represents|"
            r"provides|offers|presents)\b",
            text,
            re.I
        ):
            score -= 2

    # Transition words
    if re.match(
        r"^(However|Therefore|Thus|Moreover|"
        r"Furthermore|Additionally|Consequently|"
        r"Nevertheless|Nonetheless|Meanwhile|"
        r"Subsequently|Hence|Accordingly|"
        r"In addition|In contrast|On the other hand)",
        text,
        re.I
    ):
        score -= 2

    # Ending period
    if text.endswith("."):
        score -= 1

    # More than 18 words
    if word_count > 18 and ":" not in text:
        score -= 1

    # Abstract-style content
    if re.match(
        r"^(This study|The app|The application|"
        r"The paper|This paper|Our study)",
        text,
        re.I
    ):
        score -= 2

    # Smaller than document average
    if (
        document_avg_size > 0
        and avg_size < document_avg_size * 0.9
    ):
        score -= 1

    # Non-bold text with more than 8 words
    if bold_ratio < 0.3 and word_count > 8:
        score -= 1

    # ========================================================
    # Heading scoring rules
    # ========================================================

    # Numbered headings
    if re.match(
        r"^\d+(\.\d+)*\.?\s+",
        text
    ):

        score += 4

        if "(" in text and ")" in text:
            score += 1

        return score

    # Part / Chapter / Section
    if re.match(
        r"^(Part|Chapter|Section)\s+\w+",
        text,
        re.I
    ):

        score += 4

        if ":" in text:
            score += 1

        if " - " in text or " – " in text:
            score += 1

        return score

    # Fully uppercase
    if text.isupper() and len(text) > 3:
        score += 3

    # Short title style
    if text and text[0].isupper():

        word_count = len(text.split())

        if word_count <= 15 and not text.endswith("."):

            score += 2

            if ":" in text:
                score += 1

            if re.match(
                r"^(The|A|An)\s+",
                text
            ):
                score += 1

    # Larger font
    if (
        document_avg_size > 0
        and avg_size > document_avg_size * 1.3
    ):
        score += 3

    elif (
        document_avg_size > 0
        and avg_size > document_avg_size * 1.1
    ):
        score += 1

    # Bold
    if bold_ratio >= 0.9:
        score += 2

    elif bold_ratio >= 0.7:
        score += 1

    # Complete standalone line
    if is_line_complete(
        element,
        next_element,
        document_avg_size
    ):

        if (
            not text.endswith(".")
            and not text.endswith(":")
        ):
            score += 1

    # Short text
    word_count = len(text.split())

    if 2 <= word_count <= 12:
        score += 1

    return score

