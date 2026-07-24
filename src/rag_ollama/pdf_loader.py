import fitz
import re


def fix_ligatures(text):
    """
    Replace common PDF ligatures and special typography characters
    with their standard ASCII equivalents.
    """
    replacements = {
        'ﬁ': 'fi',
        'ﬂ': 'fl',
        'ﬃ': 'ffi',
        'ﬄ': 'ffl',
        'ﬀ': 'ff',
        'ﬅ': 'ft',
        'ﬆ': 'st',
        'Œ': 'OE',
        'œ': 'oe',
        'Æ': 'AE',
        'æ': 'ae',
        '–': '-',
        '—': '-',
    }

    for old, new in replacements.items():
        text = text.replace(old, new)

    return text


def clean_text(text):
    # Fix ligatures
    text = fix_ligatures(text)

    # Remove repeated separators
    text = re.sub(r'[`\-_,]{5,}', '', text)

    # Normalize copyright symbols
    text = re.sub(r'\([cC]\)', '(c)', text)

    # Remove empty citation brackets
    text = re.sub(r'\[\s*\]', '', text)

    # Normalize whitespace
    text = re.sub(r'\s+', ' ', text)

    # Remove duplicated page markers
    text = re.sub(r'===== Page \d+ =====', '', text)

    # Remove unsupported Unicode characters
    text = re.sub(
        r'[^\x00-\x7F\u2000-\u206F\u20A0-\u20CF]',
        '',
        text
    )

    return text


def extract_lines(pdf_path):
    """
    Extract every text line from the PDF together with
    font, size, bounding box and table information.
    """
    doc = fitz.open(pdf_path)
    elements = []

    for page_num, page in enumerate(doc):
        data = page.get_text("dict")

        # Detect tables on the current page
        tables = page.find_tables()
        table_bboxes = [table.bbox for table in tables]

        for block_id, block in enumerate(data["blocks"]):
            # Skip blocks that do not contain text lines
            if "lines" not in block:
                continue

            for line in block["lines"]:
                text = ""
                fonts = []
                sizes = []
                spans = []

                # Check whether the line is inside a table
                is_table = False
                line_bbox = line["bbox"]

                for table_bbox in table_bboxes:
                    if (
                        line_bbox[0] >= table_bbox[0]
                        and line_bbox[1] >= table_bbox[1]
                        and line_bbox[2] <= table_bbox[2]
                        and line_bbox[3] <= table_bbox[3]
                    ):
                        is_table = True
                        break

                # Extract all spans from the current line
                for span in line["spans"]:
                    span_text = span["text"]
                    span_text = clean_text(span_text)

                    text += span_text
                    fonts.append(span["font"])
                    sizes.append(span["size"])

                    spans.append({
                        "text": span_text,
                        "font": span["font"],
                        "size": span["size"]
                    })

                if text.strip():
                    full_text = clean_text(text)

                    if full_text:
                        elements.append({
                            "page": page_num + 1,
                            "block": block_id,
                            "text": full_text,
                            "font": fonts,
                            "size": sizes,
                            "spans": spans,
                            "bbox": line["bbox"],
                            "is_table": is_table
                        })

    return elements


def merge_lines(elements):

    merged = []

    for element in elements:

        if not merged:
            merged.append(element)
            continue

        previous = merged[-1]

        # Check whether both elements belong to the same block
        same_block = (
            element["page"] == previous["page"]
            and element["block"] == previous["block"]
        )

        # Check whether both elements use the same font
        same_font = (
            element["font"] == previous["font"]
        )

        # Check whether both elements use the same font size
        same_size = (
            element["size"] == previous["size"]
        )

        if same_block and same_font and same_size:

            # Merge the text of the two elements
            previous["text"] += " " + element["text"]

            # Update the bounding box to cover both elements
            previous["bbox"] = (
                previous["bbox"][0],
                previous["bbox"][1],
                element["bbox"][2],
                element["bbox"][3]
            )

        else:
            merged.append(element)

    return merged


def calculate_document_avg_size(elements):

    sizes = []

    for element in elements:
        for size in element["size"]:
            sizes.append(size)

    return sum(sizes) / len(sizes)


def get_bold_ratio(element):
    """
    Calculate the ratio of characters using a bold font.
    """
    total_chars = 0
    bold_chars = 0

    # Weight the bold ratio by character count
    for span in element.get("spans", []):
        text = span["text"]
        is_bold = "Bold" in span["font"]

        total_chars += len(text)

        if is_bold:
            bold_chars += len(text)

    if total_chars == 0:
        return 0.0

    return bold_chars / total_chars


def is_line_complete(element, next_element, document_avg_size):
    """
    Check whether the current line is likely to be complete.
    """
    text = element["text"].strip()

    # 1. Lines ending with punctuation are considered complete
    if re.search(r'[\.\?\!\:；。？！」』]$', text):
        return True

    # 2. Lines ending with conjunctions or prepositions are considered incomplete
    last_word = (
        text.split()[-1].rstrip(",;:-")
        if text.split()
        else ""
    )

    incomplete_words = {
        "a", "an", "the", "and", "or", "but", "for", "nor",
        "on", "at", "to", "by", "in", "of", "with", "without",
        "from", "up", "down", "off", "over", "under", "about"
    }

    if last_word.lower() in incomplete_words:
        return False

    # 3. Lines ending with a comma or semicolon are considered incomplete
    if re.search(r'[,;，、；]$', text):
        return False

    # 4. Lines ending with a hyphen are considered incomplete
    if text.endswith("-"):
        return False

    # 5. If there is no next line, consider the current line complete
    if next_element is None:
        return True

    # 6. If the next line is on another page, consider the current line complete
    if element["page"] != next_element["page"]:
        return True

    # 7. If the next line starts with a lowercase letter,
    # the current line is likely to continue
    next_text = next_element["text"].strip()

    if next_text and next_text[0].islower():
        return False

    # 8. If the next line starts with an uppercase letter,
    # it is likely to be a new heading or sentence
    if next_text and next_text[0].isupper():
        return True

    # Default to considering the line complete
    return True


def get_heading_score(
    element,
    previous_element,
    next_element,
    document_avg_size,
):
    """
    Compute a heuristic score indicating how likely
    a text element is to be a heading.
    Higher scores indicate a higher probability.
    """
    score = 0
    text = element["text"].strip()
    avg_size = sum(element["size"]) / len(element["size"])
    bold_ratio = get_bold_ratio(element)
    word_count = len(text.split())

    # Exclude ISO/IEC standard identifiers
    if re.search(
        r'^ISO/IEC\s+\d{4,5}:\d{4}\s*\(E\)(?:\s+.*)?$',
        text,
        re.I
    ):
        return 0

    # Exclude email addresses and common domain names
    if (
        '@' in text
        or re.search(r'\.(com|edu|my|org|net|gov)\b', text, re.I)
    ):
        return 0

    # Exclude URLs
    if re.search(r'https?://|www\.', text, re.I):
        return 0

    # Exclude text containing only numbers and periods
    if re.match(r'^[\d\.]+$', text):
        return 0

    # Exclude bullet list items
    if re.match(r'^\s*[•\-]\s+', text):
        return 0

    # Exclude descriptive body text
    if re.search(
        r'\b(also|commonly|scientifically)\s+known\s+as\b',
        text,
        re.I
    ):
        return 0

    # Exclude list-style text separated by hyphens
    if ' - ' in text or ' – ' in text:
        parts = re.split(r'\s*[-–—]\s*', text)

        if len(parts) >= 2:
            first_part = parts[0].strip()

            # Ignore short list items with three words or fewer,
            # except numbered or section headings
            if len(first_part.split()) <= 3:

                # Keep headings such as "Part I", "Chapter 1",
                # and "Section 2"
                if not re.match(
                    r'^(Part|Chapter|Section)\s+\w+',
                    first_part,
                    re.I
                ):
                    return 0

    # Exclude very long text
    if len(text) > 120:
        has_title_features = (
            ':'
            in text
            or (
                text[0].isupper()
                and len(text.split()) <= 15
            )
        )

        if not has_title_features:
            return 0

    # Exclude complete sentences
    if text.endswith('.') and len(text) > 30:
        has_title_features = (
            ':'
            in text
            or (
                text[0].isupper()
                and len(text.split()) <= 12
            )
        )

        if not has_title_features:
            return 0

    # Exclude long questions
    if '?' in text and len(text) > 30:
        return 0

    # ============================================================
    # Deduction rules
    # These rules reduce the heading score without immediately
    # excluding the element.
    # ============================================================

    # 1. Deduct points for research-related verbs
    research_verbs = (
        r'\b(showed|demonstrated|found|revealed|indicated|suggests|'
        r'highlights|emphasizes|illustrates|introduces|presents|'
        r'proposes|provides|shows|discusses|explores|investigates|'
        r'examines|describes|reports|identifies|argues|claims|'
        r'states|notes|observes)\b'
    )

    if re.search(research_verbs, text, re.I):
        score -= 3

    # 2. Deduct points for text starting with "This", "The", etc.
    if re.match(r'^(This|The|These|Those)\s+', text, re.I):
        if re.search(
            r'\b(is|are|was|were|has|have|includes|contains|'
            r'represents|provides|offers|presents)\b',
            text,
            re.I
        ):
            score -= 2

    # 3. Deduct points for text starting with transition words
    if re.match(
        r'^(However|Therefore|Thus|Moreover|Furthermore|Additionally|'
        r'Consequently|Nevertheless|Nonetheless|Meanwhile|Subsequently|'
        r'Hence|Accordingly|In addition|In contrast|On the other hand)',
        text,
        re.I
    ):
        score -= 2

    # 4. Deduct points for lines ending with a period
    # Headings rarely end with a period
    if text.endswith('.'):
        score -= 1

    # 5. Deduct points for text with more than 18 words and no colon
    if word_count > 18 and ':' not in text:
        score -= 1

    # 6. Deduct points for abstract-style content
    # Examples: "This study", "The app", and "This paper"
    if re.match(
        r'^(This study|The app|The application|The paper|'
        r'This paper|Our study)',
        text,
        re.I
    ):
        score -= 2

    # 7. Deduct points when the font size is smaller than the document average
    if avg_size < document_avg_size * 0.9:
        score -= 1

    # 8. Deduct points for non-bold text with more than 8 words
    if bold_ratio < 0.3 and word_count > 8:
        score -= 1

    # ============================================================
    # Heading scoring rules
    # ============================================================

    # Numbered headings
    if re.match(r'^\d+(\.\d+)*\.?\s+', text):
        score += 4

        if '(' in text and ')' in text:
            score += 1

        return score

    # Part / Chapter / Section headings
    if re.match(
        r'^(Part|Chapter|Section)\s+\w+',
        text,
        re.I
    ):
        score += 4

        if ':' in text:
            score += 1

        if ' - ' in text or ' – ' in text:
            score += 1

        return score

    # Fully uppercase headings
    if text.isupper() and len(text) > 3:
        score += 3

    # Short title-style text
    if text and text[0].isupper():
        word_count = len(text.split())

        if word_count <= 15 and not text.endswith('.'):
            score += 2

            if ':' in text:
                score += 1

            # Add a bonus for titles starting with "The", "A", or "An"
            if re.match(r'^(The|A|An)\s+', text):
                score += 1

    # Larger font size
    if avg_size > document_avg_size * 1.3:
        score += 3

    elif avg_size > document_avg_size * 1.1:
        score += 1

    # Bold font
    bold_ratio = get_bold_ratio(element)

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
        if not text.endswith(".") and not text.endswith(":"):
            score += 1

    # Bonus for short text
    word_count = len(text.split())

    if 2 <= word_count <= 12:
        score += 1

    return score