import re
import tiktoken
from typing import List, Dict, Optional, Tuple
from .pdf_loader import (
    get_heading_score,
    calculate_document_avg_size,
    get_bold_ratio
)


def get_token_count(text):
    """Calculate the number of tokens in the text."""
    enc = tiktoken.encoding_for_model("gpt-3.5-turbo")
    return len(enc.encode(text))

def clean_html_tags(text):
    """
    Remove HTML tags from extracted text.

    This only cleans the final document text.
    It does NOT affect:
    - heading detection
    - section creation
    - OCR extraction
    - rule-based processing before flattening

    Example:
        <table><tr><td>Item</td><td>Qty</td></tr></table>

    becomes:

        Item Qty
    """

    if not text:
        return ""

    # Remove HTML tags such as:
    # <table>
    # </table>
    # <tr>
    # </tr>
    # <td>
    # </td>
    # <td colspan="2">
    # etc.
    text = re.sub(r"<[^>]+>", " ", text)

    # Clean multiple spaces
    text = re.sub(r"[ \t]+", " ", text)

    # Clean excessive blank lines
    text = re.sub(r"\n\s*\n+", "\n\n", text)

    return text.strip()

def flatten_sections(sections, document_title, max_tokens=512):
    """
    Flatten sections into document chunks.

    First split by heading, then split long sections by token limit.

    HTML tags are removed ONLY from the final document text
    before storing into ChromaDB.

    This does not affect rule-based section detection.
    """

    documents = []

    for section in sections:

        heading = section["heading"]
        content = section.get("content", "").strip()
        page = section.get("page", 1)

        # Skip sections with empty content
        if not content:
            continue

        # ============================================================
        # Clean HTML from content BEFORE creating the final document
        # ============================================================
        clean_content = clean_html_tags(content)

        # Clean heading as well, just in case
        clean_heading = clean_html_tags(heading)

        # Skip if cleaning resulted in empty content
        if not clean_content:
            continue

        full_text = clean_heading + "\n\n" + clean_content

        # Get the number of tokens
        token_count = get_token_count(full_text)

        if token_count <= max_tokens:

            # No need to split the section
            documents.append({
                "text": full_text,
                "metadata": {
                    "title": document_title or "",
                    "heading": clean_heading,
                    "page": page,
                    "chunk_type": "full"
                }
            })

        else:

            # Split the section if the token count exceeds the limit
            chunks = split_text_by_heading(
                full_text,
                clean_heading,
                max_tokens
            )

            for i, chunk_text in enumerate(chunks):

                # Make absolutely sure no HTML survives
                chunk_text = clean_html_tags(chunk_text)

                documents.append({
                    "text": chunk_text,
                    "metadata": {
                        "title": document_title or "",
                        "heading": clean_heading,
                        "page": page,
                        "chunk_type": "split",
                        "chunk_index": i,
                        "total_chunks": len(chunks)
                    }
                })

            print(
                f"✂️ Split '{clean_heading[:30]}...' "
                f"into {len(chunks)} chunks"
            )

    return documents

def split_text_by_heading(full_text, heading, max_tokens=512):
    """
    Split text based on sentences while keeping the heading
    in every generated chunk.

    HTML should already be removed before this function is called.
    """

    # Extract the content by removing the heading
    content = full_text.replace(heading, "", 1).strip()

    # Safety cleanup
    content = clean_html_tags(content)
    heading = clean_html_tags(heading)

    # Return the original text if the content is within the token limit
    if get_token_count(content) <= max_tokens:
        return [heading + "\n\n" + content]

    # Split the content into sentences
    sentences = re.split(r'(?<=[.!?])\s+', content)

    chunks = []

    current_chunk = [heading]
    current_tokens = get_token_count(heading)

    for sentence in sentences:

        if not sentence.strip():
            continue

        sentence = clean_html_tags(sentence)

        sentence_tokens = get_token_count(sentence)

        if (
            current_tokens + sentence_tokens > max_tokens
            and len(current_chunk) > 1
        ):
            chunks.append("\n\n".join(current_chunk))

            current_chunk = [heading]
            current_tokens = get_token_count(heading)

        current_chunk.append(sentence)
        current_tokens += sentence_tokens

    if len(current_chunk) > 1:
        chunks.append("\n\n".join(current_chunk))

    return chunks

def find_title_from_candidates(heading_candidates):
    """
    Identify the document title from heading candidates using rules.
    """
    for i, h in enumerate(heading_candidates):
        text = h["text"].strip()
        
        if i > 3:
            break
        
        word_count = len(text.split())
        if word_count < 3 or word_count > 20:
            continue
        
        # Exclude headings containing email or contact information
        if re.search(r'^(Email|Contact|Correspondence)', text, re.I):
            continue

        # Exclude author name formats
        # Example: "John Smith and Jane Doe" or "Smith, John"
        if re.match(r'^[A-Z][a-z]+\s+and\s+[A-Z][a-z]+', text):
            continue
        if re.match(r'^[A-Z][a-z]+,\s+[A-Z][a-z]+', text):
            continue
        
        # Exclude numbered headings
        if re.match(r'^\d+\.\s+', text):
            continue
        
        # Exclude Part, Chapter, and Section headings
        if re.match(r'^(Part|Chapter|Section)\s+\w+', text, re.I):
            continue
        
        # Exclude common introductory and concluding headings
        if re.match(r'^(Introduction|Conclusion)($|:)', text, re.I):
            continue
        
        # Require at least one uppercase letter
        if not re.search(r'[A-Z]', text):
            continue
        
        # Exclude sentences ending with a period
        if text.endswith('.'):
            continue
        
        return text
    
    return None


def is_toc_page(elements, page_num):
    """
    Detect whether a page is a Table of Contents page.
    """
    page_elements = [e for e in elements if e["page"] == page_num]
    text = " ".join([e["text"] for e in page_elements])
    
    # Count numbered headings
    numbered_headings = re.findall(r'\b\d+\.\d+\s+[A-Z]', text)
    
    # If fewer than two numbered headings are found, it is unlikely to be a TOC page
    # Normal content pages usually contain only one or two numbered headings
    if len(numbered_headings) < 2:
        return False

    toc_patterns = [
        r'Table\s+of\s+Contents',
        r'Contents',
        r'\.{10,}\s*\d+',
        r'^\s*\d+\.\d+\s+\w+',
        r'Part\s+\w+\s+\.{10,}\s*\d+',
    ]
    
    for pattern in toc_patterns:
        if re.search(pattern, text, re.I):
            return True
    
    return False


def is_toc_line(text):
    """
    Detect whether a line belongs to a table of contents.
    """
    text = text.strip()
    
    # Do not filter numbered headings such as 5.2 or 5.1.1
    if re.match(r'^\d+(\.\d+)*\.?\s+', text):
        return False

    # Detect dotted leaders followed by a page number
    if re.search(r'\.{5,}\s*\d+', text):
        return True
    
    # Detect lines ending with a number
    if re.search(r'\d+$', text) and len(text.split()) > 3:
        return True
    
    # Detect Part headings with dotted leaders and page numbers
    if re.search(r'Part\s+\w+\s+\.{5,}\s*\d+', text, re.I):
        return True
    
    return False


def is_table_heading(text):
    """
    Check whether a text line is a table caption.
    """
    return bool(re.match(r'^Table\s+\d+[:.]', text.strip(), re.I))


def format_table_as_markdown(table_lines):
    """
    Convert extracted table content into Markdown table format.
    """
    if not table_lines:
        return ""
    
    data_rows = []
    
    for line in table_lines:
        line = line.strip()
        
        if not line:
            continue
        
        # Skip table captions
        if is_table_heading(line):
            continue
        
        parts = line.split()
        
        if parts:
            data_rows.append(parts)
    
    if len(data_rows) < 2:
        return "\n".join(table_lines)
    
    header = data_rows[0]
    
    markdown = "| " + " | ".join(header) + " |\n"
    markdown += "|" + " --- |" * len(header) + "\n"
    
    for row in data_rows[1:]:
        # Pad rows with empty values if they have fewer columns than the header
        while len(row) < len(header):
            row.append("")
        
        markdown += "| " + " | ".join(row[:len(header)]) + " |\n"
    
    return markdown


def extract_table_content(elements):
    """
    Extract table content from PDF elements.
    """
    table_data = []
    table_title = None
    
    for element in elements:
        if not element.get("is_table", False):
            continue
        
        text = element["text"].strip()
        
        if is_table_heading(text):
            table_title = text
            continue
        
        table_data.append(text)
    
    if not table_data:
        return None
    
    result = ""
    
    # Add the table title if available
    if table_title:
        result += f"**{table_title}**\n\n"
    
    markdown_table = format_table_as_markdown(table_data)
    
    if markdown_table:
        result += markdown_table
    else:
        result += "\n".join(table_data)
    
    return result


def create_sections(elements):
    """
    Create sections and handle headings that are split across multiple elements.
    """
    document_title = None
    document_avg_size = calculate_document_avg_size(elements)

    # ========== Step 1: Remove TOC ==========
    filtered_elements = []
    toc_pages = set()
    
    for element in elements:
        if is_toc_page(elements, element["page"]):
            toc_pages.add(element["page"])
    
    for element in elements:
        # Skip elements from detected TOC pages
        if element["page"] in toc_pages:
            continue
        
        # Skip individual TOC lines
        if is_toc_line(element["text"]):
            continue
        
        filtered_elements.append(element)
    
    if toc_pages:
        print(f"📋 Skipped TOC pages: {toc_pages}")
    
    # ============================================================
    # DEBUG: Display heading scores for all elements
    # ============================================================
    print("\n" + "=" * 70)
    print("🔍 HEADING SCORE DEBUG:")
    print("=" * 70)
    
    heading_scores = []

    for i, element in enumerate(filtered_elements):
        
        text_preview = element["text"][:50].replace('\n', ' ')
        
        previous_element = (
            filtered_elements[i - 1]
            if i > 0
            else None
        )

        next_element = (
            filtered_elements[i + 1]
            if i < len(filtered_elements) - 1
            else None
        )

        score = get_heading_score(
            element,
            previous_element,
            next_element,
            document_avg_size,
        )

        heading_scores.append(score)

        is_heading = "✅" if score >= 4 else "  "
        
        avg_size = (
            sum(element["size"]) / len(element["size"])
            if element["size"]
            else 0
        )
        
        bold_ratio = get_bold_ratio(element)
        
        if i < 100:
            print(
                f"  {i+1:3d}. {is_heading} | "
                f"Score: {score:2d} | "
                f"Size: {avg_size:4.1f} | "
                f"Bold: {bold_ratio:.0%} | "
                f"{text_preview}..."
            )
    
    print("=" * 70 + "\n")
    
    # ========== Step 2: Merge split headings ==========
    merged_elements = []
    i = 0
    merged_titles = []

    while i < len(filtered_elements):
        current = filtered_elements[i]
        current_score = heading_scores[i]
        is_current_heading = current_score >= 4
        
        if is_current_heading and i + 1 < len(filtered_elements):
            next_elem = filtered_elements[i + 1]
            
            next_score = get_heading_score(
                next_elem,
                current,
                None,
                document_avg_size
            )
            
            is_next_heading = next_score >= 4
            
            if is_next_heading and current["page"] == next_elem["page"]:
                y_diff = abs(
                    current["bbox"][1] - next_elem["bbox"][1]
                )
                
                if y_diff < 30:
                    # Merge split headings
                    current["text"] = (
                        current["text"].strip()
                        + " "
                        + next_elem["text"].strip()
                    )
                    
                    print(
                        f"🔗 Merged split title: "
                        f"'{current['text'][:60]}...'"
                    )
                    
                    merged_titles.append(current["text"])
                    i += 1
        
        merged_elements.append(current)
        i += 1

    filtered_elements = merged_elements

    # Recalculate heading scores after merging elements
    heading_scores = []
    
    for i, element in enumerate(filtered_elements):
        previous_element = (
            filtered_elements[i - 1]
            if i > 0
            else None
        )
        
        next_element = (
            filtered_elements[i + 1]
            if i < len(filtered_elements) - 1
            else None
        )
        
        score = get_heading_score(
            element,
            previous_element,
            next_element,
            document_avg_size
        )
        
        heading_scores.append(score)

    # ========== Step 3: Detect heading candidates ==========
    heading_candidates = []

    for i, element in enumerate(filtered_elements):
        if element.get("is_table", False):
            continue
        
        score = heading_scores[i]
        
        # Force merged headings to be added as heading candidates
        if element["text"] in merged_titles:
            print(
                f"✅ Force adding merged title: "
                f"'{element['text'][:60]}...'"
            )
            heading_candidates.append(element)
        
        elif score >= 4:
            heading_candidates.append(element)
    
    # ========== Step 4: Find document title ==========
    document_title = find_title_from_candidates(heading_candidates)
    print(f"📌 Document Title: {document_title}")
    
    # ========== Step 5: Create sections with table detection ==========
    temp_sections = []
    current_section = None
    table_title = None
    table_rows = []
    in_table = False
    
    for i, element in enumerate(filtered_elements):
        
        if element.get("is_table", False):
            if current_section:
                table_text = extract_table_content([element])
                if table_text:
                    current_section["content"] += table_text + "\n\n"
            continue
        
        score = heading_scores[i]
        is_heading = score >= 4
        text = element["text"].strip()
        
        # ✅ 检测 Table 标题
        if is_heading and re.match(r'^Table\s+\d+', text, re.I):
            # 保存之前的 section
            if current_section:
                # 如果有表格数据待保存
                if in_table and table_rows:
                    table_markdown = format_table_rows(table_title, table_rows)
                    current_section["content"] += table_markdown + "\n\n"
                temp_sections.append(current_section)
            
            table_title = text
            table_rows = []
            in_table = True
            
            current_section = {
                "heading": text,
                "page": element["page"],
                "content": ""
            }
            print(f"📂 Created Table section: {text[:50]}... (score: {score})")
            continue
        
        # ✅ 如果在 Table 模式下，收集数据行
        if in_table and not is_heading:
            # 检查是否是表格数据 (包含数字、n=、p-value 等)
            if (re.search(r'\d+\.?\d*', text) and 'n=' in text) or \
               re.search(r'p-value|t-value|p\s*<', text, re.I) or \
               re.search(r'\(\d+\.?\d*%\)', text) or \
               re.match(r'^[A-Z][a-z]+\s*\(n\s*=', text):
                table_rows.append(text)
                print(f"   📊 Table data: {text[:40]}...")
                continue
            else:
                # 不是表格数据，退出 Table 模式
                # 保存已收集的表格数据
                if table_rows:
                    table_markdown = format_table_rows(table_title, table_rows)
                    current_section["content"] += table_markdown + "\n\n"
                    table_rows = []
                in_table = False
        
        if is_heading:
            # 保存之前的 section
            if current_section:
                if in_table and table_rows:
                    table_markdown = format_table_rows(table_title, table_rows)
                    current_section["content"] += table_markdown + "\n\n"
                temp_sections.append(current_section)
            
            current_section = {
                "heading": text,
                "page": element["page"],
                "content": ""
            }
            print(f"📂 Created section: {text[:50]}... (score: {score})")
        
        else:
            if current_section:
                current_section["content"] += element["text"] + " "
    
    # 保存最后一个 section
    if current_section:
        if in_table and table_rows:
            table_markdown = format_table_rows(table_title, table_rows)
            current_section["content"] += table_markdown + "\n\n"
        temp_sections.append(current_section)
    
    # ========== Step 6: Merge empty sections ==========
    sections = merge_empty_sections(temp_sections)
    
    # ============================================================
    # Display final section preview
    # ============================================================
    print("\n" + "=" * 70)
    print("📊 FINAL SECTIONS:")
    print("=" * 70)
    
    for i, sec in enumerate(sections):
        heading = sec["heading"][:50]
        content = sec.get("content", "").strip()
        
        content_preview = (
            content[:80].replace('\n', ' ')
            if content
            else "(EMPTY)"
        )
        
        word_count = len(content.split()) if content else 0
        
        print(
            f"  {i+1:3d}. "
            f"[{word_count:4d} words] "
            f"{heading}..."
        )
        
        print(
            f"         Content: "
            f"{content_preview}..."
        )
        
        print("-" * 50)
    
    print("=" * 70)
    print(f"✅ Total sections created: {len(sections)}")
    print("=" * 70 + "\n")
    
    return sections, document_title

def format_table_rows(table_title: str, rows: List[str]) -> str:
    """
    将表格行格式化为 Markdown 表格。
    正确处理包含空格的单元格 (如 "Carbon paper, blue, 8.5x11")
    """
    if not rows or not table_title:
        return ""
    
    # 检测分割符
    # 如果行中包含多个空格，尝试按多个空格分割
    parsed_rows = []
    for row in rows:
        # 尝试按多个空格分割 (保留列结构)
        parts = re.split(r'\s{2,}', row.strip())
        if len(parts) > 1:
            parsed_rows.append([p.strip() for p in parts if p.strip()])
        else:
            # 按单个空格分割 (fallback)
            parts = row.split()
            if parts:
                parsed_rows.append(parts)
    
    if len(parsed_rows) < 2:
        return f"### {table_title}\n\n" + "\n".join(rows)
    
    # 确定最大列数
    max_cols = max(len(r) for r in parsed_rows)
    
    # 检测表头
    header_keywords = ['characteristics', 'control', 'experimental', 'group', 'variable', 'category']
    header_idx = 0
    for i, row in enumerate(parsed_rows):
        row_text = ' '.join(row).lower()
        if any(kw in row_text for kw in header_keywords):
            header_idx = i
            break
    
    # 构建 Markdown 表格
    markdown = f"### {table_title}\n\n"
    
    # 表头
    header = parsed_rows[header_idx]
    while len(header) < max_cols:
        header.append("")
    markdown += "| " + " | ".join(header) + " |\n"
    markdown += "|" + " --- |" * len(header) + "\n"
    
    # 数据行
    for i, row in enumerate(parsed_rows):
        if i == header_idx:
            continue
        while len(row) < max_cols:
            row.append("")
        markdown += "| " + " | ".join(row[:max_cols]) + " |\n"
    
    return markdown

def merge_empty_sections(sections):
    """
    Merge empty sections with the next section that contains content.
    This helps combine headings that were incorrectly split,
    such as "The Impact..." and "Reflection...".
    """
    if len(sections) < 2:
        return sections
    
    merged = []
    i = 0
    
    while i < len(sections):
        current = sections[i]
        content = current.get("content", "").strip()
        
        # Keep sections that already contain content
        if content:
            merged.append(current)
            i += 1
            continue
        
        # The current section contains only a heading
        empty_heading = current["heading"]
        current_page = current.get("page", 1)
        
        print(
            f"🔍 Found empty section: "
            f"'{empty_heading[:40]}...'"
        )
        
        # Find the next section that contains content
        found = False
        
        for j in range(i + 1, len(sections)):
            next_section = sections[j]
            next_content = next_section.get("content", "").strip()
            next_page = next_section.get("page", 1)
            
            # If the next section contains content
            if next_content:
                
                # Check whether the sections are on the same page
                # or on consecutive pages
                if (
                    next_page == current_page
                    or next_page == current_page + 1
                ):
                    # Merge the headings
                    next_section["heading"] = (
                        empty_heading
                        + " "
                        + next_section["heading"]
                    )
                    
                    # Add the merged section to the result
                    merged.append(next_section)
                    
                    found = True
                    
                    # Skip all sections that have been merged
                    i = j + 1
                    break
                
                else:
                    # Keep the empty heading if the next section
                    # is located on a different page
                    merged.append(current)
                    found = True
                    i += 1
                    break
        
        # Keep the heading if no suitable section was found
        if not found:
            merged.append(current)
            i += 1
    
    return merged