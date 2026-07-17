import re
import tiktoken
from .pdf_loader import (
    get_heading_score,
    calculate_document_avg_size,
    get_bold_ratio
)


def get_token_count(text):
    """Calculate the number of tokens in a text."""
    enc = tiktoken.encoding_for_model("gpt-3.5-turbo")
    return len(enc.encode(text))


def flatten_sections(sections, document_title, max_tokens=512):
    """
    Flatten sections into document chunks.
    First split by heading, then split long sections by token limit.
    """
    documents = []
    
    for section in sections:
        heading = section["heading"]
        content = section.get("content", "").strip()
        page = section.get("page", 1)
        
        # skip empty content heading
        if not content:
            continue
        
        full_text = heading + "\n\n" + content

        # get number of token
        token_count = get_token_count(full_text)
        
        if token_count <= max_tokens:
            # no need split 
            documents.append({
                "text": full_text,
                "metadata": {
                    "title": document_title or "",
                    "heading": heading,
                    "page": page,
                    "chunk_type": "full"
                }
            })
        else:
            # split if token_count > max_tokens
            chunks = split_text_by_heading(full_text, heading, max_tokens)
            
            for i, chunk_text in enumerate(chunks):
                documents.append({
                    "text": chunk_text,
                    "metadata": {
                        "title": document_title,
                        "heading": heading,
                        "page": page,
                        "chunk_type": "split",
                        "chunk_index": i,
                        "total_chunks": len(chunks)
                    }
                })
            
            print(f"✂️ Split '{heading[:30]}...' into {len(chunks)} chunks")
    
    return documents


def split_text_by_heading(full_text, heading, max_tokens=512):
    """
    Split text based on sentences while keeping the heading
    in every generated chunk.
    """
    # 提取内容（移除开头的标题）
    content = full_text.replace(heading, "", 1).strip()
    
    # content within limit, no split
    if get_token_count(content) <= max_tokens:
        return [full_text]
    
    # Split text into sentences
    sentences = re.split(r'(?<=[.!?])\s+', content)
    
    chunks = []
    current_chunk = [heading]
    current_tokens = get_token_count(heading)
    
    for sentence in sentences:
        if not sentence.strip():
            continue
            
        sentence_tokens = get_token_count(sentence)
        
        if current_tokens + sentence_tokens > max_tokens and current_chunk:
            chunks.append('\n\n'.join(current_chunk))
            current_chunk = [heading]
            current_tokens = get_token_count(heading)
        
        current_chunk.append(sentence)
        current_tokens += sentence_tokens
    
    if current_chunk:
        chunks.append('\n\n'.join(current_chunk))
    
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
        
        # ✅ 排除包含 "Email:", "Contact:" 的标题
        if re.search(r'^(Email|Contact|Correspondence)', text, re.I):
            continue

        # ✅ 排除作者名格式 (包含 "and" 或两个大写字母开头)
        if re.match(r'^[A-Z][a-z]+\s+and\s+[A-Z][a-z]+', text):
            continue
        if re.match(r'^[A-Z][a-z]+,\s+[A-Z][a-z]+', text):
            continue
                
        if re.match(r'^\d+\.\s+', text):
            continue
        
        if re.match(r'^(Part|Chapter|Section)\s+\w+', text, re.I):
            continue
        
        if re.match(r'^(Introduction|Conclusion)($|:)', text, re.I):
            continue
        
        if not re.search(r'[A-Z]', text):
            continue
        
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
    
     # ✅ 统计数字编号标题的数量
    numbered_headings = re.findall(r'\b\d+\.\d+\s+[A-Z]', text)
    
    # ✅ 如果少于 2 个，不是 TOC（正文页面通常只有 1-2 个标题）
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
    
    # ✅ 如果是数字编号标题 (5.2, 5.1.1 等)，不要过滤
    if re.match(r'^\d+(\.\d+)*\.?\s+', text):
        return False

    if re.search(r'\.{5,}\s*\d+', text):
        return True
    
    if re.search(r'\d+$', text) and len(text.split()) > 3:
        return True
    
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
    创建 sections，处理被拆分的标题
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
        if element["page"] in toc_pages:
            continue
        if is_toc_line(element["text"]):
            continue
        filtered_elements.append(element)
    
    if toc_pages:
        print(f"📋 Skipped TOC pages: {toc_pages}")
    
    # ============================================================
    # ✅ DEBUG: 显示所有元素的 heading score
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
        avg_size = sum(element["size"]) / len(element["size"]) if element["size"] else 0
        bold_ratio = get_bold_ratio(element)
        if i < 100:
            print(f"  {i+1:3d}. {is_heading} Score: {score:2d} | Size: {avg_size:4.1f} | Bold: {bold_ratio:.0%} | {text_preview}...")
    
    print("=" * 70 + "\n")
    
    # ========== Step 2: 合并被拆分的标题 ==========
    merged_elements = []
    i = 0
    merged_titles = []

    while i < len(filtered_elements):
        current = filtered_elements[i]
        current_score = heading_scores[i]
        is_current_heading = current_score >= 4
        
        if is_current_heading and i + 1 < len(filtered_elements):
            next_elem = filtered_elements[i + 1]
            next_score = get_heading_score(next_elem, current, None, document_avg_size)
            is_next_heading = next_score >= 4
            
            if is_next_heading and current["page"] == next_elem["page"]:
                y_diff = abs(current["bbox"][1] - next_elem["bbox"][1])
                if y_diff < 30:
                    # ✅ 合并标题
                    current["text"] = current["text"].strip() + " " + next_elem["text"].strip()
                    print(f"🔗 Merged split title: '{current['text'][:60]}...'")
                    merged_titles.append(current["text"])  # ✅ 记录合并后的标题文本
                    i += 1
        
        merged_elements.append(current)
        i += 1

    filtered_elements = merged_elements

    # ✅ 重新计算 heading_scores
    heading_scores = []
    for i, element in enumerate(filtered_elements):
        previous_element = filtered_elements[i-1] if i > 0 else None
        next_element = filtered_elements[i+1] if i < len(filtered_elements)-1 else None
        score = get_heading_score(element, previous_element, next_element, document_avg_size)
        heading_scores.append(score)

    # ========== Step 3: Detect heading candidates ==========
    heading_candidates = []

    for i, element in enumerate(filtered_elements):
        if element.get("is_table", False):
            continue
        
        score = heading_scores[i]
        
        # ✅ 如果是合并后的标题，强制加入
        if element["text"] in merged_titles:
            print(f"✅ Force adding merged title: '{element['text'][:60]}...'")
            heading_candidates.append(element)
        elif score >= 4:
            heading_candidates.append(element)
    
    # ========== Step 4: Find document title ==========
    
    document_title = find_title_from_candidates(heading_candidates)
    print(f"📌 Document Title: {document_title}")
    
    # ========== Step 5: Create sections ==========
    temp_sections = []
    current_section = None
    
    for i, element in enumerate(filtered_elements):
        if element.get("is_table", False):
            if current_section:
                table_text = extract_table_content([element])
                if table_text:
                    current_section["content"] += table_text + "\n\n"
            continue
        
        score = heading_scores[i]
        is_heading = score >= 4
        
        if is_heading:
            text = element["text"]
            
            if current_section:
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
    
    if current_section:
        temp_sections.append(current_section)
    
    # ========== Step 6: 合并空 sections ==========
    sections = merge_empty_sections(temp_sections)
    
    # ============================================================
    # ✅ 显示最终 sections 预览
    # ============================================================
    print("\n" + "=" * 70)
    print("📊 FINAL SECTIONS:")
    print("=" * 70)
    
    for i, sec in enumerate(sections):
        heading = sec["heading"][:50]
        content = sec.get("content", "").strip()
        content_preview = content[:80].replace('\n', ' ') if content else "(EMPTY)"
        word_count = len(content.split()) if content else 0
        print(f"  {i+1:3d}. [{word_count:4d} words] {heading}...")
        print(f"         Content: {content_preview}...")
        print("-" * 50)
    
    print("=" * 70)
    print(f"✅ Total sections created: {len(sections)}")
    print("=" * 70 + "\n")
    
    return sections, document_title

def merge_empty_sections(sections):
    """
    如果 section 是空的（没有 content），把它的标题合并到下一个 section
    这样被拆分的标题（如 "The Impact..." 和 "Reflection..."）会被合并
    """
    if len(sections) < 2:
        return sections
    
    merged = []
    i = 0
    
    while i < len(sections):
        current = sections[i]
        content = current.get("content", "").strip()
        
        # ✅ 如果当前 section 有内容，直接保留
        if content:
            merged.append(current)
            i += 1
            continue
        
        # ✅ 当前 section 是空的（只有标题）
        empty_heading = current["heading"]
        current_page = current.get("page", 1)
        
        print(f"🔍 Found empty section: '{empty_heading[:40]}...'")
        
        # ✅ 找下一个有内容的 section
        found = False
        for j in range(i + 1, len(sections)):
            next_section = sections[j]
            next_content = next_section.get("content", "").strip()
            next_page = next_section.get("page", 1)
            
            # 如果有内容，把空标题合并到下一个 section
            if next_content:
                # ✅ 检查是否在同一页或下一页（标题被拆分通常在同一页）
                if next_page == current_page or next_page == current_page + 1:
                    # 合并标题
                    next_section["heading"] = empty_heading + " " + next_section["heading"]
                    # print(f"🔗 Merged empty heading: '{empty_heading[:30]}...' + '{next_section['heading'][:30]}...'")
                    
                    # ✅ 把合并后的 section 加入结果
                    merged.append(next_section)
                    found = True
                    i = j + 1  # ✅ 跳过被合并的 section
                    break
                else:
                    # 空标题和下一个内容在不同页 → 保留空标题
                    merged.append(current)
                    found = True
                    i += 1
                    break
        
        # 如果没找到下一个有内容的 section，保留空标题
        if not found:
            merged.append(current)
            i += 1
    
    return merged