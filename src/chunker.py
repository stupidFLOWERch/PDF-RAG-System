import re
import tiktoken
from pdf_loader import (
    get_heading_score,
    calculate_document_avg_size
)


def get_token_count(text):
    """计算 token 数"""
    enc = tiktoken.encoding_for_model("gpt-3.5-turbo")
    return len(enc.encode(text))


def flatten_sections(sections, document_title, max_tokens=512):
    """
    展平 sections，先按 header 分 chunk，超长再按 token 切分
    """
    documents = []
    
    for section in sections:
        heading = section["heading"]
        content = section.get("content", "").strip()
        page = section.get("page", 1)
        
        # 如果内容为空，跳过（不保存空标题）
        if not content:
            continue
        
        full_text = heading + "\n\n" + content
        
        # 检查 token 数
        token_count = get_token_count(full_text)
        
        if token_count <= max_tokens:
            # 不需要切分
            documents.append({
                "text": full_text,
                "metadata": {
                    "title": document_title,
                    "heading": heading,
                    "page": page,
                    "chunk_type": "full"
                }
            })
        else:
            # 需要切分
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
    按标题切分文本，保持每个 chunk 包含标题
    """
    # 提取内容（移除开头的标题）
    content = full_text.replace(heading, "", 1).strip()
    
    # 如果内容本身很短，不需要切分
    if get_token_count(content) <= max_tokens:
        return [full_text]
    
    # 按句子切分（. ! ? 后面跟空格）
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
    """用规则找出 Document Title"""
    for i, h in enumerate(heading_candidates):
        text = h["text"].strip()
        
        if i > 3:
            break
        
        word_count = len(text.split())
        if word_count < 3 or word_count > 20:
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
    """检测是否是 TOC 页面"""
    page_elements = [e for e in elements if e["page"] == page_num]
    text = " ".join([e["text"] for e in page_elements])
    
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
    """检测是否是 TOC 行"""
    text = text.strip()
    
    if re.search(r'\.{5,}\s*\d+', text):
        return True
    
    if re.search(r'\d+$', text) and len(text.split()) > 3:
        return True
    
    if re.search(r'Part\s+\w+\s+\.{5,}\s*\d+', text, re.I):
        return True
    
    return False


def is_table_heading(text):
    """检测是否是表格标题"""
    return bool(re.match(r'^Table\s+\d+[:.]', text.strip(), re.I))


def format_table_as_markdown(table_lines):
    """把表格转换为 Markdown"""
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
    """从 elements 中提取表格内容"""
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
    创建 sections，跳过 TOC，处理表格
    """
    sections = []
    current_section = None
    document_title = None
    
    document_avg_size = calculate_document_avg_size(elements)
    
    # ========== 第一步：过滤 TOC ==========
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
    
    # ========== 第二步：找标题 ==========
    heading_candidates = []
    
    for element in filtered_elements:
        if element.get("is_table", False):
            continue
        
        if get_heading_score(element, None, document_avg_size) >= 4:
            heading_candidates.append(element)
    
    # ========== 第三步：找 Title ==========
    document_title = find_title_from_candidates(heading_candidates)
    print(f"📌 Document Title: {document_title}")
    
    # ========== 第四步：主循环 ==========
    for element in filtered_elements:
        # 表格内容特殊处理
        if element.get("is_table", False):
            if current_section:
                table_text = extract_table_content([element])
                if table_text:
                    current_section["content"] += table_text + "\n\n"
            continue
        
        score = get_heading_score(element, None, document_avg_size)
        is_heading = score >= 4
        
        if is_heading:
            text = element["text"]
            
            if current_section:
                sections.append(current_section)
            
            current_section = {
                "heading": text,
                "page": element["page"],
                "content": ""
            }
            print(f"📂 Created section: {text[:50]}...")
        
        else:
            if current_section:
                current_section["content"] += element["text"] + " "
    
    if current_section:
        sections.append(current_section)
    
    print(f"✅ Total sections created: {len(sections)}")
    return sections, document_title