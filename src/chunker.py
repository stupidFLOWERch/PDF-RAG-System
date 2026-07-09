from langchain_text_splitters import RecursiveCharacterTextSplitter
from pdf_loader import is_heading

def create_sections(elements):

    sections = []
    current_section = None
    is_previous_header = False
    first_heading = True

    for element in elements:
        
        if is_heading(element):
            if first_heading:

                document_title = element["text"]
                first_heading = False
                continue

            if is_previous_header and current_section:
                current_section["subheading"] = element["text"]

                previous_was_heading = True
                continue

            # 如果已经有 section，先保存
            if current_section:
                sections.append(current_section)


            # 创建新的 section
            current_section = {
                "heading": element["text"],
                "subheading": None,
                "page": element["page"],
                "content": ""
            }

            is_previous_header = True


        else:
            is_previous_header = False

            if current_section:
                current_section["content"] += (
                    element["text"] + " "
                )
        


    # 最后一个 section
    if current_section:
        sections.append(current_section)


    return sections, document_title

def create_chunks(sections, document_title):

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=700,
        chunk_overlap=200
    )

    chunks = []

    for page in pages:

        page_chunks = splitter.split_text(
            page["text"]
        )

        for chunk in page_chunks:

            chunks.append({
                "page": page["page"],
                "text": chunk
            })

    return chunks