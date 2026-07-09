from pdf_loader import extract_blocks
from chunker import create_sections


pdf_path = "../documents/sample.pdf"


elements = extract_blocks(pdf_path)


sections, title = create_sections(elements)


print("Total sections:", len(sections))


for section in sections:

    print("====================")
    print("HEADING:")
    print(section["heading"])
    print("SUBHEADING:")
    print(section["subheading"])
    print("CONTENT:")
    print(section["content"])