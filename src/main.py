from pdf_loader import extract_lines, merge_lines
from chunker import create_sections, flatten_sections


pdf_path = "../documents/plant-hunt-info.pdf"


elements = extract_lines(pdf_path)

elements = merge_lines(elements)

sections, title = create_sections(elements)

documents = flatten_sections(
    sections,
    title
)

print("Total sections:", len(documents))


for doc in documents:

    print("====================")

    print("HEADING:")
    print(doc["metadata"]["heading"])

    print("SUBHEADING:")
    print(doc["metadata"]["subheading"])

    print("PAGE:")
    print(doc["metadata"]["page"])

    print("TEXT:")
    print(doc["text"])