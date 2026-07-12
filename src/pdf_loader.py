import fitz
import re


def extract_lines(pdf_path):

    doc = fitz.open(pdf_path)

    elements = []


    for page_num, page in enumerate(doc):

        data = page.get_text("dict")


        for block_id, block in enumerate(data["blocks"]):

            # skip image block
            if "lines" not in block:
                continue


            for line in block["lines"]:

                text = ""
                fonts = []
                sizes = []


                for span in line["spans"]:

                    text += span["text"]
                    fonts.append(span["font"])
                    sizes.append(span["size"])


                if text.strip():

                    elements.append({
                        "page": page_num + 1,
                        "block": block_id,
                        "text": text.strip(),
                        "font": fonts,
                        "size": sizes,
                        "bbox": line["bbox"]
                    })


    return elements



def merge_lines(elements):

    merged = []

    for element in elements:

        if not merged:
            merged.append(element)
            continue


        previous = merged[-1]


        same_block = (
            element["page"] == previous["page"]
            and element["block"] == previous["block"]
        )


        same_font = (
            element["font"] == previous["font"]
        )


        same_size = (
            element["size"] == previous["size"]
        )


        if same_block and same_font and same_size:

            # print("===== MERGING =====")
            # print("BEFORE:")
            # print(previous["text"])

            # print("WITH:")
            # print(element["text"])


            previous["text"] += " " + element["text"]


            # print("AFTER:")
            # print(previous["text"])
            # print("===================")


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



def get_heading_score(element, document_avg_size):

    score = 0


    text = element["text"].strip()


    avg_size = (
        sum(element["size"])
        /
        len(element["size"])
    )


    # Rule 1: Bold
    is_bold = any(
        "Bold" in font
        for font in element["font"]
    )

    if is_bold:
        score += 2



    # Rule 2: Larger font than normal text
    if avg_size > document_avg_size:
        score += 2



    # Rule 3: Short text
    word_count = len(text.split())

    if word_count <= 12:
        score += 1



    # Rule 4: Not normal sentence
    if not text.endswith("."):
        score += 1



    # Rule 5: Numbered heading
    if re.match(
        r"^(Chapter|Part|\d+\.|\d+\.)",
        text,
        re.I
    ):
        score += 2



    return score