import fitz

def extract_blocks(pdf_path):

    doc = fitz.open(pdf_path)

    elements = []


    for page_num, page in enumerate(doc):

        data = page.get_text("dict")

        for block in data["blocks"]:

            if "lines" not in block:
                continue


            for line in block["lines"]:

                text = ""
                fonts = []


                for span in line["spans"]:

                    text += span["text"]
                    fonts.append(span["font"])


                if text.strip():

                    elements.append({
                        "page": page_num + 1,
                        "text": text.strip(),
                        "font": fonts
                    })


    return elements



def is_heading(element):

    text = element["text"].strip()

    is_bold = any(
        "Bold" in font
        for font in element["font"]
    )

    return is_bold and not text.endswith(".")
