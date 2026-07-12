import re

from pdf_loader import (
    get_heading_score,
    calculate_document_avg_size
)


def classify_heading_level(element, largest_heading_size):

    avg_size = (
        sum(element["size"])
        /
        len(element["size"])
    )

    if avg_size >= largest_heading_size:
        return 1

    else:
        return 2

def flatten_sections(sections, document_title):

    documents = []


    for section in sections:

        heading = section["heading"]


        # 没有subsection
        if not section["subsections"]:


            if section.get("content"):

                text = (
                    heading
                    + "\n\n"
                    + section["content"]
                )


                documents.append({

                    "text": text,

                    "metadata": {

                        "title": document_title,

                        "heading": heading,

                        "subheading": None,

                        "page": section["page"]

                    }

                })



        # 有subsection
        else:


            for subsection in section["subsections"]:


                subheading = subsection["subheading"]

                content = subsection["content"]


                text = (
                    heading
                    + "\n\n"
                    + subheading
                    + "\n\n"
                    + content
                )


                documents.append({

                    "text": text,

                    "metadata": {

                        "title": document_title,

                        "heading": heading,

                        "subheading": subheading,

                        "page": section["page"]

                    }

                })


    return documents

def create_sections(elements):

    sections = []

    current_section = None
    current_subsection = None

    document_title = None

    document_avg_size = calculate_document_avg_size(elements)

    heading_candidates = []

    for element in elements:

        score = get_heading_score(
            element,
            document_avg_size
        )

        if score >= 4:
            is_heading = True
            heading_candidates.append(element)
    
    largest_heading_size = max(
        sum(e["size"]) / len(e["size"])
        for e in heading_candidates[1:]
        )
                    
    for element in elements:


        score = get_heading_score(
            element,
            document_avg_size
        )


        is_heading = score >= 4



        print(
            score,
            element["text"]
        )



        if is_heading:


            text = element["text"]


            if document_title is None:

                document_title = text
                # continue



            level = classify_heading_level(
                element,
                largest_heading_size
            )
            # print(
            #     "LEVEL:",
            #     level,
            #     element["text"]
            # )


            if level == 1:


                if current_section:

                    sections.append(
                        current_section
                    )


                current_section = {

                    "heading": text,

                    "page": element["page"],

                    "content": "",

                    "subsections": []

                }


                current_subsection = None



            else:


                if current_section is None:
                    continue


                current_subsection = {

                    "subheading": text,

                    "content": ""

                }


                current_section["subsections"].append(
                    current_subsection
                )



        else:


            if current_section is None:
                continue


            if current_subsection:


                current_subsection["content"] += (
                    element["text"] + " "
                )


            else:

                current_section["content"] += (
                    element["text"] + " "
                )



    if current_section:

        sections.append(
            current_section
        )


    return sections, document_title