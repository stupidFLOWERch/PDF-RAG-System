from src.rag_ollama.chunker import (
    clean_html_tags,
    fallback_chunk_elements,
    find_title_from_candidates,
    flatten_sections,
    format_table_as_markdown,
    format_table_rows,
    is_table_heading,
    is_toc_line,
)


def test_clean_html_tags():
    text = "<table><tr><td>Hello</td></tr></table>"

    result = clean_html_tags(text)

    assert "<table>" not in result
    assert "<td>" not in result
    assert "<tr>" not in result
    assert "Hello" in result


def test_is_table_heading():
    assert is_table_heading("Table 1: Results") is True
    assert is_table_heading("Introduction") is False


def test_find_title_from_candidates():
    candidates = [
        {"text": "Email: test@example.com"},
        {"text": "My Research Project"},
    ]

    result = find_title_from_candidates(candidates)

    assert result == "My Research Project"

def test_is_toc_line():
    # Dotted leader + page number -> TOC
    assert is_toc_line("Introduction .......... 5") is True

    # Long text ending with a number -> TOC
    assert is_toc_line("Background and Methodology 10") is True

    # Normal numbered heading -> NOT TOC
    assert is_toc_line("5.2 Methodology") is False

    # Normal text -> NOT TOC
    assert is_toc_line("This is a normal paragraph.") is False


def test_format_table_as_markdown():
    table_lines = [
        "Table 1: Student Results",
        "Name Age",
        "Alice 20",
        "Bob 25",
    ]

    result = format_table_as_markdown(table_lines)

    # Table caption should be skipped
    assert "Table 1: Student Results" not in result

    # Markdown table should be created
    assert "| Name | Age |" in result
    assert "| Alice | 20 |" in result
    assert "| Bob | 25 |" in result

    # Markdown separator should exist
    assert "---" in result

    assert format_table_as_markdown([]) == ""

def test_format_table_rows():
    rows = [
        "Characteristics  Control  Experimental",
        "Age 20 25",
        "Height 170 175",
    ]

    result = format_table_rows("Table 1: Student Data", rows)

    # Table title
    assert "### Table 1: Student Data" in result

    # Header
    assert "Characteristics" in result
    assert "Control" in result
    assert "Experimental" in result

    # Data
    assert "Age" in result
    assert "20" in result
    assert "25" in result
    assert "Height" in result
    assert "170" in result
    assert "175" in result

    # Markdown format
    assert "---" in result

    assert format_table_rows("Table 1", []) == ""
    assert format_table_rows("", ["A  B"]) == ""


def test_flatten_sections_within_max_tokens():
    sections = [
        {
            "heading": "Introduction",
            "content": "This is a short introduction.",
            "page": 1,
        }
    ]

    documents = flatten_sections(
        sections,
        document_title="Test Document",
        max_tokens=256,
    )

    assert len(documents) == 1
    assert documents[0]["metadata"]["chunk_type"] == "full"

def test_flatten_sections_exceeds_max_tokens():
    sections = [
        {
            "heading": "Introduction",
            "content": "This is a sentence. " * 100,
            "page": 1,
        }
    ]

    documents = flatten_sections(
        sections,
        document_title="Test Document",
        max_tokens=20,
    )

    assert len(documents) > 1

    for i, document in enumerate(documents):
        assert document["metadata"]["chunk_type"] == "split"
        assert document["metadata"]["heading"] == "Introduction"
        assert document["metadata"]["chunk_index"] == i
        assert document["metadata"]["total_chunks"] == len(documents)
        assert "Introduction" in document["text"]

def test_fallback_chunk_elements_basic():

    elements = [
        {"text": "This is the first paragraph."},
        {"text": "This is the second paragraph."},
    ]

    documents = fallback_chunk_elements(
        elements,
        title="Test Document",
        max_tokens=256,
    )

    assert len(documents) > 0

    assert documents[0]["metadata"]["title"] == "Test Document"
    assert documents[0]["metadata"]["heading"] == ""
    assert documents[0]["metadata"]["chunk_type"] == "fallback"
    
def test_fallback_chunk_elements_table():

    elements = [
        {
            "page": 2,
            "is_table": True,
            "table_data": [
                ["Name", "Age"],
                ["Alice", "20"],
            ],
        }
    ]

    documents = fallback_chunk_elements(
        elements,
        title="Test Document",
        max_tokens=256,
    )

    assert len(documents) == 1
    assert documents[0]["metadata"]["chunk_type"] == "table"
    assert "| Name | Age |" in documents[0]["text"]
    assert documents[0]["metadata"]["page"] == 2