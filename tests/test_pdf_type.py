import pymupdf
from PIL import Image

from src.backend.app import detect_pdf_type


def test_empty_pdf_is_detected_as_empty(tmp_path):
    """
    An empty PDF should be classified as 'empty',
    not as 'scanned'.
    """

    pdf_path = tmp_path / "empty.pdf"

    # Create a PDF with one completely blank page
    doc = pymupdf.open()
    doc.new_page()
    doc.save(pdf_path)
    doc.close()

    result = detect_pdf_type(str(pdf_path))

    assert result == "empty"

def test_image_pdf_is_detected_as_scanned(tmp_path): 
    """ 
    A PDF containing an image but little/no text should be classified as 'scanned'. 
    """ 
    image_path = tmp_path / "scan.png" 
    pdf_path = tmp_path / "scanned.pdf" 

    # Create a simple image 
    image = Image.new("RGB", (500, 500), "white") 
    image.save(image_path) 

    # Put the image into a PDF page 
    doc = pymupdf.open() 
    page = doc.new_page() 
    page.insert_image( 
        pymupdf.Rect(50, 50, 550, 550), 
        filename=str(image_path), 
        ) 
        
    doc.save(pdf_path)
    doc.close() 
    result = detect_pdf_type(str(pdf_path)) 
    assert result == "scanned"

def test_text_pdf_is_detected_as_text(tmp_path): 
    """ 
    A PDF with enough extractable text should be classified as 'text'. 
    """ 
    pdf_path = tmp_path / "text.pdf" 
    doc = pymupdf.open() 
    page = doc.new_page() 
    page.insert_text( 
        (72, 72), 
        "This is a normal text based PDF document. " 
        "It contains enough text to pass the detection threshold." 
        ) 
    doc.save(pdf_path) 
    doc.close() 
    result = detect_pdf_type(str(pdf_path)) 
    assert result == "text"

def test_corrupt_pdf_is_detected(tmp_path):
    pdf_path = tmp_path / "corrupt.pdf"

    pdf_path.write_bytes(b"this is not a valid pdf")

    result = detect_pdf_type(str(pdf_path))

    assert result == "corrupt"