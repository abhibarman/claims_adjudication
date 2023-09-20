import streamlit as st
import fitz  # PyMuPDF
from PIL import Image
import io

# Set the title of the Streamlit app
st.title("PDF Viewer")

# Upload a PDF file
pdf_file = st.file_uploader("Upload a PDF file", type=["pdf"])

# Initialize page number
page_number = st.sidebar.number_input("Page Number", min_value=1, max_value=1, step=1)

# Check if a PDF file was uploaded
if pdf_file is not None:
    # Open the PDF file using PyMuPDF
    pdf_document = fitz.open(stream=pdf_file.read(), filetype="pdf")

    # Display the selected page
    if page_number > 0 and page_number <= len(pdf_document):
        page = pdf_document.load_page(page_number - 1)
        image = page.get_pixmap()

        # Convert the Pixmap to an image (PIL)
        pil_image = Image.frombytes("RGB", [image.width, image.height], image.samples)

        # Display the image using Streamlit
        st.image(pil_image, caption=f"Page {page_number}", use_column_width=True)

# Navigation controls at the bottom
st.write("Navigation:")
prev_page, next_page = st.beta_columns(2)
if prev_page.button("← Previous Page", key="prev_page") and page_number > 1:
    page_number -= 1
if next_page.button("Next Page →", key="next_page") and page_number < len(pdf_document):
    page_number += 1
