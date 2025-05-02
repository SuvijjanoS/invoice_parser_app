"""
Invoice Field Extractor

This application extracts fields from invoice documents (PDF/images) using OpenAI GPT-4 and agentic-doc.

Setup Instructions:
------------------
1. Environment Setup:
   - Install Python 3.8 or higher
   - Create and activate a virtual environment:
     ```
     python -m venv venv
     # On Windows:
     venv\\Scripts\\activate
     # On macOS/Linux:
     source venv/bin/activate
     ```

2. Install Required Packages:
   ```
   # Core dependencies
   pip install streamlit>=1.32.0 openai>=1.12.0
   
   # Document processing
   pip install agentic-doc>=0.0.21 PyMuPDF>=1.25.0
   
   # Image processing
   pip install Pillow>=10.0.0
   
   # Environment and utilities
   pip install python-dotenv>=1.0.0
   
   # Optional: for better PDF handling
   pip install pdf2image>=1.16.3
   ```

   Note: If using macOS and pdf2image, you'll need poppler:
   ```
   # macOS (using Homebrew)
   brew install poppler

   # Ubuntu/Debian
   sudo apt-get install poppler-utils
   ```

3. API Keys Setup:
   - Create a .env file in the same directory as this script
   - Add your API keys to the .env file:
     ```
     OPENAI_API_KEY=st.secrets.get("OPENAI_API_KEY")
     VISION_AGENT_API_KEY=yst.secrets.get("VISION_AGENT_API_KEY")

     if not OPENAI_API_KEY or not VISION_AGENT_API_KEY:
    st.error("API keys not found! Please add OPENAI_API_KEY and VISION_AGENT_API_KEY to the Streamlit secrets.")
    st.stop()

     ```
   - Get your OpenAI API key from: https://platform.openai.com/api-keys
   - Get your Vision Agent API key from the agentic-doc service

4. Running the Application:
   ```
   streamlit run document_parser.py
   ```

5. Using the Application:
   a. The web interface will open in your default browser
   b. Upload a document (PDF/PNG/JPG)
   c. Customize fields to extract:
      - Add new fields with name and description
      - Edit existing fields
      - Remove unwanted fields
      - Toggle which fields to extract
   d. Click "Extract Fields" to process the document
   e. View results with highlighted evidence in the document

Requirements:
------------
Core Dependencies:
- Python 3.8+
- streamlit>=1.32.0
- openai>=1.12.0
- agentic-doc>=0.0.21
- PyMuPDF>=1.25.0 (fitz)
- Pillow>=10.0.0 (PIL)
- python-dotenv>=1.0.0

Optional Dependencies:
- pdf2image>=1.16.3 (for better PDF handling)
- poppler-utils (system dependency for pdf2image)

Note: Make sure your OpenAI API key has access to GPT-4.
"""

import os
from pathlib import Path
from openai import OpenAI
from agentic_doc.parse import parse_documents
from agentic_doc.config import Settings
import streamlit as st
from typing import List, Dict, Any, Tuple
import json
from dotenv import load_dotenv
import tempfile
from PIL import Image, ImageDraw
import io
import fitz  # PyMuPDF for PDF handling
import re

# Load environment variables from .env file if it exists
load_dotenv()

import streamlit as st
from dotenv import load_dotenv
load_dotenv()  # preserves local .env for testing

# Fetch secrets first, then fallback to env vars
OPENAI_API_KEY = st.secrets.get("OPENAI_API_KEY") or os.getenv("OPENAI_API_KEY")
VISION_AGENT_API_KEY = st.secrets.get("VISION_AGENT_API_KEY") or os.getenv("VISION_AGENT_API_KEY")

if not OPENAI_API_KEY or not VISION_AGENT_API_KEY:
    st.error(
        "🔑 API keys missing!\n\n"
        "Please go to your Streamlit Cloud app’s Settings → Secrets, "
        "and add:\n"
        "  • OPENAI_API_KEY = <your OpenAI key>\n"
        "  • VISION_AGENT_API_KEY = <your Agentic Doc key>\n"
    )
    st.stop()

# Now initialize the clients with the keys
client = OpenAI(api_key=OPENAI_API_KEY)
settings = Settings(
    vision_agent_api_key=VISION_AGENT_API_KEY,
    batch_size=4,
    max_workers=5,
    max_retries=100,
    max_retry_wait_time=60,
    retry_logging_style="log_msg"
)

# Configure agentic-doc settings
settings = Settings(
    vision_agent_api_key=VISION_AGENT_API_KEY,
    batch_size=4,
    max_workers=5,
    max_retries=100,
    max_retry_wait_time=60,
    retry_logging_style="log_msg"
)

def setup_api_keys():
    """Set up API keys with fallback options."""
    # Try to get API key from Streamlit secrets first
    try:
        vision_key = st.secrets["VISION_AGENT_API_KEY"]
    except:
        # Fallback to environment variable
        vision_key = os.getenv("VISION_AGENT_API_KEY")
    
    if not vision_key:
        st.error("Vision Agent API Key not found in secrets or environment variables!")
        st.info("Please enter your Vision Agent API Key:")
        vision_key = st.text_input("Vision Agent API Key", type="password")
        if vision_key:
            os.environ["VISION_AGENT_API_KEY"] = vision_key
    
    return vision_key

def initialize_settings(api_key: str) -> Settings:
    """Initialize agentic-doc settings."""
    return Settings(
        vision_agent_api_key=api_key,
        batch_size=4,
        max_workers=5,
        max_retries=100,
        max_retry_wait_time=60,
        retry_logging_style="log_msg",
        pdf_to_image_dpi=96
    )

def get_default_fields() -> List[Dict[str, str]]:
    """Return the default invoice fields with descriptions."""
    return [
        {
            "name": "Document No.",
            "description": "A unique document number, typically starting with numbers (e.g., 27xxxxxx)"
        },
        {
            "name": "D1",
            "description": "The D1 field value from the invoice"
        },
        {
            "name": "Comcode",
            "description": "The company code or identification number"
        },
        {
            "name": "Document Type",
            "description": "The type of document (e.g., Invoice, Receipt, etc.)"
        },
        {
            "name": "Year",
            "description": "The year the document was issued"
        },
        {
            "name": "Receiving Company Name",
            "description": "The name of the company receiving the invoice"
        },
        {
            "name": "Receiving Company Address",
            "description": "The complete address of the receiving company"
        },
        {
            "name": "Receiving Company Tax ID",
            "description": "The tax identification number of the receiving company"
        },
        {
            "name": "Date",
            "description": "The date the document was issued"
        },
        {
            "name": "Issuing Company Name",
            "description": "The name of the company issuing the invoice"
        }
    ]

def manage_fields():
    """Manage extraction fields with add, edit, and remove functionality."""
    st.subheader("Manage Fields")
    
    # Initialize session state for fields if not exists
    if 'extraction_fields' not in st.session_state:
        st.session_state.extraction_fields = get_default_fields()
    
    # Add new field
    st.markdown("### Add New Field")
    col1, col2, col3 = st.columns([2, 3, 1])
    with col1:
        new_field_name = st.text_input("Field Name", key="new_field_name")
    with col2:
        new_field_desc = st.text_input("Field Description", key="new_field_desc")
    with col3:
        if st.button("Add Field"):
            if new_field_name and new_field_desc:
                st.session_state.extraction_fields.append({
                    "name": new_field_name,
                    "description": new_field_desc
                })
                # Clear input fields
                st.session_state.new_field_name = ""
                st.session_state.new_field_desc = ""
                st.rerun()
    
    # Edit existing fields
    st.markdown("### Current Fields")
    for idx, field in enumerate(st.session_state.extraction_fields):
        col1, col2, col3, col4 = st.columns([2, 3, 1, 1])
        
        with col1:
            field['name'] = st.text_input(
                "Name",
                value=field['name'],
                key=f"name_{idx}"
            )
        with col2:
            field['description'] = st.text_input(
                "Description",
                value=field['description'],
                key=f"desc_{idx}"
            )
        with col3:
            if st.button("Remove", key=f"remove_{idx}"):
                st.session_state.extraction_fields.pop(idx)
                st.rerun()
        with col4:
            st.checkbox("Extract", value=True, key=f"extract_{idx}")
    
    # Reset to defaults button
    if st.button("Reset to Default Fields"):
        st.session_state.extraction_fields = get_default_fields()
        st.rerun()
    
    # Return selected fields
    return [
        field for idx, field in enumerate(st.session_state.extraction_fields)
        if st.session_state.get(f"extract_{idx}", True)
    ]

def extract_fields_with_openai(text: str, fields: List[Dict[str, str]], chunks: List[Any] = None) -> Dict[str, Any]:
    """Use OpenAI to extract specific fields from text and track source chunks."""
    field_instructions = "\n".join([
        f"- {field['name']}: {field['description']}"
        for field in fields
    ])
    
    prompt = f"""
    Extract the following fields from the invoice text below.
    For each field, use the description to identify the correct information.
    
    Fields to extract:
    {field_instructions}
    
    Text:
    {text}
    
    Return the results in JSON format with the following structure for each field:
    {{
        "field_name": {{
            "value": "extracted value"
        }}
    }}
    
    If a field is not found, set its value to null.
    Ensure values match the expected format described in the field descriptions.
    """
    
    response = client.chat.completions.create(
        model="gpt-4",
        messages=[
            {
                "role": "system", 
                "content": "You are a specialized invoice parser that accurately extracts fields."
            },
            {"role": "user", "content": prompt}
        ],
        temperature=0.1
    )
    
    try:
        extracted_data = json.loads(response.choices[0].message.content)
        
        # If chunks are provided, find matching chunks for each extracted field
        if chunks:
            for field_name, field_data in extracted_data.items():
                if field_data.get('value'):
                    # Find chunks that contain the value
                    matching_chunks = []
                    value = field_data['value'].strip().lower()
                    for chunk in chunks:
                        if value in chunk.text.lower():
                            matching_chunks.append(chunk)
                    field_data['matching_chunks'] = matching_chunks
        
        return extracted_data
    except json.JSONDecodeError:
        return {field['name']: {'value': None} for field in fields}

def convert_pdf_page_to_image(pdf_path: str, page_number: int) -> Image.Image:
    """Convert a PDF page to a PIL Image."""
    try:
        pdf_document = fitz.open(pdf_path)
        page = pdf_document[page_number]
        # Get the page's pixmap (rasterize the page)
        pix = page.get_pixmap(matrix=fitz.Matrix(2, 2))  # 2x scaling for better quality
        # Convert pixmap to PIL Image
        img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
        pdf_document.close()
        return img
    except Exception as e:
        st.error(f"Error converting PDF page to image: {str(e)}")
        return None

def get_document_image(doc_path: str, page_idx: int) -> Image.Image:
    """Get image for a specific page, handling both PDFs and images."""
    try:
        if doc_path.lower().endswith('.pdf'):
            return convert_pdf_page_to_image(doc_path, page_idx)
        else:
            return Image.open(doc_path)
    except Exception as e:
        st.error(f"Error loading document image: {str(e)}")
        return None

def parse_box_string(box_str: str):
    """Parse a box string like 'l=0.11625 t=0.3486931818181818 r=0.2275 b=0.3631818181818182' into [left, top, right, bottom]."""
    st.write("Attempting to parse box string:", box_str)
    
    if not isinstance(box_str, str):
        st.error("Box string is not a string type")
        return None
        
    try:
        # Split the string and create a dictionary of coordinates
        parts = box_str.split()
        st.write("Split parts:", parts)
        
        coords = {}
        for part in parts:
            st.write("Processing part:", part)
            key, value = part.split('=')
            coords[key] = float(value)
        
        st.write("Parsed coordinates:", coords)
        
        # Check if we have all required coordinates
        if all(k in coords for k in ['l', 't', 'r', 'b']):
            result = [coords['l'], coords['t'], coords['r'], coords['b']]
            st.write("Final coordinates:", result)
            return result
            
        st.error("Missing required coordinates")
        return None
        
    except Exception as e:
        st.error(f"Error parsing box string: {str(e)}")
        import traceback
        st.write("Traceback:", traceback.format_exc())
        return None

def draw_bounding_box(image: Image.Image, box_coords: List[float], color: Tuple[int, int, int] = (255, 0, 0), thickness: int = 2) -> Image.Image:
    """Draw a bounding box on an image using normalized coordinates."""
    try:
        # Create a copy of the image to avoid modifying the original
        img_copy = image.copy()
        draw = ImageDraw.Draw(img_copy)
        
        # Get image dimensions
        width, height = img_copy.size
        
        # Convert normalized coordinates to pixel coordinates
        x0, y0, x1, y1 = box_coords
        pixel_coords = [
            int(x0 * width),   # left
            int(y0 * height),  # top
            int(x1 * width),   # right
            int(y1 * height)   # bottom
        ]
        
        # Draw rectangle with specified thickness
        for i in range(thickness):
            draw.rectangle([
                pixel_coords[0] - i,
                pixel_coords[1] - i,
                pixel_coords[2] + i,
                pixel_coords[3] + i
            ], outline=color)
        
        return img_copy
    except Exception as e:
        st.error(f"Error drawing bounding box: {str(e)}")
        return image

def get_box_coordinates(box):
    """Extract coordinates from various box formats."""
    if isinstance(box, (list, tuple)) and len(box) == 4:
        return box
    elif isinstance(box, str):
        return parse_box_string(box)
    else:
        # Handle ChunkGroundingBox object
        try:
            return [
                getattr(box, 'l', 0),
                getattr(box, 't', 0),
                getattr(box, 'r', 1),
                getattr(box, 'b', 1)
            ]
        except Exception as e:
            st.error(f"Error extracting coordinates from box object: {str(e)}")
            return None

def display_chunk_evidence(chunk, field_name: str, doc_path: str):
    """Display a chunk with its evidence for a field extraction."""
    if hasattr(chunk, 'grounding') and chunk.grounding:
        for g in chunk.grounding:
            page_idx = getattr(g, 'page_idx', 0)
            box = getattr(g, 'box', None)
            
            # Try to get box coordinates
            box_coords = get_box_coordinates(box)
            if box_coords:
                try:
                    # Load and display the page image
                    page_image = get_document_image(doc_path, page_idx)
                    if page_image:
                        # Draw bounding box
                        annotated_image = draw_bounding_box(
                            page_image,
                            box_coords,
                            color=(255, 0, 0),  # Red
                            thickness=3
                        )
                        st.image(annotated_image)
                except Exception as e:
                    st.error(f"Error processing image: {str(e)}")

def translate_markdown(markdown_text: str) -> str:
    """Translate markdown content from Thai to English while preserving markdown formatting."""
    prompt = """
    Translate the following markdown text from Thai to English. 
    Preserve all markdown formatting, including:
    - Headers (#, ##, etc.)
    - Lists (* or -, numbers)
    - Tables
    - Links
    - Code blocks
    - Bold and italic formatting
    
    Original markdown:
    {text}
    
    Provide only the translated markdown without any explanations.
    """
    
    response = client.chat.completions.create(
        model="gpt-4",
        messages=[
            {"role": "system", "content": "You are a professional translator that preserves markdown formatting while translating content."},
            {"role": "user", "content": prompt.format(text=markdown_text)}
        ],
        temperature=0.1
    )
    
    return response.choices[0].message.content

def get_box_area(box_coords):
    """Calculate the area of a bounding box using normalized coordinates."""
    if not box_coords or len(box_coords) != 4:
        return float('inf')
    width = abs(box_coords[2] - box_coords[0])  # |right - left|
    height = abs(box_coords[3] - box_coords[1])  # |bottom - top|
    return width * height

def get_smallest_relevant_chunk(chunks, extracted_value):
    """Get the smallest chunk that contains the exact extracted value."""
    if not extracted_value:
        return None
        
    # Normalize the extracted value for comparison
    extracted_value = extracted_value.strip().lower()
    
    smallest_chunk = None
    smallest_area = float('inf')
    
    for chunk in chunks:
        # Check if this chunk's text contains the exact extracted value
        if not hasattr(chunk, 'text') or extracted_value not in chunk.text.lower():
            continue
            
        if hasattr(chunk, 'grounding') and chunk.grounding:
            for g in chunk.grounding:
                box = getattr(g, 'box', None)
                if box:
                    box_coords = get_box_coordinates(box)
                    if box_coords:
                        area = get_box_area(box_coords)
                        # Update if we find a smaller area that contains our value
                        if area > 0 and area < smallest_area:
                            smallest_area = area
                            smallest_chunk = chunk
    
    return smallest_chunk

def main():
    st.title("📄 Invoice Field Extractor")
    st.write("Upload an invoice document to extract specific fields")
    
    # Display API key status
    if not VISION_AGENT_API_KEY:
        st.error("Vision Agent API Key is not set!")
        st.stop()
    else:
        st.success(f"System Ready")
    
    # File upload
    uploaded_file = st.file_uploader("Upload document", type=["pdf", "png", "jpg", "jpeg"])
    
    # Manage fields
    selected_fields = manage_fields()
    
    # Translation option
    translate_to_english = st.checkbox("Translate content from Thai to English", value=False)
    
    if uploaded_file and selected_fields and st.button("Extract Fields"):
        with st.spinner("Processing document..."):
            # Create a temporary directory for processing
            with tempfile.TemporaryDirectory() as temp_dir:
                # Save uploaded file temporarily
                temp_path = Path(temp_dir) / f"input_doc{Path(uploaded_file.name).suffix}"
                with open(temp_path, "wb") as f:
                    f.write(uploaded_file.getvalue())
                
                try:
                    # Parse document with agentic-doc
                    results = parse_documents([str(temp_path)])
                    if not results:
                        st.error("No results returned from parse_documents")
                        return
                    
                    parsed_doc = results[0]
                    
                    # Combine all text from chunks for field extraction
                    full_text = "\n".join(chunk.text for chunk in parsed_doc.chunks)
                    if translate_to_english:
                        with st.spinner("Translating text for field extraction..."):
                            full_text = translate_markdown(full_text)
                    
                    # Extract fields using OpenAI with chunks
                    extracted_data = extract_fields_with_openai(full_text, selected_fields, parsed_doc.chunks)
                    
                    # Display results with evidence
                    for field_name, field_data in extracted_data.items():
                        if field_data.get('value'):
                            col1, col2 = st.columns([1, 3])
                            with col1:
                                st.markdown(f"**{field_name}:**")
                                st.write(field_data['value'])
                            with col2:
                                if field_data.get('matching_chunks'):
                                    # Get the smallest chunk containing the exact value
                                    smallest_chunk = get_smallest_relevant_chunk(
                                        field_data['matching_chunks'],
                                        field_data['value']
                                    )
                                    if smallest_chunk:
                                        display_chunk_evidence(smallest_chunk, field_name, str(temp_path))
                
                except Exception as e:
                    st.error(f"Error processing document: {str(e)}")

if __name__ == "__main__":
    main() 
