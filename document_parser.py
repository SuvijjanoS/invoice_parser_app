import os
from pathlib import Path
import json
import tempfile
import re
import random
import io
import base64
import glob

import streamlit as st
from dotenv import load_dotenv
from openai import OpenAI
import fitz  # PyMuPDF for PDF handling
from typing import List, Dict, Any, Tuple, Optional, Union
from PIL import Image, ImageDraw, ImageFont
from collections import defaultdict
import pandas as pd

# Handle different versions of OpenAI package
try:
    from openai.error import InvalidRequestError, OpenAIError
except ImportError:
    # Define fallback error classes if needed
    class InvalidRequestError(Exception): pass
    class OpenAIError(Exception): pass

# Conditionally import agentic_doc if available
try:
    from agentic_doc.parse import parse_documents
    from agentic_doc.config import Settings
    agentic_imported = True
except ImportError:
    agentic_imported = False
    st.error("⚠️ agentic_doc package not installed. Please install it with `pip install agentic_doc`")

# Load local .env for development; secrets will override in cloud
load_dotenv()

# Fetch API keys: prefer Streamlit secrets, fallback to environment
OPENAI_API_KEY = st.secrets.get("OPENAI_API_KEY", os.getenv("OPENAI_API_KEY"))
VISION_AGENT_API_KEY = st.secrets.get("VISION_AGENT_API_KEY", os.getenv("VISION_AGENT_API_KEY"))

# Define a set of distinct colors for bounding boxes
COLORS = [
    (255, 0, 0),      # Red
    (0, 255, 0),      # Green
    (0, 0, 255),      # Blue
    (255, 255, 0),    # Yellow
    (255, 0, 255),    # Magenta
    (0, 255, 255),    # Cyan
    (255, 128, 0),    # Orange
    (128, 0, 255),    # Purple
    (0, 128, 0),      # Dark Green
    (0, 0, 128),      # Navy Blue
    (128, 128, 0),    # Olive
    (128, 0, 0),      # Maroon
    (0, 128, 128),    # Teal
    (255, 128, 128),  # Light Red
    (128, 255, 128),  # Light Green
]

def initialize_clients():
    if not OPENAI_API_KEY:
        st.error(
            "🔑 OpenAI API key missing!\n\n"
            "Please go to your Streamlit Cloud app's Settings → Secrets, and add:\n"
            "  • OPENAI_API_KEY = <your OpenAI key>\n"
        )
        return None
    
    client = OpenAI(api_key=OPENAI_API_KEY)
    
    if agentic_imported and not VISION_AGENT_API_KEY:
        st.error(
            "🔑 Vision Agent API key missing!\n\n"
            "Please go to your Streamlit Cloud app's Settings → Secrets, and add:\n"
            "  • VISION_AGENT_API_KEY = <your Agentic Doc key>\n"
        )
        return None
    
    if agentic_imported:
        try:
            # Initialize with the correct parameters according to the agentic-doc documentation
            settings = Settings(
                vision_agent_api_key=VISION_AGENT_API_KEY,
                batch_size=4,
                max_workers=5,
                max_retries=100,
                max_retry_wait_time=60,
                retry_logging_style="log_msg"
            )
        except Exception as e:
            st.error(f"Error initializing Settings: {e}")
            return None
    
    return client

# Helper functions

def get_default_fields() -> List[Dict[str, str]]:
    return [
        {"name": "Document Type", "description": "The type of document (e.g., Invoice, Receipt, etc.)"},
        {"name": "Document Date", "description": "The date [dd-month-year the document was issued"},
        {"name": "Receiving Company Name", "description": "The name of the company receiving the invoice"},
        {"name": "Receiving Company Address", "description": "The complete address of the receiving company"},
        {"name": "Receiving Company Tax ID", "description": "The tax identification number of the receiving company"},
        {"name": "Issuing Company Name", "description": "The name of the company/vendor/supplier issuing the invoice"},
        {"name": "Amount", "description": "The Grand Total amount of the invoice."},
        {"name": "VAT", "description": "The value-added-tax on the invoice."}
    ]


def manage_fields():
    st.subheader("Manage Fields")
    if 'extraction_fields' not in st.session_state:
        st.session_state.extraction_fields = get_default_fields()

    st.markdown("### Add New Field")
    col1, col2, col3 = st.columns([2, 3, 1])
    with col1:
        new_name = st.text_input("Field Name", key="new_name")
    with col2:
        new_desc = st.text_input("Field Description", key="new_desc")
    with col3:
        if st.button("Add Field") and new_name and new_desc:
            st.session_state.extraction_fields.append({"name": new_name, "description": new_desc})
            st.session_state.new_name = ""
            st.session_state.new_desc = ""
            st.rerun()

    st.markdown("### Current Fields")
    for idx, fld in enumerate(st.session_state.extraction_fields):
        c1, c2, c3, c4 = st.columns([2, 3, 1, 1])
        with c1:
            fld['name'] = st.text_input("Name", value=fld['name'], key=f"name_{idx}")
        with c2:
            fld['description'] = st.text_input("Description", value=fld['description'], key=f"desc_{idx}")
        with c3:
            if st.button("Remove", key=f"rm_{idx}"):
                st.session_state.extraction_fields.pop(idx)
                st.rerun()
        with c4:
            st.checkbox("Extract", value=True, key=f"ext_{idx}")

    if st.button("Reset to Default Fields"):
        st.session_state.extraction_fields = get_default_fields()
        st.rerun()

    return [fld for i, fld in enumerate(st.session_state.extraction_fields) if st.session_state.get(f"ext_{i}", True)]


def extract_fields_with_openai(client, text: str, fields: List[Dict[str, str]], chunks: List[Any] = None) -> Dict[str, Any]:
    """Use OpenAI to extract specific fields from text and track source chunks."""
    field_instructions = "\n".join([
        f"- {field['name']}: {field['description']}"
        for field in fields
    ])
    prompt = f"""
Extract the following fields from the invoice text below.

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
    try:
        response = client.chat.completions.create(
            model="gpt-4",
            messages=[
                {"role": "system", "content": "You are a specialized invoice parser that accurately extracts fields."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.1
        )
    except Exception as e:
        # Handle different error structures
        status = getattr(e, 'status_code', None) or getattr(e, 'http_status', None)
        code = getattr(e, 'code', None)
        
        if status == 402 or code == 'insufficient_quota':
            st.error("🚫 You've run out of API credits. Please add more credits to continue.")
            st.stop()
        else:
            st.error(f"OpenAI API error: {e}")
            st.stop()

    try:
        extracted_data = json.loads(response.choices[0].message.content)
        if chunks:
            for field_name, field_data in extracted_data.items():
                if field_data.get('value'):
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
    try:
        doc = fitz.open(pdf_path)
        pix = doc[page_number].get_pixmap(matrix=fitz.Matrix(2,2))
        img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
        doc.close()
        return img
    except Exception as e:
        st.error(f"Error converting PDF: {e}")
        return None


def get_document_image(path: str, idx: int) -> Image.Image:
    return convert_pdf_page_to_image(path, idx) if path.lower().endswith('.pdf') else Image.open(path)


def parse_box_string(s: str):
    try:
        parts = s.split()
        coords = {k: float(v) for part in parts for k, v in [part.split('=')]}
        return [coords[k] for k in ('l', 't', 'r', 'b')]
    except Exception as e:
        st.error(f"Error parsing box string: {s}, {e}")
        return None


def draw_bounding_box(img: Image.Image, box: List[float], color: Tuple[int, int, int] = (255, 0, 0), 
                     label: Optional[str] = None) -> Image.Image:
    """Draw a bounding box on an image with optional label."""
    out = img.copy()
    d = ImageDraw.Draw(out)
    w, h = out.size
    x0, y0, x1, y1 = [int(c*dim) for c, dim in zip(box, [w, h, w, h])]
    d.rectangle([x0, y0, x1, y1], outline=color, width=2)
    
    # Add label if provided
    if label:
        try:
            # Try to use a default font, but gracefully fail if not available
            font = ImageFont.truetype("arial.ttf", 14)
        except IOError:
            font = ImageFont.load_default()
        
        # Draw text background for better visibility
        text_width, text_height = d.textsize(label, font=font) if hasattr(d, 'textsize') else (len(label) * 7, 16)
        d.rectangle([x0, y0-text_height-4, x0+text_width+4, y0], fill=color)
        d.text((x0+2, y0-text_height-2), label, fill=(255, 255, 255), font=font)
    
    return out


def get_chunk_coordinates(chunk, path: str) -> List[Dict]:
    """Get coordinates for a chunk's bounding boxes."""
    coords_list = []
    
    if hasattr(chunk, 'grounding'):
        for g in chunk.grounding:
            box = getattr(g, 'box', None)
            page_idx = getattr(g, 'page_idx', 0)
            
            coords = parse_box_string(box) if isinstance(box, str) else (
                getattr(box, 'l', None) is not None and [box.l, box.t, box.r, box.b]
            )
            
            if coords:
                coords_list.append({
                    'coords': coords,
                    'page_idx': page_idx
                })
    
    return coords_list


def display_chunk_evidence(chunk, name: str, path: str, color: Tuple[int, int, int] = (255, 0, 0)):
    """Display a single chunk with bounding box."""
    if hasattr(chunk, 'grounding'):
        for g in chunk.grounding:
            box = getattr(g, 'box', None)
            coords = parse_box_string(box) if isinstance(box, str) else (
                getattr(box, 'l', None) is not None and [box.l, box.t, box.r, box.b]
            )
            if coords:
                img = get_document_image(path, getattr(g, 'page_idx', 0))
                if img: 
                    st.image(draw_bounding_box(img, coords, color=color, label=name))


def display_unified_evidence(data: Dict[str, Any], path: str):
    """Display all fields on a single image with color-coded bounding boxes."""
    # Group all matching chunks by page_idx
    page_chunks = defaultdict(list)
    
    for field_name, field_data in data.items():
        if field_data.get('value') and field_data.get('matching_chunks'):
            chunk = field_data['matching_chunks'][0]  # Take first matching chunk
            coords_list = get_chunk_coordinates(chunk, path)
            
            for item in coords_list:
                page_chunks[item['page_idx']].append({
                    'field_name': field_name,
                    'value': field_data['value'],
                    'coords': item['coords']
                })
    
    # Create a color mapping for fields
    color_map = {}
    for idx, field_name in enumerate(data.keys()):
        if data[field_name].get('value'):
            color_idx = idx % len(COLORS)
            color_map[field_name] = COLORS[color_idx]
    
    # Create a tabular display for the color legend
    st.markdown("### Extracted Fields Color Legend")
    
    # Create DataFrame-compatible data for st.table
    table_data = []
    for field_name, field_data in data.items():
        if field_data.get('value'):
            color = color_map.get(field_name, (255, 0, 0))
            color_hex = f"#{color[0]:02x}{color[1]:02x}{color[2]:02x}"
            table_data.append({
                "Field": field_name,
                "Extracted Value": field_data['value'],
                "Color": f"<div style='background-color:{color_hex}; width:20px; height:20px; border-radius:4px;'></div>"
            })
    
    # Display the table
    if table_data:
        # Use a markdown table for better styling control
        table_md = "| Field | Extracted Value | Color |\n| --- | --- | :---: |\n"
        for row in table_data:
            color = row["Color"]
            table_md += f"| {row['Field']} | {row['Extracted Value']} | {color} |\n"
        
        st.markdown(table_md, unsafe_allow_html=True)
    
    # For each page with data, create a combined visualization
    st.markdown("### Document with All Fields")
    for page_idx, chunks in page_chunks.items():
        # Only process if there are chunks on this page
        if not chunks:
            continue
            
        img = get_document_image(path, page_idx)
        if not img:
            continue
            
        # Start with the base image
        final_img = img.copy()
        
        # Add all bounding boxes
        for chunk_data in chunks:
            field_name = chunk_data['field_name']
            coords = chunk_data['coords']
            color = color_map.get(field_name, (255, 0, 0))  # Default to red if no color found
            
            final_img = draw_bounding_box(
                final_img, 
                coords, 
                color=color, 
                label=field_name
            )
        
        # Display the final image with all bounding boxes
        st.image(final_img, caption=f"Page {page_idx + 1} with all extracted fields")


def get_image_thumbnail(img: Image.Image, max_width: int = 300, max_height: int = 150) -> Image.Image:
    """Create a thumbnail of the given image with the specified max dimensions."""
    img_copy = img.copy()
    img_copy.thumbnail((max_width, max_height))
    return img_copy

def get_clickable_image(img: Image.Image, caption: str = "") -> str:
    """Create HTML for a clickable image that can be expanded."""
    # Convert image to base64
    buffered = io.BytesIO()
    img.save(buffered, format="PNG")
    img_str = base64.b64encode(buffered.getvalue()).decode()
    
    # Generate unique ID for this image
    img_id = f"img_{random.randint(10000, 99999)}"
    
    # Create HTML with JavaScript for click-to-zoom functionality
    html = f'''
    <style>
    .thumbnail {{
        cursor: pointer;
        border: 1px solid #ddd;
        border-radius: 4px;
        padding: 5px;
        transition: 0.3s;
        max-width: 100%;
    }}
    .thumbnail:hover {{
        box-shadow: 0 0 2px 1px rgba(0, 140, 186, 0.5);
    }}
    .modal {{
        display: none;
        position: fixed;
        z-index: 1000;
        padding-top: 100px;
        left: 0;
        top: 0;
        width: 100%;
        height: 100%;
        overflow: auto;
        background-color: rgba(0,0,0,0.9);
    }}
    .modal-content {{
        margin: auto;
        display: block;
        max-width: 90%;
        max-height: 90%;
    }}
    .caption {{
        margin: auto;
        display: block;
        width: 80%;
        text-align: center;
        color: white;
        padding: 10px 0;
    }}
    .close {{
        position: absolute;
        top: 15px;
        right: 35px;
        color: #f1f1f1;
        font-size: 40px;
        font-weight: bold;
        transition: 0.3s;
    }}
    .close:hover, .close:focus {{
        color: #bbb;
        text-decoration: none;
        cursor: pointer;
    }}
    </style>

    <img id="thumb_{img_id}" src="data:image/png;base64,{img_str}" class="thumbnail" onclick="document.getElementById('{img_id}').style.display='block'">
    
    <div id="{img_id}" class="modal">
        <span class="close" onclick="document.getElementById('{img_id}').style.display='none'">&times;</span>
        <img class="modal-content" src="data:image/png;base64,{img_str}">
        <div class="caption">{caption}</div>
    </div>
    
    <script>
        // Close modal when clicking outside of it
        window.onclick = function(event) {{
            if (event.target.id == '{img_id}') {{
                document.getElementById('{img_id}').style.display = "none";
            }}
        }}
    </script>
    '''
    return html

def get_download_link(df: pd.DataFrame, filename: str = "extracted_fields.xlsx") -> str:
    """Generate a download link for a DataFrame as Excel file."""
    output = io.BytesIO()
    with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
        df.to_excel(writer, sheet_name='Extracted Fields', index=False)
        # Adjust column widths
        worksheet = writer.sheets['Extracted Fields']
        for i, col in enumerate(df.columns):
            max_len = max(df[col].astype(str).map(len).max(), len(col)) + 2
            worksheet.set_column(i, i, max_len)
    
    b64 = base64.b64encode(output.getvalue()).decode()
    href = f'<a href="data:application/vnd.openxmlformats-officedocument.spreadsheetml.sheet;base64,{b64}" download="{filename}" class="download-button">Download Excel File</a>'
    
    # Add some CSS to style the button
    button_style = '''
    <style>
    .download-button {
        background-color: #4CAF50;
        border: none;
        color: white;
        padding: 12px 24px;
        text-align: center;
        text-decoration: none;
        display: inline-block;
        font-size: 16px;
        margin: 4px 2px;
        border-radius: 4px;
        cursor: pointer;
        transition: background-color 0.3s;
    }
    .download-button:hover {
        background-color: #45a049;
    }
    </style>
    '''
    
    return button_style + href

def process_file(client, file_path: str, selected_fields: List[Dict[str, str]]) -> Dict[str, Dict[str, Any]]:
    """Process a single file and extract fields."""
    if not agentic_imported:
        st.error("Cannot process without agentic_doc package. Please install it.")
        return None
    
    try:
        # Parse document
        results = parse_documents([file_path])
        doc = results[0]
        text = "\n".join(c.text for c in doc.chunks)
        
        # Extract fields
        data = extract_fields_with_openai(client, text, selected_fields, doc.chunks)
        return {
            'data': data,
            'doc': doc,
            'path': file_path
        }
    except Exception as e:
        st.error(f"Error processing file {os.path.basename(file_path)}: {e}")
        return None

def process_multiple_files(client, file_paths: List[str], selected_fields: List[Dict[str, str]]) -> List[Dict]:
    """Process multiple files and extract fields from each."""
    results = []
    
    # Use st.progress bar to show processing status
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    for i, file_path in enumerate(file_paths):
        status_text.text(f"Processing file {i+1}/{len(file_paths)}: {os.path.basename(file_path)}")
        result = process_file(client, file_path, selected_fields)
        if result:
            results.append(result)
        progress_bar.progress((i + 1) / len(file_paths))
    
    status_text.text("Processing complete!")
    return results

def display_option1_results(results: List[Dict], color_map: Dict[str, Tuple[int, int, int]]):
    """Display Option 1 results with tabular format and clickable thumbnails."""
    st.subheader("Extracted Fields with Individual Images")
    
    for result in results:
        st.markdown(f"## Document: {os.path.basename(result['path'])}")
        
        # Prepare data for table
        table_data = []
        html_images = {}
        
        for idx, (field_name, field_data) in enumerate(result['data'].items()):
            if field_data.get('value'):
                color = color_map.get(field_name, COLORS[idx % len(COLORS)])
                color_hex = f"#{color[0]:02x}{color[1]:02x}{color[2]:02x}"
                
                # Prepare image column data
                image_html = ""
                if field_data.get('matching_chunks'):
                    chunk = field_data['matching_chunks'][0]
                    if hasattr(chunk, 'grounding'):
                        for g in chunk.grounding:
                            box = getattr(g, 'box', None)
                            page_idx = getattr(g, 'page_idx', 0)
                            coords = parse_box_string(box) if isinstance(box, str) else (
                                getattr(box, 'l', None) is not None and [box.l, box.t, box.r, box.b]
                            )
                            
                            if coords:
                                img = get_document_image(result['path'], page_idx)
                                if img:
                                    # Create image with bounding box
                                    bbox_img = draw_bounding_box(img, coords, color=color, label=field_name)
                                    # Create thumbnail
                                    thumb = get_image_thumbnail(bbox_img)
                                    # Get clickable HTML
                                    caption = f"{field_name}: {field_data['value']} (Page {page_idx+1})"
                                    image_html = get_clickable_image(thumb, caption)
                                    break  # Just use the first valid grounding
                
                # Add to table data
                table_data.append({
                    "Field": field_name,
                    "Extracted Value": field_data['value'],
                    "Color": f"<div style='background-color:{color_hex}; width:20px; height:20px; border-radius:4px;'></div>",
                    "Reference Image": image_html
                })
        
        if table_data:
            # Convert to DataFrame for easier handling
            df = pd.DataFrame(table_data)
            
            # Create a copy without the HTML for Excel download
            df_excel = df.copy()
            df_excel = df_excel[['Field', 'Extracted Value']]  # Only include text columns
            
            # Show download button for Excel
            st.markdown(get_download_link(df_excel, f"extracted_{os.path.basename(result['path'])}.xlsx"), unsafe_allow_html=True)
            
            # Display table with HTML content
            st.write("### Extracted Fields")
            
            # We need to use st.markdown for the HTML in the table
            table_html = "<table style='width:100%; border-collapse: collapse;'>"
            # Header row
            table_html += "<tr style='background-color: #f2f2f2;'>"
            for col in ['Field', 'Extracted Value', 'Color', 'Reference Image']:
                table_html += f"<th style='border: 1px solid #ddd; padding: 8px; text-align: left;'>{col}</th>"
            table_html += "</tr>"
            
            # Data rows
            for _, row in df.iterrows():
                table_html += "<tr style='border-bottom: 1px solid #ddd;'>"
                table_html += f"<td style='padding: 8px; border: 1px solid #ddd;'>{row['Field']}</td>"
                table_html += f"<td style='padding: 8px; border: 1px solid #ddd;'>{row['Extracted Value']}</td>"
                table_html += f"<td style='padding: 8px; border: 1px solid #ddd;'>{row['Color']}</td>"
                table_html += f"<td style='padding: 8px; border: 1px solid #ddd;'>{row['Reference Image']}</td>"
                table_html += "</tr>"
            
            table_html += "</table>"
            st.markdown(table_html, unsafe_allow_html=True)
        else:
            st.warning("No fields were successfully extracted from this document.")

def is_valid_file_type(file_path: str) -> bool:
    """Check if file is a valid type (PDF or image)."""
    valid_extensions = ['.pdf', '.png', '.jpg', '.jpeg', '.tiff', '.tif', '.bmp']
    ext = os.path.splitext(file_path.lower())[1]
    return ext in valid_extensions

def check_directory_files(directory_path: str) -> Tuple[List[str], List[str]]:
    """Check directory and return lists of valid and invalid files."""
    valid_files = []
    invalid_files = []
    
    # Get all files in directory and subdirectories
    all_files = []
    for root, _, files in os.walk(directory_path):
        for file in files:
            file_path = os.path.join(root, file)
            all_files.append(file_path)
    
    # Check each file
    for file_path in all_files:
        if is_valid_file_type(file_path):
            valid_files.append(file_path)
        else:
            invalid_files.append(file_path)
    
    return valid_files, invalid_files

def main():
    st.set_page_config(layout="wide")
    st.title("📄 Invoice Field Extractor")
    st.write("Upload invoices to extract fields.")

    client = initialize_clients()
    if client is None:
        st.stop()

    selected = manage_fields()
    
    # Add visualization options
    st.subheader("Visualization Options")
    vis_option = st.radio(
        "Choose how to display extracted fields:",
        ["Option 1: Individual field images with tabular display",
         "Option 2: Multiple color-coded bounding boxes per reference document"]
    )
    
    # File upload options
    upload_option = st.radio(
        "Choose upload method:",
        ["Upload files", "Select local directory"]
    )
    
    # Store processed files in session state
    if 'processed_results' not in st.session_state:
        st.session_state.processed_results = []
    
    if upload_option == "Upload files":
        # Multi-file uploader
        uploaded_files = st.file_uploader(
            "Upload documents (PDF or images)", 
            type=['pdf', 'png', 'jpg', 'jpeg'], 
            accept_multiple_files=True
        )
        
        if uploaded_files and selected and st.button("Extract Fields"):
            st.session_state.processed_results = []  # Reset previous results
            
            with st.spinner("Processing files..."):
                # Create a temporary directory for processing
                with tempfile.TemporaryDirectory() as td:
                    file_paths = []
                    
                    # Save uploaded files to temp directory
                    for uploaded_file in uploaded_files:
                        file_path = Path(td) / f"in_{uploaded_file.name}"
                        file_path.write_bytes(uploaded_file.getvalue())
                        file_paths.append(str(file_path))
                    
                    # Process files
                    results = process_multiple_files(client, file_paths, selected)
                    if results:
                        st.session_state.processed_results = results
    
    else:  # Select local directory
        directory_path = st.text_input("Enter path to local directory:")
        
        if directory_path and os.path.isdir(directory_path):
            if st.button("Check Directory"):
                valid_files, invalid_files = check_directory_files(directory_path)
                
                if invalid_files:
                    st.warning(f"Found {len(invalid_files)} invalid file types. Only PDF and image files will be processed.")
                    st.write("Invalid files:")
                    for file in invalid_files[:10]:  # Show first 10 invalid files
                        st.write(f"- {file}")
                    if len(invalid_files) > 10:
                        st.write(f"...and {len(invalid_files) - 10} more")
                
                st.success(f"Found {len(valid_files)} valid files ready for processing.")
                
                # Store valid files in session state
                st.session_state.valid_directory_files = valid_files
                
                # Show extract button only if valid files found
                if valid_files and st.button("Extract Fields from Directory"):
                    st.session_state.processed_results = []  # Reset previous results
                    
                    with st.spinner("Processing files..."):
                        results = process_multiple_files(client, valid_files, selected)
                        if results:
                            st.session_state.processed_results = results
        elif directory_path:
            st.error("Invalid directory path. Please enter a valid path.")
    
    # Display results if available
    if st.session_state.processed_results:
        # Create color mapping
        color_map = {}
        # Use the first result to determine field colors consistently
        first_result = st.session_state.processed_results[0]
        for idx, field_name in enumerate(first_result['data'].keys()):
            color_idx = idx % len(COLORS)
            color_map[field_name] = COLORS[color_idx]
        
        if "Option 1" in vis_option:
            display_option1_results(st.session_state.processed_results, color_map)
        else:  # Option 2
            for result in st.session_state.processed_results:
                st.markdown(f"## Document: {os.path.basename(result['path'])}")
                display_unified_evidence(result['data'], result['path'])

if __name__ == "__main__":
    main()
