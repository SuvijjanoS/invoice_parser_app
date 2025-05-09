import os
from pathlib import Path
import json
import tempfile
import re
import random
import time
from datetime import timedelta

import streamlit as st
from dotenv import load_dotenv
from openai import OpenAI
import fitz  # PyMuPDF for PDF handling
from typing import List, Dict, Any, Tuple, Optional
from PIL import Image, ImageDraw, ImageFont
from collections import defaultdict

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

# Constants for match/mismatch colors
MATCH_COLOR = (0, 255, 0)      # Green
MISMATCH_COLOR = (255, 0, 0)   # Red

def initialize_clients():
    """Initialize OpenAI client and Settings if possible."""
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
def get_option1_fields() -> List[Dict[str, str]]:
    """Fields to be used when Option 1 is selected."""
    return [
        {"name": "Document Type", "description": "The type of document (e.g., Invoice, Receipt, etc.)"},
        {"name": "Document Date", "description": "The date [dd-month-year] the document was issued"},
        {"name": "Receiving Company Name", "description": "The name of the company receiving the invoice"},
        {"name": "Receiving Company Address", "description": "The complete address of the receiving company"},
        {"name": "Receiving Company Tax ID", "description": "The tax identification number of the receiving company"},
        {"name": "Issuing Company Name", "description": "The name of the company/vendor/supplier issuing the invoice"},
        {"name": "Amount", "description": "The Grand Total amount of the invoice."},
        {"name": "VAT", "description": "The value-added-tax on the invoice."}
    ]

def get_reference_fields() -> List[Dict[str, str]]:
    """Fields to be used for reference documents and for files to check in Option 2."""
    return [
        {"name": "Receiving Company Name", "description": "The name of the company receiving the invoice"},
        {"name": "Receiving Company Address", "description": "The complete address of the receiving company"},
        {"name": "Receiving Company Tax ID", "description": "The tax identification number of the receiving company"},
    ]

def manage_fields(container, prefix):
    """Manage fields for extraction, with appropriate defaults based on prefix."""
    # No title here anymore - removed "Manage Fields" heading
    
    # Initialize with appropriate fields based on the section
    if f'{prefix}_extraction_fields' not in st.session_state:
        if prefix == "source":
            st.session_state[f'{prefix}_extraction_fields'] = get_option1_fields()
        else:  # prefix == "reference"
            st.session_state[f'{prefix}_extraction_fields'] = get_reference_fields()

    container.markdown("### Add New Field")
    col1, col2, col3 = container.columns([2, 3, 1])
    with col1:
        new_name = st.text_input("Field Name", key=f"{prefix}_new_name")
    with col2:
        new_desc = st.text_input("Field Description", key=f"{prefix}_new_desc")
    with col3:
        if st.button("Add Field", key=f"{prefix}_add_field") and new_name and new_desc:
            st.session_state[f'{prefix}_extraction_fields'].append({"name": new_name, "description": new_desc})
            st.session_state[f'{prefix}_new_name'] = ""
            st.session_state[f'{prefix}_new_desc'] = ""
            st.rerun()

    container.markdown("### Current Fields")
    for idx, fld in enumerate(st.session_state[f'{prefix}_extraction_fields']):
        c1, c2, c3, c4 = container.columns([2, 3, 1, 1])
        with c1:
            fld['name'] = st.text_input("Name", value=fld['name'], key=f"{prefix}_name_{idx}")
        with c2:
            fld['description'] = st.text_input("Description", value=fld['description'], key=f"{prefix}_desc_{idx}")
        with c3:
            if st.button("Remove", key=f"{prefix}_rm_{idx}"):
                st.session_state[f'{prefix}_extraction_fields'].pop(idx)
                st.rerun()
        with c4:
            st.checkbox("Extract", value=True, key=f"{prefix}_ext_{idx}")

    if st.button("Reset to Default Fields", key=f"{prefix}_reset"):
        if prefix == "source":
            st.session_state[f'{prefix}_extraction_fields'] = get_option1_fields()
        else:  # prefix == "reference"
            st.session_state[f'{prefix}_extraction_fields'] = get_reference_fields()
        st.rerun()

    return [fld for i, fld in enumerate(st.session_state[f'{prefix}_extraction_fields']) 
            if st.session_state.get(f"{prefix}_ext_{i}", True)]


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
    """Convert a single PDF page to a PIL Image."""
    try:
        doc = fitz.open(pdf_path)
        pix = doc[page_number].get_pixmap(matrix=fitz.Matrix(2,2))
        img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
        doc.close()
        return img
    except Exception as e:
        st.error(f"Error converting PDF: {e}")
        return None


def convert_pdf_to_images(pdf_path: str) -> List[Image.Image]:
    """Convert all pages of a PDF to a list of PIL Images."""
    images = []
    try:
        doc = fitz.open(pdf_path)
        for page_num in range(len(doc)):
            pix = doc[page_num].get_pixmap(matrix=fitz.Matrix(2,2))
            img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
            images.append(img)
        doc.close()
        return images
    except Exception as e:
        st.error(f"Error converting PDF to images: {e}")
        return []


def get_document_image(path: str, idx: int) -> Image.Image:
    """Get a document image, handling both PDFs and images."""
    return convert_pdf_page_to_image(path, idx) if path.lower().endswith('.pdf') else Image.open(path)


def parse_box_string(s: str):
    """Parse a box string into coordinates."""
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


def display_comparison_evidence(source_data: Dict[str, Any], reference_data: Dict[str, Any], 
                               source_path: str):
    """Display comparison between source and reference with match/mismatch highlighting."""
    # Group all matching chunks by page_idx
    page_chunks = defaultdict(list)
    
    # Build comparison data
    comparison_results = {}
    
    for field_name, field_data in source_data.items():
        source_value = field_data.get('value')
        reference_value = reference_data.get(field_name, {}).get('value')
        
        match_status = False
        if source_value and reference_value:
            # Basic string comparison (could be enhanced with fuzzy matching)
            match_status = source_value.strip().lower() == reference_value.strip().lower()
        
        comparison_results[field_name] = {
            'source_value': source_value,
            'reference_value': reference_value,
            'match': match_status
        }
        
        # Get coordinates for visualization
        if field_data.get('value') and field_data.get('matching_chunks'):
            chunk = field_data['matching_chunks'][0]  # Take first matching chunk
            coords_list = get_chunk_coordinates(chunk, source_path)
            
            for item in coords_list:
                color = MATCH_COLOR if match_status else MISMATCH_COLOR
                page_chunks[item['page_idx']].append({
                    'field_name': field_name,
                    'value': field_data['value'],
                    'coords': item['coords'],
                    'match': match_status
                })
    
    # Create a tabular display for comparison results
    st.markdown("### Field Comparison Results")
    
    # Create DataFrame-compatible data for comparison table
    table_md = "| Field | Source Value | Reference Value | Match |\n| --- | --- | --- | :---: |\n"
    
    for field_name, result in comparison_results.items():
        source_val = result['source_value'] or "Not found"
        ref_val = result['reference_value'] or "Not found"
        match_icon = "✅" if result['match'] else "❌"
        table_md += f"| {field_name} | {source_val} | {ref_val} | {match_icon} |\n"
    
    st.markdown(table_md)
    
    # For each page with data, create a combined visualization
    st.markdown("### Document with Match/Mismatch Highlighting")
    for page_idx, chunks in page_chunks.items():
        # Only process if there are chunks on this page
        if not chunks:
            continue
            
        img = get_document_image(source_path, page_idx)
        if not img:
            continue
            
        # Start with the base image
        final_img = img.copy()
        
        # Add all bounding boxes
        for chunk_data in chunks:
            field_name = chunk_data['field_name']
            coords = chunk_data['coords']
            color = MATCH_COLOR if chunk_data['match'] else MISMATCH_COLOR
            match_status = "✓" if chunk_data['match'] else "✗"
            label = f"{field_name} {match_status}"
            
            final_img = draw_bounding_box(
                final_img, 
                coords, 
                color=color, 
                label=label
            )
        
        # Display the final image with all bounding boxes
        st.image(final_img, caption=f"Page {page_idx + 1} with match/mismatch highlighting")


def save_image_file(img: Image.Image, temp_dir: Path, filename: str) -> Path:
    """Save a PIL Image to a temporary file."""
    out_path = temp_dir / filename
    img.save(out_path)
    return out_path


def format_time(seconds: float) -> str:
    """Format seconds into minutes and seconds."""
    minutes = int(seconds // 60)
    seconds = seconds % 60
    return f"{minutes} mins {seconds:.2f} secs"


def process_file(file_path: Path, selected_fields, client, temp_dir: Path):
    """Process a single file and return extracted data."""
    file_paths_to_process = []
    
    # If PDF, convert to images first
    if str(file_path).lower().endswith('.pdf'):
        images = convert_pdf_to_images(str(file_path))
        
        # Save each page as an image file
        for i, img in enumerate(images):
            img_path = save_image_file(img, temp_dir, f"page_{i}_{file_path.name}.png")
            file_paths_to_process.append(img_path)
    else:
        # For image files, use directly
        file_paths_to_process.append(file_path)
    
    all_results = []
    
    # Process all paths for this file
    for path_idx, path in enumerate(file_paths_to_process):
        if agentic_imported:
            try:
                # Parse documents
                results = parse_documents([str(path)])
                doc = results[0]
                text = "\n".join(c.text for c in doc.chunks)
                
                data = extract_fields_with_openai(client, text, selected_fields, doc.chunks)
                all_results.append({
                    'path': path,
                    'data': data,
                    'doc': doc
                })
                
            except Exception as e:
                st.error(f"Error processing {path.name}: {e}")
        else:
            st.error("Cannot process without agentic_doc package. Please install it.")
            return []
    
    return all_results


def main():
    st.set_page_config(layout="wide", page_title="Document Field Extractor (Invoices)")
    
    # Check if client can be initialized
    client = initialize_clients()
    if client is None:
        st.stop()
    
    # Create the three-frame layout
    st.markdown("# 📄 Document Field Extractor (Invoices)")
    
    # First, define the processing option
    st.markdown("## Processing Options")
    process_option = st.radio(
        "Choose processing option:",
        ["Option 1: Extract desired field information only",
         "Option 2: Extract and Compare fields against reference document"]
    )
    
    # Split the screen into top and bottom frames
    top_container = st.container()
    bottom_container = st.container()
    
    # Split the top container into left and right
    with top_container:
        # For Option 1, only show Files to Check section
        if "Option 1" in process_option:
            # Files to check section
            st.markdown("## Files to Check")
            source_selected = manage_fields(top_container, "source")
            source_files = st.file_uploader("Upload documents to check", 
                                          type=['pdf', 'png', 'jpg', 'jpeg'], 
                                          accept_multiple_files=True,
                                          key="source_files")
            
            # Create a hidden reference for reference files without showing UI
            # Use session state to store empty reference files
            if "reference_files" not in st.session_state:
                st.session_state.reference_files = []
            reference_files = st.session_state.reference_files
        
        # For Option 2, show both columns
        else:
            left_col, right_col = st.columns(2)
            
            # Left frame (Files to check)
            with left_col:
                st.markdown("## Files to Check")
                st.info("When comparing documents, both 'Files to Check' and 'Reference Files' use the same field set defined under 'Reference Files'.")
                source_selected = get_reference_fields()  # Use reference fields for Option 2
                source_files = st.file_uploader("Upload documents to check", 
                                              type=['pdf', 'png', 'jpg', 'jpeg'], 
                                              accept_multiple_files=True,
                                              key="source_files")
            
            # Right frame (Reference files)
            with right_col:
                st.markdown("## Reference Files")
                reference_selected = manage_fields(right_col, "reference")
                reference_files = st.file_uploader("Upload reference documents", 
                                                 type=['pdf', 'png', 'jpg', 'jpeg'], 
                                                 accept_multiple_files=True,
                                                 key="reference_files")
    
    # Bottom frame (Additional Options)
    with bottom_container:
        # Only show visualization options for Option 1
        if "Option 1" in process_option:
            st.markdown("## Visualization Options")
            vis_option = st.radio(
                "Choose visualization style:",
                ["Show each field with corresponding reference image",
                 "Show multiple color-coded bounding boxes per document"],
                key="vis_option"
            )
        else:
            # For Option 2, the visualization is fixed to the comparison view
            vis_option = "comparison"  # Default for Option 2
        
        # Add spacer
        st.markdown("---")
        
        # Process button
        if st.button("Process Documents"):
            if not source_files:
                st.error("Please upload at least one document to check")
                st.stop()
                
            if "Option 2" in process_option and not reference_files:
                st.error("Please upload at least one reference document for comparison")
                st.stop()
            
            # Create a temporary directory for processing
            with tempfile.TemporaryDirectory() as td:
                temp_dir = Path(td)
                
                # Initialize timing metrics
                start_time = time.time()
                
                # Create progress tracking elements
                progress_bar = st.progress(0)
                progress_text = st.empty()
                
                # Option 1: Extract only
                if "Option 1" in process_option:
                    st.subheader("Extraction Results")
                    
                    # Calculate total file size
                    total_kb = sum(file.size / 1024 for file in source_files)
                    st.write(f"Total file size: {total_kb:.2f} KB")
                    
                    # Process each source file
                    for file_idx, uploaded_file in enumerate(source_files):
                        progress_text.text(f"Processing file {file_idx + 1}/{len(source_files)}: {uploaded_file.name}")
                        progress_bar.progress((file_idx) / len(source_files))
                        
                        # Save uploaded file to temp directory
                        original_path = temp_dir / f"source_{uploaded_file.name}"
                        original_path.write_bytes(uploaded_file.getvalue())
                        
                        # Process timing for this specific file
                        file_start_time = time.time()
                        
                        # Process the file using option1 fields
                        results = process_file(original_path, source_selected, client, temp_dir)
                        
                        # Calculate timing information
                        file_end_time = time.time()
                        file_processing_time = file_end_time - file_start_time
                        file_size_kb = original_path.stat().st_size / 1024
                        time_per_kb = file_processing_time / file_size_kb if file_size_kb > 0 else 0
                        
                        st.success(f"✅ File {uploaded_file.name} processed in {format_time(file_processing_time)}")
                        st.info(f"File size: {file_size_kb:.2f} KB | Time per KB: {format_time(time_per_kb)} per KB")
                        
                        # Display results for each page/image processed
                        for result in results:
                            st.subheader(f"Results for {result['path'].name}")
                            
                            # Display based on visualization option
                            if "each field" in vis_option:
                                # Option 1 - Individual images per field
                                for idx, (field_name, field_data) in enumerate(result['data'].items()):
                                    if field_data.get('value'):
                                        st.markdown(f"### {field_name}: {field_data['value']}")
                                        if field_data.get('matching_chunks'):
                                            # Get color for this field
                                            color_idx = idx % len(COLORS)
                                            display_chunk_evidence(
                                                field_data['matching_chunks'][0], 
                                                field_name, 
                                                str(result['path']), 
                                                COLORS[color_idx]
                                            )
                            else:
                                # Option 2 - Unified visualization
                                display_unified_evidence(result['data'], str(result['path']))
                
                # Option 2: Extract and Compare
                else:
                    st.subheader("Comparison Results")
                    
                    # Check for uploaded files
                    if not source_files or not reference_files:
                        st.error("Please upload both source and reference documents")
                        st.stop()
                    
                    # Make sure we have the reference fields defined
                    reference_fields = get_reference_fields()
                    
                    # Get the reference file (we'll use the first one)
                    reference_file = reference_files[0]
                    reference_path = temp_dir / f"reference_{reference_file.name}"
                    reference_path.write_bytes(reference_file.getvalue())
                    
                    # Process the reference file once
                    progress_text.text(f"Processing reference file: {reference_file.name}")
                    progress_bar.progress(0.2)
                    reference_results = process_file(reference_path, reference_fields, client, temp_dir)
                    
                    if not reference_results:
                        st.error("Error processing reference file")
                        st.stop()
                    
                    reference_data = reference_results[0]['data']
                    
                    # Process all source files
                    total_size_kb = reference_file.size / 1024
                    total_files = len(source_files)
                    
                    for file_idx, source_file in enumerate(source_files):
                        # Update progress
                        progress_value = 0.2 + (0.8 * (file_idx / total_files))
                        progress_text.text(f"Processing source file {file_idx+1}/{total_files}: {source_file.name}")
                        progress_bar.progress(progress_value)
                        
                        # Save source file
                        source_path = temp_dir / f"source_{source_file.name}"
                        source_path.write_bytes(source_file.getvalue())
                        total_size_kb += source_file.size / 1024
                        
                        # Process timing
                        comparison_start_time = time.time()
                        
                        # Process source file
                        source_results = process_file(source_path, reference_fields, client, temp_dir)
                        
                        if not source_results:
                            st.error(f"Error processing source file: {source_file.name}")
                            continue
                        
                        # Display comparison for this file
                        st.markdown(f"### Comparison Results for: {source_file.name}")
                        source_data = source_results[0]['data']
                        
                        # Display comparison
                        display_comparison_evidence(
                            source_data, 
                            reference_data, 
                            str(source_results[0]['path'])
                        )
                        
                        # Calculate timing information
                        comparison_end_time = time.time()
                        comparison_time = comparison_end_time - comparison_start_time
                        file_size_kb = source_path.stat().st_size / 1024
                        time_per_kb = comparison_time / file_size_kb if file_size_kb > 0 else 0
                        
                        st.success(f"✅ Comparison for {source_file.name} completed in {format_time(comparison_time)}")
                        st.info(f"File size: {file_size_kb:.2f} KB | Time per KB: {format_time(time_per_kb)} per KB")
                        
                        # Add a divider between files
                        if file_idx < len(source_files) - 1:
                            st.markdown("---")
                
                # Update progress to complete
                progress_bar.progress(1.0)
                progress_text.text("Processing complete!")
                
                # Display overall timing information
                end_time = time.time()
                total_processing_time = end_time - start_time
                
                # For Option 1, calculate total KB
                if "Option 1" in process_option:
                    total_kb = sum(file.size / 1024 for file in source_files)
                    time_per_kb_overall = total_processing_time / total_kb if total_kb > 0 else 0
                else:
                    # For Option 2, use source + reference file size
                    total_kb = source_file.size / 1024 + reference_file.size / 1024
                    time_per_kb_overall = total_processing_time / total_kb if total_kb > 0 else 0
                
                # Create a metrics display for timing
                st.markdown("## Processing Time Metrics")
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Total Processing Time", format_time(total_processing_time))
                with col2:
                    st.metric("Total File Size", f"{total_kb:.2f} KB")
                with col3:
                    st.metric("Avg Time per KB", format_time(time_per_kb_overall))

if __name__ == "__main__":
    main()
