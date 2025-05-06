import os
from pathlib import Path
import json
import tempfile
import re
import random

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
        {"name": "Document No.", "description": "A unique document number, typically starting with numbers (e.g., 27xxxxxx)"},
        {"name": "D1", "description": "The D1 field value from the invoice"},
        {"name": "Comcode", "description": "The company code or identification number"},
        {"name": "Document Type", "description": "The type of document (e.g., Invoice, Receipt, etc.)"},
        {"name": "Year", "description": "The year the document was issued"},
        {"name": "Receiving Company Name", "description": "The name of the company receiving the invoice"},
        {"name": "Receiving Company Address", "description": "The complete address of the receiving company"},
        {"name": "Receiving Company Tax ID", "description": "The tax identification number of the receiving company"},
        {"name": "Date", "description": "The date the document was issued"},
        {"name": "Issuing Company Name", "description": "The name of the company issuing the invoice"}
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
    
    # Create a legend for the color coding
    st.markdown("### Color Legend")
    color_map = {}
    legend_cols = st.columns(5)
    for idx, (field_name, _) in enumerate(data.items()):
        if data[field_name].get('value'):
            color_idx = idx % len(COLORS)
            color = COLORS[color_idx]
            color_map[field_name] = color
            with legend_cols[idx % 5]:
                st.markdown(f"<div style='color:rgb{color};'>■</div> {field_name}: {data[field_name]['value']}", unsafe_allow_html=True)
    
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


def create_interactive_view(data: Dict[str, Any], path: str):
    """Create an interactive view with mouse-over highlighting."""
    # Prepare color mapping
    color_map = {}
    for idx, field_name in enumerate(data.keys()):
        if data[field_name].get('value'):
            color_idx = idx % len(COLORS)
            color_map[field_name] = COLORS[color_idx]
    
    # Initialize session state for highlighted field if not exists
    if 'highlighted_field' not in st.session_state:
        st.session_state.highlighted_field = None
    
    # Create a container for our interactive display
    st.markdown("### Interactive Field Extraction Results")
    
    # Create two columns - one for the table, one for the document
    col1, col2 = st.columns([2, 3])
    
    # Group data by page first - we'll need this for both columns
    page_data = defaultdict(list)
    field_images = {}
    
    # Prepare document images and bounding boxes data
    for field_name, field_info in data.items():
        if field_info.get('value') and field_info.get('matching_chunks'):
            chunk = field_info['matching_chunks'][0]
            coords_list = get_chunk_coordinates(chunk, path)
            
            for coord_item in coords_list:
                page_data[coord_item['page_idx']].append({
                    'field_name': field_name,
                    'coords': coord_item['coords'],
                    'color': color_map.get(field_name, (255, 0, 0))
                })
    
    # Render the table in first column
    with col1:
        st.markdown("#### Extracted Fields")
        
        # Create table container
        for field_name, field_data in data.items():
            if field_data.get('value'):
                color = color_map.get(field_name, (255, 0, 0))
                hex_color = "#{:02x}{:02x}{:02x}".format(*color)
                
                # Check if this field is highlighted
                is_highlighted = st.session_state.highlighted_field == field_name
                bg_color = hex_color + "30" if is_highlighted else "#ffffff"
                
                # Create a container for this field that highlights on click
                field_container = st.container()
                with field_container:
                    # Add the field to the table with clickable behavior
                    if st.button(
                        f"{field_name}: {field_data['value']}", 
                        key=f"field_{field_name}",
                        help=f"Click to highlight {field_name} in the document",
                        use_container_width=True,
                        type="secondary" if not is_highlighted else "primary"
                    ):
                        # Toggle highlighting
                        if st.session_state.highlighted_field == field_name:
                            st.session_state.highlighted_field = None
                        else:
                            st.session_state.highlighted_field = field_name
                        st.rerun()
    
    # Render the document view in second column
    with col2:
        st.markdown("#### Document View")
        
        # Process each page with data
        for page_idx, items in page_data.items():
            img = get_document_image(path, page_idx)
            if not img:
                continue
                
            # Create the base image and a highlighted version
            base_img = img.copy()
            
            # For each field, create a highlighted version of the image
            highlighted_images = {}
            for field_name in data.keys():
                if data[field_name].get('value'):
                    # Find items for this field on this page
                    field_items = [item for item in items if item['field_name'] == field_name]
                    
                    if field_items:
                        # Create a copy for this field
                        highlighted_img = img.copy()
                        
                        # Add all bounding boxes for this field
                        for item in field_items:
                            highlighted_img = draw_bounding_box(
                                highlighted_img,
                                item['coords'],
                                color=item['color'],
                                label=field_name
                            )
                        
                        highlighted_images[field_name] = highlighted_img
            
            # Create image area
            img_area = st.container()
            
            # If a field is highlighted, show that image, otherwise show base
            if st.session_state.highlighted_field and st.session_state.highlighted_field in highlighted_images:
                img_area.image(
                    highlighted_images[st.session_state.highlighted_field],
                    caption=f"Page {page_idx + 1} - Highlighting {st.session_state.highlighted_field}",
                    use_container_width=True
                )
            else:
                img_area.image(
                    base_img,
                    caption=f"Page {page_idx + 1}",
                    use_container_width=True
                )
                
            # Add clickable hotspots for each field on this page
            for item in items:
                field_name = item['field_name']
                coords = item['coords']
                
                # Create a small container to show which areas are clickable
                tooltip_container = st.empty()
                
                # Add a button that shows which field this region belongs to
                w, h = base_img.size
                x_center = int((coords[0] + coords[2]) * w / 2)
                y_center = int((coords[1] + coords[3]) * h / 2)
                
                # Define a narrow column where this hotspot should appear
                col = st.columns([coords[0], coords[2]-coords[0], 1-coords[2]])[1]
                
                with col:
                    # Create a transparent button that highlights this field when clicked
                    if st.button(
                        "",  # Empty label
                        key=f"hotspot_{field_name}_{page_idx}_{x_center}_{y_center}",
                        help=f"Highlight {field_name}",
                        type="secondary"
                    ):
                        # Toggle highlighting
                        if st.session_state.highlighted_field == field_name:
                            st.session_state.highlighted_field = None
                        else:
                            st.session_state.highlighted_field = field_name
                        st.rerun()


def main():
    st.set_page_config(layout="wide")
    st.title("📄 Invoice Field Extractor")
    st.write("Upload an invoice to extract fields.")

    client = initialize_clients()
    if client is None:
        st.stop()

    selected = manage_fields()
    
    # Add visualization options
    st.subheader("Visualization Options")
    vis_option = st.radio(
        "Choose how to display extracted fields:",
        ["Option 1: Output each field with corresponding reference image",
         "Option 2: Multiple color-coded bounding boxes per reference document",
         "Option 3: Highlight on mouse-over (interactive table and document view)"]
    )
    
    up = st.file_uploader("Upload document", type=['pdf', 'png', 'jpg', 'jpeg'])

    if up and selected and st.button("Extract Fields"):
        with st.spinner("Processing..."):
            # Create a temporary directory for processing
            with tempfile.TemporaryDirectory() as td:
                path = Path(td) / f"in{up.name}"
                path.write_bytes(up.getvalue())
                
                if agentic_imported:
                    try:
                        # Parse documents
                        results = parse_documents([str(path)])
                    except Exception as e:
                        st.error(f"Error during document parsing: {e}")
                        st.stop()

                    doc = results[0]
                    text = "\n".join(c.text for c in doc.chunks)
                    
                    data = extract_fields_with_openai(client, text, selected, doc.chunks)
                    
                    # Display results based on selected visualization option
                    if "Option 1" in vis_option:  # Original method - one image per field
                        st.subheader("Extracted Fields with Individual Images")
                        for idx, (nm, fd) in enumerate(data.items()):
                            if fd.get('value'):
                                st.markdown(f"### {nm}: {fd['value']}")
                                if fd.get('matching_chunks'):
                                    # Get color for this field
                                    color_idx = idx % len(COLORS)
                                    display_chunk_evidence(fd['matching_chunks'][0], nm, str(path), COLORS[color_idx])
                    
                    elif "Option 2" in vis_option:  # Combined visualization
                        st.subheader("Extracted Field Values")
                        # Display field values in a more compact format first
                        cols = st.columns(3)
                        for idx, (nm, fd) in enumerate(data.items()):
                            if fd.get('value'):
                                with cols[idx % 3]:
                                    st.markdown(f"**{nm}:** {fd['value']}")
                        
                        # Then show the unified visualization
                        display_unified_evidence(data, str(path))
                    
                    elif "Option 3" in vis_option:  # Interactive mouse-over
                        # Create the interactive view
                        create_interactive_view(data, str(path))
                        
                else:
                    st.error("Cannot process without agentic_doc package. Please install it.")

if __name__ == "__main__":
    main()
