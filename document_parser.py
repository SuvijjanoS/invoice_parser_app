import os
from pathlib import Path
import json
import tempfile
import re
import random
import time
from datetime import timedelta
import io

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
    from agentic_doc.parse import parse_documents, Document as AgenticDocument
    from agentic_doc.config import Settings
    # Attempt to import a visualizer if available (actual name might differ)
    try:
        from agentic_doc.utils.visualize import viz_parsed_document # Check if this path is correct
        agentic_visualizer_imported = True
    except ImportError:
        agentic_visualizer_imported = False
        st.warning("⚠️ agentic_doc visualizer (e.g., viz_parsed_document) not found. Using manual drawing.")
    agentic_imported = True
except ImportError:
    agentic_imported = False
    st.error("⚠️ agentic_doc package not installed. Please install it with `pip install agentic_doc`")
    # Stop execution if agentic_doc is critical and not found
    # st.stop()


# Load local .env for development; secrets will override in cloud
load_dotenv()

# Fetch API keys: prefer Streamlit secrets, fallback to environment
OPENAI_API_KEY = st.secrets.get("OPENAI_API_KEY", os.getenv("OPENAI_API_KEY"))
VISION_AGENT_API_KEY = st.secrets.get("VISION_AGENT_API_KEY", os.getenv("VISION_AGENT_API_KEY")) # This is agentic_doc API key

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
    openai_client = None
    agentic_settings = None

    if OPENAI_API_KEY:
        openai_client = OpenAI(api_key=OPENAI_API_KEY)
    else:
        # OpenAI client might not be needed for all new options,
        # but existing options depend on it.
        # We can show a warning if it's expected for some operations.
        pass # Allow to proceed if only agentic-doc is needed for an option

    if agentic_imported:
        if not VISION_AGENT_API_KEY:
            st.error(
                "🔑 Vision Agent API key (for Agentic Doc) missing!\n\n"
                "Please go to your Streamlit Cloud app's Settings → Secrets, and add:\n"
                "  • VISION_AGENT_API_KEY = <your Agentic Doc key>\n"
            )
            # Depending on the option, this might be a stopping error
            # For now, allow proceeding, errors will be caught later if used.
        else:
            try:
                agentic_settings = Settings(
                    vision_agent_api_key=VISION_AGENT_API_KEY,
                    batch_size=4, # Default, can be configured
                    max_workers=5, # Default
                    # include_marginalia=True, # Example: if you want to try this
                    # include_metadata_in_markdown=True # Example
                )
            except Exception as e:
                st.error(f"Error initializing Agentic Doc Settings: {e}")
                return openai_client, None # Return None for settings if error
    
    # If only Option 3 or 4 is used, OpenAI client might not be strictly needed for all parts
    # But Option 1 and 2 rely on it.
    if not openai_client and (st.session_state.get("process_option","").startswith("Option 1") or \
                             st.session_state.get("process_option","").startswith("Option 2")):
         st.error(
            "🔑 OpenAI API key missing!\n\n"
            "Needed for Option 1 & 2. Please go to your Streamlit Cloud app's Settings → Secrets, and add:\n"
            "  • OPENAI_API_KEY = <your OpenAI key>\n"
        )
         # st.stop() # Or handle gracefully

    return openai_client, agentic_settings


# Helper functions (existing ones like get_option1_fields, manage_fields, etc. remain mostly unchanged)
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
            # Clear inputs after adding
            st.session_state[f'{prefix}_new_name'] = ""
            st.session_state[f'{prefix}_new_desc'] = ""
            st.rerun()

    container.markdown("### Current Fields")
    fields_to_remove = []
    for idx, fld in enumerate(st.session_state[f'{prefix}_extraction_fields']):
        c1, c2, c3, c4 = container.columns([2, 3, 1, 1])
        with c1:
            fld['name'] = st.text_input("Name", value=fld['name'], key=f"{prefix}_name_{idx}")
        with c2:
            fld['description'] = st.text_input("Description", value=fld['description'], key=f"{prefix}_desc_{idx}")
        with c3:
            if st.button("Remove", key=f"{prefix}_rm_{idx}"):
                fields_to_remove.append(idx)
        with c4:
            # Ensure checkbox state is initialized if not present
            if f"{prefix}_ext_{idx}" not in st.session_state:
                st.session_state[f"{prefix}_ext_{idx}"] = True
            st.checkbox("Extract", value=st.session_state[f"{prefix}_ext_{idx}"], key=f"{prefix}_ext_{idx}")
    
    if fields_to_remove:
        for idx in sorted(fields_to_remove, reverse=True):
            st.session_state[f'{prefix}_extraction_fields'].pop(idx)
        st.rerun()


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
    if not client:
        st.error("OpenAI client not initialized. Cannot extract fields with OpenAI.")
        return {field['name']: {'value': None} for field in fields}
        
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
            model="gpt-4", # Consider making model configurable
            messages=[
                {"role": "system", "content": "You are a specialized invoice parser that accurately extracts fields."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.1
        )
    except Exception as e:
        status = getattr(e, 'status_code', None) or getattr(e, 'http_status', None)
        code = getattr(e, 'code', None)
        
        if status == 402 or code == 'insufficient_quota':
            st.error("🚫 You've run out of API credits for OpenAI. Please add more credits to continue.")
            st.stop()
        else:
            st.error(f"OpenAI API error: {e}")
            st.stop()

    try:
        extracted_data = json.loads(response.choices[0].message.content)
        if chunks: # This part for associating chunks might need refinement based on how agentic-doc chunks map
            for field_name, field_data in extracted_data.items():
                if field_data.get('value'):
                    matching_chunks = []
                    value_to_match = str(field_data['value']).strip().lower() # Ensure string for matching
                    for chunk in chunks:
                        if value_to_match in chunk.text.lower():
                            matching_chunks.append(chunk)
                    field_data['matching_chunks'] = matching_chunks # These are agentic-doc chunks
        return extracted_data
    except json.JSONDecodeError:
        st.error("Failed to decode JSON from OpenAI response.")
        return {field['name']: {'value': None} for field in fields}
    except AttributeError: # Handle cases where response.choices[0].message.content is not as expected
        st.error("Unexpected response structure from OpenAI.")
        return {field['name']: {'value': None} for field in fields}


def convert_pdf_page_to_image(pdf_path: str, page_number: int, dpi=200) -> Image.Image:
    """Convert a single PDF page to a PIL Image."""
    try:
        doc = fitz.open(pdf_path)
        if not 0 <= page_number < len(doc):
            st.error(f"Page number {page_number} is out of range for PDF with {len(doc)} pages.")
            doc.close()
            return None
        # Higher DPI for better quality, matrix scales rendering
        mat = fitz.Matrix(dpi/72, dpi/72) 
        pix = doc[page_number].get_pixmap(matrix=mat)
        img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
        doc.close()
        return img
    except Exception as e:
        st.error(f"Error converting PDF page {page_number} to image: {e}")
        return None


def convert_pdf_to_images(pdf_path: str, dpi=200) -> List[Image.Image]:
    """Convert all pages of a PDF to a list of PIL Images."""
    images = []
    try:
        doc = fitz.open(pdf_path)
        for page_num in range(len(doc)):
            mat = fitz.Matrix(dpi/72, dpi/72)
            pix = doc[page_num].get_pixmap(matrix=mat)
            img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
            images.append(img)
        doc.close()
        return images
    except Exception as e:
        st.error(f"Error converting PDF to images: {e}")
        return []


def get_document_image(path: str, page_idx: int) -> Optional[Image.Image]:
    """Get a document image (PDF page or direct image)."""
    try:
        if Path(path).suffix.lower() == '.pdf':
            return convert_pdf_page_to_image(path, page_idx)
        elif page_idx == 0: # For non-PDFs, only page_idx 0 is valid
            return Image.open(path)
        else:
            st.warning(f"Requested page {page_idx} for non-PDF file {path}. Returning None.")
            return None
    except Exception as e:
        st.error(f"Error opening image {path}: {e}")
        return None

def parse_box_string(s: str) -> Optional[List[float]]:
    """Parse a box string 'l=0.1 t=0.2 r=0.3 b=0.4' into [l, t, r, b]."""
    try:
        coords = {}
        parts = s.split()
        for part in parts:
            key, value = part.split('=')
            coords[key] = float(value)
        return [coords['l'], coords['t'], coords['r'], coords['b']]
    except Exception: # More specific exceptions could be caught
        # st.error(f"Error parsing box string: {s}, {e}") # Can be noisy
        return None


def draw_bounding_box(img: Image.Image, box: List[float], color: Tuple[int, int, int] = (255, 0, 0),
                     label: Optional[str] = None, width: int = 2) -> Image.Image:
    """Draw a bounding box on an image with optional label."""
    out_img = img.copy()
    draw = ImageDraw.Draw(out_img)
    img_w, img_h = out_img.size

    # Box coordinates are relative (0.0 to 1.0)
    x0, y0, x1, y1 = int(box[0]*img_w), int(box[1]*img_h), int(box[2]*img_w), int(box[3]*img_h)
    
    draw.rectangle([x0, y0, x1, y1], outline=color, width=width)

    if label:
        try:
            font = ImageFont.truetype("arial.ttf", 14) # Check if font is available
        except IOError:
            font = ImageFont.load_default()
        
        text_bbox = draw.textbbox((x0, y0 - 16), label, font=font) # Use textbbox for modern Pillow
        text_w = text_bbox[2] - text_bbox[0]
        text_h = text_bbox[3] - text_bbox[1]

        # Adjust text position if it goes off-screen
        text_y_pos = y0 - text_h - 4
        if text_y_pos < 0: # If text goes above image, draw below box start
            text_y_pos = y0 + 2

        draw.rectangle([x0, text_y_pos, x0 + text_w + 4, text_y_pos + text_h + 4], fill=color)
        draw.text((x0 + 2, text_y_pos + 2), label, fill=(255, 255, 255), font=font)
    
    return out_img


def get_chunk_coordinates(chunk: Any, file_path_for_error: str) -> List[Dict[str, Any]]:
    """Get coordinates for a chunk's bounding boxes.
       Chunk is expected to be an agentic-doc chunk object.
    """
    coords_list = []
    if hasattr(chunk, 'grounding') and chunk.grounding:
        for g_idx, g in enumerate(chunk.grounding):
            page_idx = getattr(g, 'page_idx', 0) # Default to page 0 if not specified
            box_data = getattr(g, 'box', None)

            if box_data:
                if isinstance(box_data, str): # e.g. "l=0.07 t=0.1 r=0.9 b=0.2"
                    coords = parse_box_string(box_data)
                elif hasattr(box_data, 'l') and hasattr(box_data, 't') and \
                     hasattr(box_data, 'r') and hasattr(box_data, 'b'): # If it's an object with l,t,r,b
                    coords = [box_data.l, box_data.t, box_data.r, box_data.b]
                else:
                    # st.warning(f"Unsupported box format in grounding for chunk in {file_path_for_error}: {box_data}")
                    coords = None
                
                if coords and all(isinstance(c, float) for c in coords) and len(coords) == 4:
                    coords_list.append({'coords': coords, 'page_idx': page_idx})
                # else:
                    # st.warning(f"Failed to parse or invalid coords for chunk in {file_path_for_error}, grounding {g_idx}, box: {box_data}")
    return coords_list


def display_chunk_evidence(chunk, name: str, path: str, color: Tuple[int, int, int] = (255, 0, 0)):
    """Display a single chunk with bounding box. (Used in Option 1)"""
    coords_data = get_chunk_coordinates(chunk, path)
    if coords_data:
        # Display evidence from the first coordinate set found for the chunk
        # This might need adjustment if a chunk spans multiple boxes/pages in a way
        # that needs specific handling for Option 1's display logic.
        first_coord_info = coords_data[0]
        img = get_document_image(path, first_coord_info['page_idx'])
        if img:
            st.image(draw_bounding_box(img, first_coord_info['coords'], color=color, label=name))
    # else:
    #     st.write(f"No valid grounding coordinates found for field '{name}'.")


def display_unified_evidence(extracted_data: Dict[str, Any], original_doc_path: str, agentic_doc_object: Optional[Any] = None):
    """Display all fields on a single image with color-coded bounding boxes. (Used in Option 1)
       `extracted_data` is the OpenAI output.
       `agentic_doc_object` is the result from agentic_doc.parse_documents (optional, for better grounding).
    """
    page_visualizations = defaultdict(lambda: {'image': None, 'boxes': []})
    
    field_color_map = {}
    color_idx_counter = 0

    # Prepare field colors and gather bounding boxes
    for field_name, field_info in extracted_data.items():
        if field_info.get('value'): # Field was found
            if field_name not in field_color_map:
                field_color_map[field_name] = COLORS[color_idx_counter % len(COLORS)]
                color_idx_counter += 1
            
            color = field_color_map[field_name]
            
            # Prefer matching_chunks if available (from OpenAI extraction step)
            chunks_to_draw = field_info.get('matching_chunks', [])
            
            if not chunks_to_draw and agentic_doc_object: # Fallback: find text in agentic_doc_object if specific chunks aren't pre-matched
                # This fallback can be complex if values are not exact matches.
                # For simplicity, this example assumes 'matching_chunks' is populated correctly.
                pass


            for chunk_obj in chunks_to_draw: # These are agentic_doc chunks
                coords_list = get_chunk_coordinates(chunk_obj, original_doc_path)
                for coord_item in coords_list:
                    page_idx = coord_item['page_idx']
                    if page_visualizations[page_idx]['image'] is None:
                        page_visualizations[page_idx]['image'] = get_document_image(original_doc_path, page_idx)
                    
                    if page_visualizations[page_idx]['image']: # Ensure image was loaded
                        page_visualizations[page_idx]['boxes'].append({
                            'coords': coord_item['coords'],
                            'color': color,
                            'label': field_name
                        })
    
    # Display legend
    st.markdown("### Extracted Fields Color Legend")
    legend_data = []
    for field_name, color in field_color_map.items():
        if extracted_data.get(field_name, {}).get('value'): # Only show legend for fields found
            color_hex = f"#{color[0]:02x}{color[1]:02x}{color[2]:02x}"
            legend_data.append({
                "Field": field_name,
                "Extracted Value": extracted_data[field_name]['value'],
                "Color": f"<div style='background-color:{color_hex}; width:20px; height:20px; border-radius:4px;'></div>"
            })
    if legend_data:
        table_md = "| Field | Extracted Value | Color |\n| --- | --- | :---: |\n"
        for row in legend_data:
            table_md += f"| {row['Field']} | {row['Extracted Value']} | {row['Color']} |\n"
        st.markdown(table_md, unsafe_allow_html=True)
    else:
        st.write("No fields extracted or no values found to display in legend.")

    # Display images with boxes
    st.markdown("### Document with All Fields Highlighted")
    if not page_visualizations:
        st.write("No visual evidence to display (no fields extracted or no bounding boxes found).")
        return

    for page_idx, viz_data in sorted(page_visualizations.items()):
        img = viz_data['image']
        if img:
            final_img = img.copy()
            for box_info in viz_data['boxes']:
                final_img = draw_bounding_box(final_img, box_info['coords'], box_info['color'], box_info['label'])
            st.image(final_img, caption=f"Page {page_idx + 1} with extracted fields")
        # else:
        #     st.warning(f"Could not load image for page {page_idx + 1} of {original_doc_path}")


def display_comparison_evidence(source_data: Dict[str, Any], reference_data: Dict[str, Any],
                                source_doc_path: str, source_agentic_doc: Optional[Any] = None):
    """Display comparison with match/mismatch highlighting. (Used in Option 2)"""
    page_visualizations = defaultdict(lambda: {'image': None, 'boxes': []})
    comparison_results_table = []

    for field_name, source_field_info in source_data.items():
        source_value = source_field_info.get('value')
        ref_value_info = reference_data.get(field_name, {})
        ref_value = ref_value_info.get('value')
        
        is_match = False
        if source_value is not None and ref_value is not None:
            # Normalize for comparison (simple case)
            is_match = str(source_value).strip().lower() == str(ref_value).strip().lower()
        elif source_value is None and ref_value is None:
            is_match = True # Both not found can be considered a "match" in terms of absence

        comparison_results_table.append({
            "Field": field_name,
            "Source Value": source_value if source_value is not None else "N/A",
            "Reference Value": ref_value if ref_value is not None else "N/A",
            "Match": "✅" if is_match else "❌"
        })

        if source_value is not None: # Only draw boxes for fields found in source
            color = MATCH_COLOR if is_match else MISMATCH_COLOR
            
            # Prefer matching_chunks if available
            chunks_to_draw = source_field_info.get('matching_chunks', [])
            # Fallback if necessary (similar to display_unified_evidence) could be added here

            for chunk_obj in chunks_to_draw:
                coords_list = get_chunk_coordinates(chunk_obj, source_doc_path)
                for coord_item in coords_list:
                    page_idx = coord_item['page_idx']
                    if page_visualizations[page_idx]['image'] is None:
                        page_visualizations[page_idx]['image'] = get_document_image(source_doc_path, page_idx)
                    
                    if page_visualizations[page_idx]['image']:
                        page_visualizations[page_idx]['boxes'].append({
                            'coords': coord_item['coords'],
                            'color': color,
                            'label': f"{field_name} {'✓' if is_match else '✗'}"
                        })
    
    st.markdown("### Field Comparison Results")
    if comparison_results_table:
        table_md = "| Field | Source Value | Reference Value | Match |\n| --- | --- | --- | :---: |\n"
        for row in comparison_results_table:
            table_md += f"| {row['Field']} | {row['Source Value']} | {row['Reference Value']} | {row['Match']} |\n"
        st.markdown(table_md)
    else:
        st.write("No fields to compare.")

    st.markdown("### Document with Match/Mismatch Highlighting")
    if not page_visualizations:
        st.write("No visual evidence to display for comparison.")
        return

    for page_idx, viz_data in sorted(page_visualizations.items()):
        img = viz_data['image']
        if img:
            final_img = img.copy()
            for box_info in viz_data['boxes']:
                final_img = draw_bounding_box(final_img, box_info['coords'], box_info['color'], box_info['label'])
            st.image(final_img, caption=f"Page {page_idx + 1} with comparison highlights")


def save_image_file(img: Image.Image, temp_dir: Path, filename: str) -> Path:
    """Save a PIL Image to a temporary file, ensuring PNG format for consistency."""
    # Ensure filename has a .png extension
    base, _ = os.path.splitext(filename)
    out_filename = base + ".png"
    out_path = temp_dir / out_filename
    try:
        img.save(out_path, format="PNG")
        return out_path
    except Exception as e:
        st.error(f"Error saving image {out_filename}: {e}")
        raise # Re-raise to signal failure


def format_time(seconds: float) -> str:
    """Format seconds into minutes and seconds."""
    td = timedelta(seconds=seconds)
    minutes, secs = divmod(td.seconds, 60)
    # For sub-second precision if needed:
    # return f"{minutes} min {secs}.{str(td.microseconds // 1000).zfill(3)} sec"
    return f"{minutes} min {secs:02d} sec"


def process_file_for_options_1_2(
    file_path: Path,
    selected_fields: List[Dict[str, str]],
    openai_client: OpenAI, # Pass the client
    agentic_settings: Settings, # Pass agentic_doc settings
    temp_dir: Path
    ) -> Tuple[Optional[Dict[str, Any]], Optional[Any], Optional[Path], int]:
    """
    Process a single file for Options 1 & 2.
    Converts to image(s), uses agentic-doc for text/chunks, then OpenAI for field extraction.
    Returns: (extracted_data_openai, agentic_doc_result, processed_image_path_or_original_pdf, num_pages)
    The returned path is the first page image for PDFs, or the original image path.
    For PDFs, agentic-doc will parse the whole PDF, but OpenAI extraction is based on text from all pages.
    Visualization will typically be on a per-page basis later.
    """
    if not agentic_imported or not agentic_settings:
        st.error("Agentic Doc is not properly configured. Cannot process file.")
        return None, None, None, 0

    # Agentic-doc can parse PDFs directly, or paths to images.
    # For consistency with drawing and page-based display, if it's a PDF,
    # we might still want to handle it page by page for visualization later,
    # but agentic-doc itself parses the whole document.

    original_file_type = Path(file_path).suffix.lower()
    num_pages_in_doc = 1
    path_for_agentic_parse = str(file_path) # Agentic-doc takes a list of paths

    # This path is used later for loading images for display.
    # If PDF, we'll use original_path and page_idx. If image, just original_path.
    display_source_path = file_path 

    # Agentic-doc parsing
    try:
        # parse_documents expects a list of file paths
        # Settings are now passed globally when client is initialized if vision_agent_api_key is present
        parsed_docs_list: List[AgenticDocument] = parse_documents(
            doc_paths=[path_for_agentic_parse],
            # settings=agentic_settings # Settings are typically configured globally for the client
                                        # or if parse_documents takes it directly, pass it.
                                        # The library usually uses a global or context-based setting.
                                        # Let's assume Settings() on import or via client is enough.
                                        # If VISION_AGENT_API_KEY is set, Settings() should pick it up.
        )

        if not parsed_docs_list:
            st.error(f"Agentic Doc parsing returned no results for {file_path.name}.")
            return None, None, None, 0
        
        agentic_doc_result = parsed_docs_list[0] # Assuming one doc processed
        
        # Collect text and chunks from all pages for OpenAI
        all_text = ""
        all_chunks = []
        if agentic_doc_result and hasattr(agentic_doc_result, 'chunks'):
            all_text = "\n".join(c.text for c in agentic_doc_result.chunks if hasattr(c, 'text'))
            all_chunks = agentic_doc_result.chunks
        
        if not all_text.strip():
            st.warning(f"No text extracted by Agentic Doc from {file_path.name}.")
            # Proceeding to OpenAI with empty text will likely yield no results.
            # return None, agentic_doc_result, display_source_path, num_pages_in_doc


        # Determine number of pages for PDFs for later display
        if original_file_type == '.pdf':
            try:
                with fitz.open(file_path) as fitz_doc:
                    num_pages_in_doc = len(fitz_doc)
            except Exception as e:
                st.warning(f"Could not determine page count for PDF {file_path.name}: {e}")
                num_pages_in_doc = 1 # Default, or use agentic_doc_result.page_info if available

    except Exception as e:
        st.error(f"Error during Agentic Doc processing for {file_path.name}: {e}")
        return None, None, None, 0

    # OpenAI extraction
    if not openai_client: # Check if OpenAI client is available (for Option 1 & 2)
        st.error("OpenAI client not available, cannot perform OpenAI-based field extraction.")
        # For Option 1/2, this is an issue. For a future Option 3, this might be fine.
        # Return agentic_doc results if available, but OpenAI extraction is None
        return None, agentic_doc_result, display_source_path, num_pages_in_doc


    extracted_data_openai = extract_fields_with_openai(openai_client, all_text, selected_fields, all_chunks)
    
    # For display, we need a path to an image. If PDF, it's the original PDF path.
    # If image, it's the image path. `get_document_image` handles page selection.
    return extracted_data_openai, agentic_doc_result, display_source_path, num_pages_in_doc


# --- New functions for Option 3 ---
def process_general_extraction(
    file_path: Path, 
    agentic_settings: Optional[Settings]
    ) -> Optional[List[AgenticDocument]]:
    """
    Processes a file using agentic-doc for general extraction (Option 3).
    Returns the list of parsed Document objects from agentic-doc.
    """
    if not agentic_imported:
        st.error("Agentic Doc package not found.")
        return None
    if not VISION_AGENT_API_KEY: # agentic_settings might be None if key is missing
        st.error("Agentic Doc API Key (VISION_AGENT_API_KEY) is missing.")
        return None
    
    try:
        # If agentic_settings were not initialized due to missing key, this will fail
        # or parse_documents might try to use global settings.
        # Explicitly ensure settings are passed or handled if global.
        # For now, assuming agentic_settings is passed if key was present.
        st.write(f"Processing {file_path.name} with Agentic Doc...")

        # `parse_documents` uses the global settings if `Settings` was initialized with the API key.
        # Or, if it accepts a settings object directly:
        # parsed_docs = parse_documents(doc_paths=[str(file_path)], settings=agentic_settings)
        parsed_docs = parse_documents(doc_paths=[str(file_path)]) # Relies on global settings from initialize_clients

        if not parsed_docs:
            st.error(f"Agentic Doc returned no results for {file_path.name}.")
            return None
        st.success(f"Successfully processed {file_path.name} with Agentic Doc.")
        return parsed_docs
    except Exception as e:
        st.error(f"Error during Agentic Doc general extraction for {file_path.name}: {e}")
        return None

def generate_markdown_from_agentic_doc(doc: AgenticDocument) -> str:
    """Generates a simple Markdown string from an AgenticDocument object."""
    md_content = f"# Document: {doc.file_name or 'Untitled'}\n\n"
    # Check for include_metadata_in_markdown like option in agentic-doc
    # For now, basic chunk text concatenation
    for i, chunk in enumerate(doc.chunks):
        md_content += f"## Chunk {i+1} (Type: {getattr(chunk, 'type', 'N/A')})\n"
        md_content += f"{chunk.text}\n\n"
        if hasattr(chunk, 'grounding') and chunk.grounding:
            md_content += "Grounding:\n"
            for g in chunk.grounding:
                md_content += f"- Page: {g.page_idx}, Box: {g.box}\n"
            md_content += "\n"
    return md_content

def agentic_document_to_dict(doc: AgenticDocument) -> Dict:
    """Converts an AgenticDocument object to a serializable dictionary."""
    if not doc: return {}
    
    chunks_list = []
    if hasattr(doc, 'chunks'):
        for chunk in doc.chunks:
            chunk_dict = {
                "text": getattr(chunk, 'text', None),
                "type": getattr(chunk, 'type', None),
                "id": getattr(chunk, 'id', None),
                "grounding": []
            }
            if hasattr(chunk, 'grounding') and chunk.grounding:
                for g in chunk.grounding:
                    chunk_dict["grounding"].append({
                        "page_idx": getattr(g, 'page_idx', None),
                        "box": getattr(g, 'box', None) # box can be string or object
                    })
            chunks_list.append(chunk_dict)
            
    return {
        "file_name": getattr(doc, 'file_name', None),
        "doc_type": getattr(doc, 'doc_type', None),
        "page_count": getattr(doc, 'page_count', None) if hasattr(doc, 'page_count') else (len(doc.page_info) if hasattr(doc, 'page_info') else None),
        "chunks": chunks_list,
        "page_info": getattr(doc, 'page_info', None), # list of dicts usually
        "errors": getattr(doc, 'errors', None)
    }


def display_general_extraction_output(
    uploaded_file_path: Path, 
    agentic_results: List[AgenticDocument], 
    original_file_bytes: bytes
    ):
    """Displays the output for Option 3."""
    if not agentic_results:
        st.error("No results from Agentic Doc to display.")
        return

    doc_result = agentic_results[0] # Assuming single document processing for Option 3 for now

    left_col, right_col = st.columns(2)

    with left_col:
        st.subheader(f"Visual Grounding: {doc_result.file_name or uploaded_file_path.name}")
        
        # Determine number of pages for display
        num_pages_to_display = 1
        is_pdf = uploaded_file_path.suffix.lower() == ".pdf"
        if is_pdf:
            try:
                with fitz.open(uploaded_file_path) as fitz_doc:
                    num_pages_to_display = len(fitz_doc)
            except Exception:
                # Fallback if page_info exists in agentic_doc result
                if hasattr(doc_result, 'page_info') and doc_result.page_info:
                    num_pages_to_display = len(doc_result.page_info)
                elif hasattr(doc_result, 'page_count') and doc_result.page_count is not None:
                     num_pages_to_display = doc_result.page_count


        page_to_show = 0
        if num_pages_to_display > 1:
            page_to_show = st.selectbox("Select page to view:", range(num_pages_to_display), key="option3_page_select")
        
        # Try using agentic_doc's visualizer if available and suitable
        # viz_dir = None
        # if agentic_visualizer_imported and agentic_imported:
        #     try:
        #         with tempfile.TemporaryDirectory() as viz_temp_dir:
        #             viz_parsed_document(doc_result, output_dir=viz_temp_dir, doc_path=str(uploaded_file_path))
        #             # Now find the image for the selected page in viz_temp_dir
        #             # This requires knowing the naming convention of viz_parsed_document output.
        #             # For simplicity, this path is not fully implemented here.
        #             # st.image(os.path.join(viz_temp_dir, f"page_{page_to_show}.png")) # Example
        #             st.info("Agentic Doc visualization would be shown here if fully configured.")
        #     except Exception as e:
        #         st.warning(f"Could not use agentic_doc visualizer: {e}. Falling back to manual.")
        #         agentic_visualizer_imported = False # Disable if it failed

        # Manual drawing as fallback or primary
        # if not agentic_visualizer_imported or not viz_dir: # Condition to use manual drawing
        current_page_image = get_document_image(str(uploaded_file_path), page_to_show)
        if current_page_image:
            img_with_boxes = current_page_image.copy()
            color_map_op3 = {} # For consistent color per chunk type or ID
            
            if hasattr(doc_result, 'chunks'):
                for chunk_idx, chunk in enumerate(doc_result.chunks):
                    chunk_coords = get_chunk_coordinates(chunk, str(uploaded_file_path))
                    for coord_item in chunk_coords:
                        if coord_item['page_idx'] == page_to_show:
                            # Assign color based on chunk type or index for variety
                            label_text = f"{getattr(chunk, 'type', 'Chunk')} {chunk_idx+1}"
                            color_key = getattr(chunk, 'type', str(chunk_idx)) # Use type or index for color consistency
                            if color_key not in color_map_op3:
                                color_map_op3[color_key] = COLORS[len(color_map_op3) % len(COLORS)]
                            
                            img_with_boxes = draw_bounding_box(
                                img_with_boxes,
                                coord_item['coords'],
                                color=color_map_op3[color_key],
                                label=label_text
                            )
            st.image(img_with_boxes, caption=f"Page {page_to_show + 1} with detected elements", use_column_width=True)
        else:
            st.warning(f"Could not load image for page {page_to_show + 1}.")

    with right_col:
        st.subheader("Extracted Data")
        
        markdown_output = generate_markdown_from_agentic_doc(doc_result)
        json_data_dict = agentic_document_to_dict(doc_result)
        json_output_str = json.dumps(json_data_dict, indent=2)

        tab_titles = ["Markdown Output", "JSON Output", "Downloads"]
        tab_md, tab_json, tab_dl = st.tabs(tab_titles)

        with tab_md:
            st.markdown("#### Markdown View")
            st.text_area("Markdown", value=markdown_output, height=400, key="option3_md_disp")
        
        with tab_json:
            st.markdown("#### JSON View")
            st.text_area("JSON", value=json_output_str, height=400, key="option3_json_disp")

        with tab_dl:
            st.markdown("#### Download Files")
            st.download_button(
                label="📥 Download Original File",
                data=original_file_bytes,
                file_name=uploaded_file_path.name,
                mime=None # Let browser infer
            )
            st.download_button(
                label="📥 Download Markdown (.md)",
                data=markdown_output.encode('utf-8'),
                file_name=f"{uploaded_file_path.stem}_extracted.md",
                mime="text/markdown"
            )
            st.download_button(
                label="📥 Download JSON (.json)",
                data=json_output_str.encode('utf-8'),
                file_name=f"{uploaded_file_path.stem}_extracted.json",
                mime="application/json"
            )
# --- End of Option 3 functions ---


# --- Placeholder functions for Option 4 ---
def handle_option4_uploads_and_display():
    st.subheader("Option 4: Upload Processed Files & View with Mouse-over")
    st.info("This feature allows you to upload a previously processed document (image/PDF), its JSON data (with bounding boxes), and optionally its Markdown representation, to visualize the bounding boxes.")
    
    original_file_op4 = st.file_uploader("1. Upload Original Document (Image or PDF)", type=['pdf', 'png', 'jpg', 'jpeg'], key="op4_original")
    json_file_op4 = st.file_uploader("2. Upload JSON Data File (.json from Option 3)", type=['json'], key="op4_json")
    # md_file_op4 = st.file_uploader("3. (Optional) Upload Markdown File (.md from Option 3)", type=['md'], key="op4_md")

    if original_file_op4 and json_file_op4:
        try:
            json_data = json.load(io.TextIOWrapper(json_file_op4, encoding='utf-8'))
            # Ensure json_data has the expected structure (e.g., 'file_name', 'chunks' with 'grounding')
            if not isinstance(json_data, dict) or 'chunks' not in json_data:
                st.error("Invalid JSON format. Expected a dictionary with a 'chunks' key.")
                return

            # Save original file to a temp path to handle it like other options
            with tempfile.NamedTemporaryFile(delete=False, suffix=Path(original_file_op4.name).suffix) as tmp_orig:
                tmp_orig.write(original_file_op4.getvalue())
                original_file_path_op4 = Path(tmp_orig.name)
            
            st.markdown("---")
            st.subheader(f"Document: {json_data.get('file_name', original_file_op4.name)}")

            # Determine number of pages for display
            num_pages_op4 = 1
            is_pdf_op4 = original_file_path_op4.suffix.lower() == ".pdf"
            if is_pdf_op4:
                try:
                    with fitz.open(original_file_path_op4) as fitz_doc:
                        num_pages_op4 = len(fitz_doc)
                except Exception:
                    num_pages_op4 = json_data.get('page_count', 1)
            
            page_to_show_op4 = 0
            if num_pages_op4 > 1:
                page_to_show_op4 = st.selectbox("Select page to view:", range(num_pages_op4), key="option4_page_select")

            current_page_image_op4 = get_document_image(str(original_file_path_op4), page_to_show_op4)

            if current_page_image_op4:
                img_with_boxes_op4 = current_page_image_op4.copy()
                
                # For mouse-over, we need a more interactive component.
                # For now, we'll draw all boxes from JSON.
                # A simple "highlight on click" or "list elements" could be a step.
                
                # st.write("Interactive mouse-over highlighting is an advanced feature.")
                # st.write("Below is the image with all bounding boxes from the JSON file:")

                chunks_on_page = []
                color_map_op4 = {} # For consistent color per chunk type or ID

                for chunk_idx, chunk_data in enumerate(json_data.get('chunks', [])):
                    if 'grounding' in chunk_data:
                        for g_data in chunk_data['grounding']:
                            if g_data.get('page_idx') == page_to_show_op4 and g_data.get('box'):
                                coords = None
                                if isinstance(g_data['box'], str):
                                    coords = parse_box_string(g_data['box'])
                                elif isinstance(g_data['box'], list) and len(g_data['box']) == 4: # Assuming list format [l,t,r,b]
                                    coords = g_data['box']
                                elif isinstance(g_data['box'], dict) and all(k in g_data['box'] for k in ['l','t','r','b']):
                                    coords = [g_data['box']['l'], g_data['box']['t'], g_data['box']['r'], g_data['box']['b']]


                                if coords:
                                    label_text = f"{chunk_data.get('type', 'Chunk')} {chunk_idx+1}"
                                    color_key = chunk_data.get('type', str(chunk_idx))
                                    if color_key not in color_map_op4:
                                        color_map_op4[color_key] = COLORS[len(color_map_op4) % len(COLORS)]
                                    
                                    img_with_boxes_op4 = draw_bounding_box(
                                        img_with_boxes_op4,
                                        coords,
                                        color=color_map_op4[color_key],
                                        label=label_text
                                    )
                                    # Store info for potential simple interaction
                                    chunks_on_page.append({
                                        "text": chunk_data.get('text', 'N/A'),
                                        "type": chunk_data.get('type', 'N/A'),
                                        "id": chunk_data.get('id', chunk_idx),
                                        "box_info": {"coords": coords, "color": color_map_op4[color_key], "label": label_text}
                                    })
                
                st.image(img_with_boxes_op4, caption=f"Page {page_to_show_op4 + 1} with elements from JSON", use_column_width=True)

                if chunks_on_page:
                    st.markdown("#### Extracted Elements on this Page (from JSON):")
                    # This could be made interactive, e.g., clicking an item highlights it.
                    # For simplicity, just listing them.
                    for item in chunks_on_page:
                        exp = st.expander(f"{item['label']}: {item['text'][:100]}...")
                        exp.write(f"Text: {item['text']}")
                        exp.write(f"Type: {item['type']}")
                        # exp.write(f"Box Coords: {item['box_info']['coords']}")
                
            else:
                st.warning(f"Could not load image for page {page_to_show_op4 + 1} of {original_file_op4.name}")

            # Clean up temp file
            if original_file_path_op4.exists():
                try:
                    os.remove(original_file_path_op4)
                except Exception as e_clean:
                    st.warning(f"Could not delete temporary file {original_file_path_op4}: {e_clean}")
                    
        except json.JSONDecodeError:
            st.error("Invalid JSON file. Could not parse.")
        except Exception as e:
            st.error(f"An error occurred processing files for Option 4: {e}")
            # Clean up temp file in case of other errors too
            if 'original_file_path_op4' in locals() and original_file_path_op4.exists():
                try:
                    os.remove(original_file_path_op4)
                except Exception: pass
# --- End of Option 4 functions ---


def main():
    st.set_page_config(layout="wide", page_title="Document Field Extractor")
    
    # Initialize clients (OpenAI and Agentic Doc settings)
    # Pass the chosen option to initialize_clients if it needs to behave differently
    if 'process_option' not in st.session_state:
        st.session_state.process_option = "Option 1: Extract desired field information only" # Default

    openai_client, agentic_settings = initialize_clients()

    st.markdown("# 📄 Document Field Extractor")
    st.markdown("## Processing Options")
    
    process_option = st.radio(
        "Choose processing option:",
        ["Option 1: Extract desired field information only",
         "Option 2: Extract and Compare fields against reference document",
         "Option 3: Extract General Data with Visual Grounding (Agentic Doc)",
         "Option 4: Upload and View Processed Files (with Bounding Boxes)"],
        key="process_option" # Persist choice
    )

    # UI structure common to Option 1 & 2
    if process_option.startswith("Option 1") or process_option.startswith("Option 2"):
        top_container = st.container()
        bottom_container = st.container()

        with top_container:
            if "Option 1" in process_option:
                st.markdown("## Files to Check & Fields for Extraction (Option 1)")
                source_selected_fields = manage_fields(top_container, "source") # Manages source_extraction_fields
                source_files = st.file_uploader("Upload documents to check (PDF, PNG, JPG, JPEG)", 
                                              type=['pdf', 'png', 'jpg', 'jpeg'], 
                                              accept_multiple_files=True,
                                              key="source_files_op1")
                reference_files = [] # No reference files for pure Option 1
            
            else: # Option 2
                left_col, right_col = st.columns(2)
                with left_col:
                    st.markdown("## Files to Check (Option 2)")
                    st.info("For Option 2, 'Files to Check' will use the field set defined under 'Reference Files'.")
                    # For Option 2, source files use reference fields for extraction
                    # The manage_fields for "reference" will define the fields used for both.
                    # We don't call manage_fields here for source, but will use reference_selected_fields.
                    source_files = st.file_uploader("Upload documents to check (PDF, PNG, JPG, JPEG)", 
                                                  type=['pdf', 'png', 'jpg', 'jpeg'], 
                                                  accept_multiple_files=True,
                                                  key="source_files_op2")
                with right_col:
                    st.markdown("## Reference Files & Fields for Comparison (Option 2)")
                    reference_selected_fields = manage_fields(right_col, "reference") # Manages reference_extraction_fields
                    reference_files = st.file_uploader("Upload reference documents (PDF, PNG, JPG, JPEG)", 
                                                     type=['pdf', 'png', 'jpg', 'jpeg'], 
                                                     accept_multiple_files=True, # Typically one ref, but allow multiple
                                                     key="reference_files_op2")

        with bottom_container:
            if "Option 1" in process_option:
                st.markdown("## Visualization Options (Option 1)")
                vis_option = st.radio(
                    "Choose visualization style:",
                    ["Show each field with corresponding reference image",
                     "Show multiple color-coded bounding boxes per document"],
                    key="vis_option_op1"
                )
            # For Option 2, visualization is fixed to comparison view, so no vis_option needed here.
            
            st.markdown("---")
            if st.button("Process Documents", key="process_button_op1_2"):
                # Validation for Option 1 & 2
                if not source_files:
                    st.error("Please upload at least one document to check.")
                    st.stop()
                if "Option 2" in process_option and not reference_files:
                    st.error("Please upload at least one reference document for comparison (Option 2).")
                    st.stop()
                if ("Option 1" in process_option or "Option 2" in process_option) and not openai_client:
                    st.error("OpenAI API Key is missing or client failed to initialize. Needed for Option 1 & 2.")
                    st.stop()
                if not agentic_settings: # Check if agentic_settings initialized (needs API key)
                    st.error("Agentic Doc settings not initialized (likely missing VISION_AGENT_API_KEY). This is required for processing.")
                    st.stop()


                with tempfile.TemporaryDirectory() as td:
                    temp_dir = Path(td)
                    overall_start_time = time.time()
                    total_kb_processed = 0
                    
                    progress_bar = st.progress(0)
                    progress_text = st.empty()

                    if "Option 1" in process_option:
                        st.subheader("Extraction Results (Option 1)")
                        num_source_files = len(source_files)
                        for file_idx, uploaded_file in enumerate(source_files):
                            progress_text.text(f"Processing file {file_idx + 1}/{num_source_files}: {uploaded_file.name}")
                            
                            original_path = temp_dir / f"source_{file_idx}_{uploaded_file.name}"
                            original_path.write_bytes(uploaded_file.getvalue())
                            total_kb_processed += original_path.stat().st_size / 1024
                            
                            file_start_time = time.time()
                            extracted_data, agentic_doc_obj, display_path, num_pages = process_file_for_options_1_2(
                                original_path, source_selected_fields, openai_client, agentic_settings, temp_dir
                            )
                            file_processing_time = time.time() - file_start_time
                            
                            st.success(f"✅ File {uploaded_file.name} ({num_pages} pg(s)) processed in {format_time(file_processing_time)}")
                            # Add per KB timing if useful

                            if extracted_data and display_path:
                                # Display logic for Option 1
                                if "each field" in vis_option: # Show each field with corresponding reference image
                                    for field_name, field_data_val in extracted_data.items():
                                        if field_data_val.get('value') and field_data_val.get('matching_chunks'):
                                            st.markdown(f"### {field_name}: {field_data_val['value']}")
                                            # Find a color for this field
                                            color_for_field = COLORS[list(extracted_data.keys()).index(field_name) % len(COLORS)]
                                            display_chunk_evidence(
                                                field_data_val['matching_chunks'][0], # Display first matching chunk
                                                field_name,
                                                str(display_path), # Path to the (potentially multi-page) PDF or image
                                                color=color_for_field
                                            )
                                else: # Show multiple color-coded bounding boxes per document
                                    # This needs to handle multi-page PDFs by iterating through pages
                                    display_unified_evidence(extracted_data, str(display_path), agentic_doc_obj)
                            elif not extracted_data and agentic_doc_obj : # Parsing ok, OpenAI failed or no fields
                                st.warning(f"Agentic Doc parsed {uploaded_file.name}, but no fields were extracted by OpenAI.")
                            else:
                                st.error(f"Failed to process or extract data from {uploaded_file.name}.")
                            
                            progress_bar.progress((file_idx + 1) / num_source_files)
                            if file_idx < num_source_files - 1: st.markdown("---")
                    
                    elif "Option 2" in process_option:
                        st.subheader("Comparison Results (Option 2)")
                        if not reference_files: # Should be caught earlier, but double check
                            st.error("Reference file needed for Option 2.")
                            st.stop()

                        ref_file_obj = reference_files[0] # Assuming one reference for now
                        ref_path_temp = temp_dir / f"ref_{ref_file_obj.name}"
                        ref_path_temp.write_bytes(ref_file_obj.getvalue())
                        total_kb_processed += ref_path_temp.stat().st_size / 1024

                        progress_text.text(f"Processing reference file: {ref_file_obj.name}")
                        progress_bar.progress(0.1)

                        ref_extracted_data, ref_agentic_doc, _, ref_num_pages = process_file_for_options_1_2(
                            ref_path_temp, reference_selected_fields, openai_client, agentic_settings, temp_dir
                        )
                        
                        if not ref_extracted_data:
                            st.error(f"Could not process reference file {ref_file_obj.name} or extract fields. Cannot proceed with comparison.")
                            st.stop()
                        st.success(f"Reference file {ref_file_obj.name} ({ref_num_pages} pg(s)) processed.")
                        progress_bar.progress(0.2)

                        num_source_files = len(source_files)
                        for file_idx, uploaded_file in enumerate(source_files):
                            progress_text.text(f"Comparing source file {file_idx + 1}/{num_source_files}: {uploaded_file.name}")
                            
                            source_path_temp = temp_dir / f"source_comp_{file_idx}_{uploaded_file.name}"
                            source_path_temp.write_bytes(uploaded_file.getvalue())
                            total_kb_processed += source_path_temp.stat().st_size / 1024

                            file_start_time = time.time()
                            src_extracted_data, src_agentic_doc, src_display_path, src_num_pages = process_file_for_options_1_2(
                                source_path_temp, reference_selected_fields, openai_client, agentic_settings, temp_dir # Use ref_fields for source
                            )
                            file_processing_time = time.time() - file_start_time

                            st.success(f"✅ Source file {uploaded_file.name} ({src_num_pages} pg(s)) processed for comparison in {format_time(file_processing_time)}")

                            if src_extracted_data and src_display_path:
                                st.markdown(f"### Comparison: {uploaded_file.name} vs {ref_file_obj.name}")
                                display_comparison_evidence(src_extracted_data, ref_extracted_data, str(src_display_path), src_agentic_doc)
                            else:
                                st.error(f"Failed to process or extract data from source file {uploaded_file.name} for comparison.")
                            
                            current_progress = 0.2 + (0.8 * (file_idx + 1) / num_source_files)
                            progress_bar.progress(current_progress)
                            if file_idx < num_source_files - 1: st.markdown("---")

                    # Overall timing for Option 1 & 2
                    progress_bar.progress(1.0)
                    progress_text.text("All processing complete!")
                    total_processing_time = time.time() - overall_start_time
                    
                    st.markdown("## Processing Time Metrics (Option 1 & 2)")
                    col_m1, col_m2, col_m3 = st.columns(3)
                    col_m1.metric("Total Processing Time", format_time(total_processing_time))
                    col_m2.metric("Total File Size Processed", f"{total_kb_processed:.2f} KB")
                    if total_kb_processed > 0:
                        col_m3.metric("Avg Time per KB", format_time(total_processing_time / total_kb_processed))
                    else:
                        col_m3.metric("Avg Time per KB", "N/A")


    elif process_option.startswith("Option 3"):
        st.header("Option 3: Extract General Data with Visual Grounding")
        if not agentic_imported or not agentic_settings : # VISION_AGENT_API_KEY needs to be set for settings
             st.error("Agentic Doc is not available or not configured (check VISION_AGENT_API_KEY). Option 3 cannot proceed.")
             st.stop()

        uploaded_file_op3 = st.file_uploader(
            "Upload a document (PDF, PNG, JPG, JPEG) for general extraction",
            type=['pdf', 'png', 'jpg', 'jpeg'],
            key="op3_uploader"
        )

        if uploaded_file_op3:
            if st.button("Process Document (Option 3)", key="op3_process"):
                with tempfile.TemporaryDirectory() as td_op3:
                    temp_dir_op3 = Path(td_op3)
                    # Save uploaded file to temp dir
                    file_path_op3 = temp_dir_op3 / uploaded_file_op3.name
                    original_bytes = uploaded_file_op3.getvalue()
                    file_path_op3.write_bytes(original_bytes)

                    with st.spinner(f"Processing {uploaded_file_op3.name} with Agentic Doc..."):
                        agentic_results = process_general_extraction(file_path_op3, agentic_settings)
                    
                    if agentic_results:
                        display_general_extraction_output(file_path_op3, agentic_results, original_bytes)
                    else:
                        st.error("Failed to get results from Agentic Doc for Option 3.")
    
    elif process_option.startswith("Option 4"):
        # Call the handler for Option 4 UI and logic
        handle_option4_uploads_and_display()


if __name__ == "__main__":
    main()
