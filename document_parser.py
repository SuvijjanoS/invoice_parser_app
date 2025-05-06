import os
from pathlib import Path
import json
import tempfile
import re

import streamlit as st
from dotenv import load_dotenv
from openai import OpenAI
from openai.error import InvalidRequestError, OpenAIError
from agentic_doc.parse import parse_documents
from agentic_doc.config import Settings
from PIL import Image, ImageDraw
import fitz  # PyMuPDF for PDF handling
from typing import List, Dict, Any, Tuple

# Load local .env for development; secrets will override in cloud
load_dotenv()

# Fetch API keys: prefer Streamlit secrets, fallback to environment
OPENAI_API_KEY = st.secrets.get("OPENAI_API_KEY") or os.getenv("OPENAI_API_KEY")
VISION_AGENT_API_KEY = st.secrets.get("VISION_AGENT_API_KEY") or os.getenv("VISION_AGENT_API_KEY")

if not OPENAI_API_KEY or not VISION_AGENT_API_KEY:
    st.error(
        "🔑 API keys missing!\n\n"
        "Please go to your Streamlit Cloud app’s Settings → Secrets, and add:\n"
        "  • OPENAI_API_KEY = <your OpenAI key>\n"
        "  • VISION_AGENT_API_KEY = <your Agentic Doc key>\n"
    )
    st.stop()

# Initialize clients
client = OpenAI(api_key=OPENAI_API_KEY)
settings = Settings(
    vision_agent_api_key=VISION_AGENT_API_KEY,
    batch_size=4,
    max_workers=5,
    max_retries=100,
    max_retry_wait_time=60,
    retry_logging_style="log_msg"
)

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


def extract_fields_with_openai(text: str, fields: List[Dict[str, str]], chunks: List[Any] = None) -> Dict[str, Any]:
    """Use OpenAI to extract specific fields from text and track source chunks."""
    field_instructions = "
".join([
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
    # Debug print: before sending to OpenAI
    st.write("🔎 Sending prompt to OpenAI…")
    try:
        response = client.chat.completions.create(
            model="gpt-4",
            messages=[
                {"role": "system", "content": "You are a specialized invoice parser that accurately extracts fields."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.1
        )
    except InvalidRequestError as e:
        status = getattr(e, 'http_status', None)
        code = getattr(e, 'code', None)
        if status == 402 or code == 'insufficient_quota':
            st.error("🚫 You’ve run out of API credits. Please add more credits to continue.")
            st.stop()
        else:
            st.error(f"Invalid request to OpenAI API: {e}")
            st.stop()
    except OpenAIError as e:
        st.error(f"OpenAI API error: {e}")
        st.stop()
    # Debug print: after receiving from OpenAI
    st.write("✅ OpenAI replied, processing response…")

    try:
        extracted_data = json.loads(response.choices[0].message.content)
        
        # If chunks are provided, find matching chunks for each extracted field
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
        return {field['name']: {'value': None} for field in fields}(text: str, fields: List[Dict[str, str]], chunks: List[Any] = None) -> Dict[str, Any]:
    instr = "\n".join([f"- {f['name']}: {f['description']}" for f in fields])
    prompt = f"""
Extract the following fields from the invoice text below.
Fields:
{instr}
Text:
{text}
Return JSON: {{"field_name":{{"value":"..."}}}}, null if not found.
"""
    try:
        response = client.chat.completions.create(
            model="gpt-4",
            messages=[{"role":"system","content":"You are a specialized invoice parser."},
                      {"role":"user","content":prompt}],
            temperature=0.1
        )
    except InvalidRequestError as e:
        st.error("🚫 You’ve run out of API credits. Please add more credits to continue.")
        st.stop()
    except OpenAIError as e:
        st.error(f"OpenAI API error: {e}")
        st.stop()

    try:
        data = json.loads(response.choices[0].message.content)
        if chunks:
            for name, fd in data.items():
                if fd.get('value'):
                    val = fd['value'].strip().lower()
                    matches = [c for c in chunks if val in c.text.lower()]
                    fd['matching_chunks'] = matches
        return data
    except json.JSONDecodeError:
        return {f['name']:{'value':None} for f in fields}


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
        coords = {k:float(v) for part in parts for k,v in [part.split('=')]}
        return [coords[k] for k in ('l','t','r','b')]
    except:
        return None


def draw_bounding_box(img: Image.Image, box: List[float]) -> Image.Image:
    out = img.copy(); d=ImageDraw.Draw(out)
    w,h = out.size; x0,y0,x1,y1 = [int(c*dim) for c,dim in zip(box,[w,h,w,h])]
    d.rectangle([x0,y0,x1,y1], outline=(255,0,0), width=2)
    return out


def display_chunk_evidence(chunk, name: str, path: str):
    if hasattr(chunk,'grounding'):
        for g in chunk.grounding:
            box = getattr(g,'box',None)
            coords = parse_box_string(box) if isinstance(box,str) else getattr(box,'l',None) and [box.l,box.t,box.r,box.b]
            if coords:
                img = get_document_image(path, getattr(g,'page_idx',0))
                if img: st.image(draw_bounding_box(img, coords))


def main():
    st.set_page_config(layout="wide")
    st.title("📄 Invoice Field Extractor")
    st.write("Upload an invoice to extract fields.")

    selected = manage_fields()
    translate = st.checkbox("Translate Thai→English", value=False)
    up = st.file_uploader("Upload document", type=['pdf','png','jpg','jpeg'])

    if up and selected and st.button("Extract Fields"):
        with st.spinner("Processing..."):
            with tempfile.TemporaryDirectory() as td:
                path = Path(td)/f"in{up.name}"
                path.write_bytes(up.getvalue())
                results = parse_documents([str(path)])
                if not results: st.error("Parse returned nothing"); return
                doc = results[0]
                text = "\n".join(c.text for c in doc.chunks)
                if translate:
                    text = text  # translation logic here if needed
                data = extract_fields_with_openai(text, selected, doc.chunks)
                for nm, fd in data.items():
                    if fd.get('value'):
                        c1,c2 = st.columns([1,3])
                        with c1: st.markdown(f"**{nm}:** {fd['value']}")
                        with c2:
                            if fd.get('matching_chunks'):
                                display_chunk_evidence(fd['matching_chunks'][0], nm, str(path))

if __name__ == "__main__":
    main()
