import streamlit as st
import pandas as pd
from PIL import Image
import fitz  # PyMuPDF
import os
import tempfile
import time
import logging
from combined import extract_data_from_image
import google.generativeai as genai
from dotenv import load_dotenv
from typing import List, Optional
import openai
from old_code.gpt import extract_data_from_image_gpt

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# --- Page Configuration ---
st.set_page_config(
    page_title="Document Data Extractor",
    page_icon="📄",
    layout="wide"
)

# --- Helper Functions ---
def pdf_to_images(pdf_bytes: bytes) -> List[Image.Image]:
    """Convert PDF bytes to a list of PIL Images with error handling."""
    try:
        pdf_doc = fitz.open(stream=pdf_bytes, filetype="pdf")
        images = []
        
        if len(pdf_doc) == 0:
            st.error("PDF appears to be empty or corrupted.")
            return []
            
        for page_num in range(len(pdf_doc)):
            try:
                page = pdf_doc.load_page(page_num)
                pix = page.get_pixmap(dpi=200)
                img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
                images.append(img)
            except Exception as e:
                st.warning(f"Error processing page {page_num + 1}: {e}")
                continue
                
        pdf_doc.close()
        return images
        
    except Exception as e:
        st.error(f"Error processing PDF: {e}")
        return []

@st.cache_data
def convert_df_to_csv(df: pd.DataFrame) -> bytes:
    """Caches the conversion of a DataFrame to CSV."""
    try:
        return df.to_csv(index=False).encode('utf-8')
    except Exception as e:
        st.error(f"Error converting to CSV: {e}")
        return b""

def safe_delete_temp_file(file_path: str, max_attempts: int = 5) -> bool:
    """Safely delete a temporary file with retry logic for Windows."""
    for attempt in range(max_attempts):
        try:
            if os.path.exists(file_path):
                os.unlink(file_path)
            return True
        except PermissionError:
            time.sleep(0.1 * (attempt + 1))  # Increasing delay
        except FileNotFoundError:
            return True  # File already deleted
        except Exception as e:
            logger.warning(f"Unexpected error deleting file: {e}")
            break
    return False

def validate_uploaded_file(uploaded_file) -> bool:
    """Validate uploaded file."""
    if uploaded_file is None:
        return False
        
    # Check file size (limit to 10MB)
    if uploaded_file.size > 10 * 1024 * 1024:
        st.error("File size too large. Please upload files smaller than 10MB.")
        return False
        
    # Check file type
    allowed_types = ['image/jpeg', 'image/png', 'image/jpg', 'application/pdf']
    if uploaded_file.type not in allowed_types:
        st.error("Unsupported file type. Please upload JPG, PNG, or PDF files.")
        return False
        
    return True

def process_single_image(image: Image.Image, page_key: int, ai_model: str = "gemini") -> Optional[dict]:
    """Process a single image and return extracted data."""
    tmp_file = None
    tmp_file_path = None
    
    try:
        # Create temporary file with delete=False to control deletion manually
        tmp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".png")
        tmp_file_path = tmp_file.name
        tmp_file.close()  # Close the file handle
        
        # Save image to temp file
        image.save(tmp_file_path)
        
        # Process with selected AI model
        if ai_model == "gpt4":
            extracted_data = extract_data_from_image_gpt(tmp_file_path)
        else:  # Default to Gemini
            extracted_data = extract_data_from_image(tmp_file_path)
        
        if extracted_data and 'document_content' in extracted_data:
            return extracted_data
        else:
            return None
            
    except Exception as e:
        st.error(f"Error processing image: {e}")
        return None
        
    finally:
        # Safely delete the temporary file
        if tmp_file_path and not safe_delete_temp_file(tmp_file_path):
            logger.warning(f"Could not delete temporary file: {tmp_file_path}")

# --- Initialize Session State ---
def initialize_session_state():
    """Initialize session state variables."""
    if 'all_entries' not in st.session_state:
        st.session_state.all_entries = []
    if 'processed_pages' not in st.session_state:
        st.session_state.processed_pages = {}
    if 'processing_errors' not in st.session_state:
        st.session_state.processing_errors = {}
    if 'selected_ai_model' not in st.session_state:
        st.session_state.selected_ai_model = "gemini"

initialize_session_state()

# --- Sidebar for Controls ---
with st.sidebar:
    st.title("📄 Document Extractor")
    st.write("Enhanced with West Bengal geographical knowledge")
    st.write("---")
    
    # --- AI Model Selection ---
    st.subheader("🤖 AI Model Selection")
    ai_model = st.radio(
        "Choose AI Model:",
        options=["gemini", "gpt4"],
        format_func=lambda x: "Google Gemini" if x == "gemini" else "OpenAI GPT-4",
        key="ai_model_selector"
    )
    st.session_state.selected_ai_model = ai_model
    
    # --- API Key Configuration ---
    try:
        load_dotenv()
        
        if ai_model == "gemini":
            api_key = os.getenv("GOOGLE_API_KEY")
            if api_key:
                try:
                    genai.configure(api_key=api_key)
                    st.success("✅ Gemini API Key loaded successfully!")
                except Exception as e:
                    st.error(f"❌ Gemini API Key configuration failed: {e}")
                    st.stop()
            else:
                st.error("❌ `GOOGLE_API_KEY` not found in your .env file.")
                st.info("Please create a .env file in the root directory and add your key.")
                st.stop()
        else:  # GPT-4
            openai_api_key = os.getenv("OPENAI_API_KEY")
            if openai_api_key:
                try:
                    openai.api_key = openai_api_key
                    st.success("✅ OpenAI API Key loaded successfully!")
                except Exception as e:
                    st.error(f"❌ OpenAI API Key configuration failed: {e}")
                    st.stop()
            else:
                st.error("❌ `OPENAI_API_KEY` not found in your .env file.")
                st.info("Please add your OpenAI API key to the .env file.")
                st.stop()
                
    except Exception as e:
        st.error(f"❌ Error loading environment: {e}")
        st.stop()

    # --- File Uploader ---
    uploaded_file = st.file_uploader(
        "Upload a document",
        type=['jpg', 'png', 'jpeg', 'pdf'],
        help="Upload an image or a multi-page PDF for data extraction."
    )
    
    # Validate uploaded file
    if uploaded_file and not validate_uploaded_file(uploaded_file):
        st.stop()
    
    # --- Page Navigation for PDFs ---
    if uploaded_file and uploaded_file.type == "application/pdf":
        if 'pdf_images' not in st.session_state or st.session_state.get('file_name') != uploaded_file.name:
            with st.spinner("Processing PDF..."):
                st.session_state.pdf_images = pdf_to_images(uploaded_file.getvalue())
                st.session_state.file_name = uploaded_file.name
                st.session_state.processed_pages = {} # Reset for new file
                st.session_state.all_entries = []
                st.session_state.processing_errors = {}

        if st.session_state.pdf_images:
            st.success(f"📄 {len(st.session_state.pdf_images)} pages loaded successfully")
            page_options = [f"Page {i+1}" for i in range(len(st.session_state.pdf_images))]
            st.session_state.selected_page_index = st.selectbox(
                "Select a page to view", 
                range(len(page_options)), 
                format_func=lambda x: page_options[x]
            )
        else:
            st.error("❌ No pages could be extracted from the PDF.")

    # --- Processing Statistics ---
    if uploaded_file:
        if uploaded_file.type == "application/pdf" and 'pdf_images' in st.session_state:
            total_pages = len(st.session_state.pdf_images)
            processed_pages = len(st.session_state.processed_pages)
            st.write("---")
            st.metric("Processing Progress", f"{processed_pages}/{total_pages}")
            
            if st.session_state.processing_errors:
                st.error(f"❌ {len(st.session_state.processing_errors)} pages had errors")

# --- Main Panel ---
st.header("📋 Document Review and Extracted Data")
st.write(f"Specialized for West Bengal registration documents using {('OpenAI GPT-4' if ai_model == 'gpt4' else 'Google Gemini')}")

if not uploaded_file:
    st.info("👆 Please upload a document using the sidebar to begin.")
    
    # Display feature information
    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown("""
        **🎯 Features:**
        - INDEX I & II document support
        - Handwriting interpretation
        - Ditto mark resolution
        """)
    with col2:
        st.markdown("""
        **🗺️ Geographic Intelligence:**
        - West Bengal 1960s knowledge
        - Auto-correct place names
        - District/PS recognition
        """)
    with col3:
        st.markdown("""
        **🔧 Robust Processing:**
        - Multi-page PDF support
        - Error recovery
        - Data validation
        """)
    st.stop()

# --- Two-Column Layout ---
col1, col2 = st.columns([1, 1])

# Determine the image to display
image_to_display = None
page_key = 0

try:
    if uploaded_file.type != "application/pdf":
        image_to_display = Image.open(uploaded_file)
        page_key = 0
    elif 'selected_page_index' in st.session_state and st.session_state.pdf_images:
        page_key = st.session_state.selected_page_index
        image_to_display = st.session_state.pdf_images[page_key]
except Exception as e:
    st.error(f"Error loading image: {e}")

# --- Column 1: Document Viewer ---
with col1:
    st.subheader("📄 Document Viewer")
    if image_to_display:
        try:
            st.image(image_to_display, use_container_width=True)
            
            # Display image info
            st.caption(f"Image size: {image_to_display.size}")
            
        except Exception as e:
            st.error(f"Error displaying image: {e}")
    else:
        st.warning("No image to display")

# --- Column 2: Extracted Data ---
with col2:
    st.subheader("🤖 Extraction Results")
    
    # Display "Process" button if the page hasn't been processed yet
    if image_to_display and page_key not in st.session_state.processed_pages:
        model_name = "GPT-4" if ai_model == "gpt4" else "Gemini"
        if st.button(f"🚀 Process Page {page_key + 1} with {model_name}", key=f"process_{page_key}", type="primary"):
            with st.spinner(f"🤖 {model_name} is analyzing the page... This may take 30-60 seconds."):
                progress_bar = st.progress(0)
                progress_bar.progress(25)
                
                extracted_data = process_single_image(image_to_display, page_key, ai_model)
                progress_bar.progress(75)
                
                if extracted_data and 'document_content' in extracted_data:
                    entries = extracted_data['document_content'].get('entries', [])
                    st.session_state.processed_pages[page_key] = entries
                    
                    # Add metadata to session state
                    if 'metadata' not in st.session_state:
                        st.session_state.metadata = {}
                    st.session_state.metadata[page_key] = extracted_data.get('document_metadata', {})
                    
                    # Add to all_entries, ensuring no duplicates
                    for entry in entries:
                        if entry not in st.session_state.all_entries:
                            st.session_state.all_entries.append(entry)
                    
                    progress_bar.progress(100)
                    st.success(f"✅ Successfully processed {len(entries)} entries with {model_name}!")
                    time.sleep(0.5)  # Brief pause for user feedback
                    st.rerun()
                else:
                    st.session_state.processed_pages[page_key] = []
                    st.session_state.processing_errors[page_key] = "No data extracted"
                    progress_bar.progress(100)
                    st.warning("⚠️ No data could be extracted from this page.")

    # Display data if it has been processed
    if page_key in st.session_state.processed_pages:
        page_entries = st.session_state.processed_pages[page_key]
        
        if page_entries:
            # Display metadata if available
            if 'metadata' in st.session_state and page_key in st.session_state.metadata:
                metadata = st.session_state.metadata[page_key]
                col_a, col_b = st.columns(2)
                with col_a:
                    st.metric("Document Type", metadata.get('index_type', 'Unknown'))
                with col_b:
                    st.metric("Confidence", metadata.get('extraction_confidence', 'Unknown'))
            
            # Flatten data for display
            flat_data = []
            for entry in page_entries:
                row = {"serial_number": entry.get("serial_number")}
                row.update(entry.get("entry_details", {}))
                row["confidence_notes"] = ", ".join(entry.get("confidence_notes", []))
                flat_data.append(row)
            
            df_page = pd.DataFrame(flat_data)
            st.dataframe(df_page, use_container_width=True)
            st.caption(f"📊 {len(page_entries)} entries found on this page")
            
        else:
            if page_key in st.session_state.processing_errors:
                st.error(f"❌ {st.session_state.processing_errors[page_key]}")
            else:
                st.info("ℹ️ No entries were found on this page.")

# --- Aggregated Results and Download ---
if st.session_state.all_entries:
    st.write("---")
    st.subheader("📊 Aggregated Data from All Processed Pages")
    
    try:
        agg_flat_data = []
        for entry in st.session_state.all_entries:
            row = {"serial_number": entry.get("serial_number")}
            row.update(entry.get("entry_details", {}))
            row["confidence_notes"] = ", ".join(entry.get("confidence_notes", []))
            agg_flat_data.append(row)
        
        df_all = pd.DataFrame(agg_flat_data)
        
        # Display summary metrics
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Total Entries", len(df_all))
        with col2:
            st.metric("Total Pages Processed", len(st.session_state.processed_pages))
        with col3:
            entries_with_notes = len([row for row in agg_flat_data if row.get("confidence_notes")])
            st.metric("Entries with Notes", entries_with_notes)
        
        st.dataframe(df_all, use_container_width=True)
        
        # Download options
        csv_data = convert_df_to_csv(df_all)
        if csv_data:
            col1, col2 = st.columns(2)
            with col1:
                st.download_button(
                   label="📥 Download as CSV",
                   data=csv_data,
                   file_name='extracted_data.csv',
                   mime='text/csv',
                )
            with col2:
                # JSON download
                import json
                json_data = json.dumps(st.session_state.all_entries, indent=2, ensure_ascii=False)
                st.download_button(
                    label="📥 Download as JSON",
                    data=json_data,
                    file_name='extracted_data.json',
                    mime='application/json',
                )
        
    except Exception as e:
        st.error(f"Error displaying aggregated data: {e}")

# --- Footer ---
st.write("---")
st.caption(f"🔧 Enhanced with West Bengal geographical knowledge using {('OpenAI GPT-4' if ai_model == 'gpt4' else 'Google Gemini')}")