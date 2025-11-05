# Python + Streamlit +Remark + Mongodb + Click or Import Cards
import streamlit as st
import google.generativeai as genai
import os
from dotenv import load_dotenv
from PIL import Image
from io import BytesIO
import re
import json
import tempfile
import pandas as pd
import time
import pymongo
from datetime import datetime
from pymongo import MongoClient

# ==========================
# CONFIGURATION & SETUP
# ==========================
load_dotenv()

# MongoDB Configuration
MONGODB_URI = os.getenv("MONGODB_URI")
if not MONGODB_URI:
    st.error("MONGODB_URI environment variable is not set.")
DB_NAME = "Product_Invoice_DB"
COLLECTION_NAME = "Product_Invoice"
REMARKS_COLLECTION_NAME = "user_remarks"

# ==========================
# MONGODB SETUP
# ==========================
def get_mongo_client():
    """Establish a MongoDB Atlas connection using SRV URI."""
    uri = os.getenv("MONGODB_URI")

    if not uri:
        st.error("❌ MONGODB_URI not found in environment variables.")
        return None

    try:
        client = MongoClient(uri, serverSelectionTimeoutMS=10000)
        client.admin.command("ping")
        print("✅ Successfully connected to MongoDB Atlas!")
        return client
    except Exception as e:
        st.error(f"❌ MongoDB connection error: {e}")
        print(f"❌ MongoDB connection error: {e}")
        return None

def init_database():
    """Initialize database and collection if they don't exist."""
    client = get_mongo_client()
    if client is not None:
        db = client[DB_NAME]
        collection = db[COLLECTION_NAME]
        remarks_collection = db[REMARKS_COLLECTION_NAME]
        return client, collection, remarks_collection
    return None, None, None

def save_to_mongodb(invoice_data):
    """Save extracted invoice data to MongoDB."""
    client, collection, _ = init_database()
    if collection is not None:
        try:
            invoice_data['extraction_timestamp'] = datetime.now()
            result = collection.insert_one(invoice_data)
            return result.inserted_id
        except Exception as e:
            st.error(f"Failed to save to MongoDB: {str(e)}")
            return None
    else:
        st.warning("Could not connect to MongoDB. Data not saved.")
        return None

def save_remarks_to_mongodb(remarks_data):
    """Save user remarks/feedback to MongoDB."""
    client, _, remarks_collection = init_database()
    if remarks_collection is not None:
        try:
            # Add timestamp
            remarks_data['submission_timestamp'] = datetime.now()
            result = remarks_collection.insert_one(remarks_data)
            return result.inserted_id
        except Exception as e:
            st.error(f"Failed to save remarks to MongoDB: {str(e)}")
            return None
    else:
        st.warning("Could not connect to MongoDB. Remarks not saved.")
        return None

# ==========================
# SPEECH TO TEXT FUNCTIONALITY - WORKING VERSION
# ==========================
import os
if os.getenv('RENDER') != 'True':
    import sounddevice as sd
    from scipy.io.wavfile import write
    import numpy as np
    import speech_recognition as sr
else:
    sd = None  # Placeholder to avoid undefined variable errors
    
def record_audio(duration=10):
    """Record audio and return the transcribed text."""
    try:
        # Record audio
        fs = 16000
        recording = sd.rec(int(duration * fs), samplerate=fs, channels=1, dtype='int16')
        sd.wait()

        # Save to temporary file
        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmpfile:
            write(tmpfile.name, fs, recording)
            tmp_path = tmpfile.name

        # Convert to text
        r = sr.Recognizer()
        with sr.AudioFile(tmp_path) as source:
            audio_data = r.record(source)
            text = r.recognize_google(audio_data)
            
        # Clean up
        try:
            os.unlink(tmp_path)
        except:
            pass
            
        return text
            
    except sr.UnknownValueError:
        return ""
    except Exception as e:
        st.error(f"Recording error: {e}")
        return ""

# ==========================
# JSON HANDLING
# ==========================
def safe_json_loads(json_str):
    """Cleaner JSON extraction from typical LLM output with markdown/links."""
    try:
        cleaned = json_str.strip()
        cleaned = re.sub(r"```(?:json)?", "", cleaned)
        cleaned = re.sub(r"\[([^\]]+)\]\([^\)]+\)", r"\1", cleaned)
        m = re.search(r'\{.*\}', cleaned, flags=re.DOTALL)
        if not m:
            return None
        json_sub = m.group(0)
        return json.loads(json_sub)
    except Exception as e:
        return None

# ==========================
# GEMINI EXTRACTION
# ==========================
def extract_with_gemini(file_data, file_type):
    """Send file (PDF or image) to Gemini and extract structured invoice data."""
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        return {"error": "GEMINI_API_KEY not found in .env file."}

    genai.configure(api_key=api_key)
    model = genai.GenerativeModel('gemini-2.5-flash')

    prompt = """
You are an expert OCR and data extraction system for invoices. Your PRIMARY FOCUS is to extract GSTIN/UIN, GSTIN fields ("seller_gstin_uin" and "buyer_gstin_uin") with ABSOLUTE ACCURACY, ensuring they are EXACTLY 15 alphanumeric characters matching the Indian GSTIN format (e.g., 22AAAAA0000A1Z5). For all other fields, extract data verbatim with high accuracy.

CRITICAL GSTIN/UIN RULES:
- "seller_gstin_uin" and "buyer_gstin_uin": MUST be EXACTLY 15 alphanumeric characters (2 digits, 5 letters, 4 digits, 1 letter, 1 digit, 1 letter/digit, 1 letter/digit, e.g., 22AAAAA0000A1Z5).
- Validate character-by-character:
  - Positions 1-2: Digits (0-9).
  - Positions 3-7: Letters (A-Z).
  - Positions 8-11: Digits (0-9).
  - Position 12: Letter (A-Z).
  - Position 13: Digit (0-9).
  - Position 14: Letter (A-Z) or Digit (0-9).
  - Position 15: Letter (A-Z) or Digit (0-9).
- If the GSTIN/UIN is unclear, blurry, has fewer than 15 characters (e.g., 12 or 13), more than 15, contains invalid characters, or has ANY ambiguity, return EMPTY STRING ("").
- Check for OCR errors EVERY TIME: 'O' vs '0', 'I' vs '1', 'l' vs '1', 'S' vs '5', 'B' vs '8', 'G' vs '6'. If uncertain, return "".
- Do NOT assume, correct, truncate, or extend GSTIN/UIN. Extract EXACTLY as shown, including prefixes like "GST#" if present.

GENERAL RULES:
- Extract ALL fields EXACTLY as they appear, character by character, without assumptions, corrections, or inferences.
- Handle handwritten, printed, multiple fonts, low-quality scans, rotated text, or noise. If unclear, return "".
- If a field is missing or not explicitly labeled, return "".
- Return ONLY a valid JSON object—NO explanations, NO extra text, NO markdown, NO code fences. If extraction fails, return {}.
- Numeric fields (e.g., quantity, rate, GST %, amount) must be returned as strings in numeric format (e.g., "18.0", "2413.83").

FIELD-SPECIFIC GUIDELINES:
- "product_name": The main product/service name or part/model heading (e.g., 'September 2025 Sponsored Jobs on Indeed.com').
- "item_description": Include extra details, warranty, specs, or comments, including GST details (e.g., 'Integrated GST @ 18% on 2413.83 SAC:998365'). If GST or HSN/SAC details appear on the same line or as supporting text, include them in "item_description" and do not create a separate item. If only a single line exists, use it as "product_name" and leave "item_description" empty.
- Combine related product and GST details into a single item entry. Do not split GST into a separate item.
- "quantity" and "rate": Extract if present (e.g., "quantity": "1", "rate": "2413.83"). If not explicitly listed, infer "quantity": "1" for single-item invoices or leave empty if unclear.
- "seller_state": Include both name and code if available (e.g., "Telangana (36)").
- "buyer_state": Include only the state name (e.g., "Maharashtra") or "" if no clear state name; do not include state code.
- "terms_of_delivery": Extract only if explicitly labeled as "Terms of Delivery", "Delivery Terms", or similar. Do not extract from payment-related terms unless clearly indicating delivery conditions.
- "purchase_order_number" and "purchase_order_date": Extract if labeled as "PO No." and "PO Date."

Return JSON in this EXACT format:
{
  "seller": {"name": "", "address": "", "telephone_number": "", "gstin_uin": "", "state": "", "email": ""},
  "buyer": {"name": "", "address": "", "gstin_uin": "", "state": ""},
  "invoice": {
    "number": "",
    "dated": "",
    "purchase_order_number": "",
    "purchase_order_date": "",
    "terms_of_delivery": ""
  },
  "items": [
    {
      "product_name": "",
      "item_description": "",
      "hsn_sac": "",
      "quantity": "",
      "rate": "",
      "percentage": "",
      "discount": "",
      "amount": "",
      "cgst_percentage": "",
      "sgst_percentage": "",
      "igst_percentage": "",
      "cgst_amount": "",
      "sgst_amount": "",
      "igst_amount": ""
    }
  ],
  "total_amount": ""
}
"""

    try:
        if file_type == 'application/pdf':
            with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as temp_file:
                temp_file.write(file_data)
                temp_file_path = temp_file.name
            uploaded_file = genai.upload_file(temp_file_path, mime_type="application/pdf")
            response = model.generate_content([prompt, uploaded_file])
            genai.delete_file(uploaded_file.name)
            os.unlink(temp_file_path)
        else:
            img = Image.open(BytesIO(file_data))
            response = model.generate_content([prompt, img])

        content = response.text.strip()
        extracted = safe_json_loads(content)
        if extracted is None:
            return {"error": f"Failed to parse JSON from API response. Raw response: {content}"}
        return extracted

    except Exception as e:
        return {"error": f"API call failed: {str(e)}"}

# ==========================
# GST CALCULATION LOGIC
# ==========================
def calculate_gst_fields(item):
    """Auto-calculate missing GST fields."""
    try:
        base_amount = float(item.get("amount", 0) or 0)
        cgst_per = float(item.get("cgst_percentage") or 0)
        sgst_per = float(item.get("sgst_percentage") or 0)
        igst_per = float(item.get("igst_percentage") or 0)
        cgst_amt = float(item.get("cgst_amount") or 0)
        sgst_amt = float(item.get("sgst_amount") or 0)
        igst_amt = float(item.get("igst_amount") or 0)

        if igst_per or igst_amt:
            if not igst_amt and igst_per:
                igst_amt = round((base_amount * igst_per) / 100, 2)
            elif not igst_per and igst_amt and base_amount:
                igst_per = round((igst_amt / base_amount) * 100, 2)
            cgst_per = sgst_per = cgst_amt = sgst_amt = 0
        else:
            total_per = cgst_per + sgst_per
            total_amt = cgst_amt + sgst_amt

            if not total_per and item.get("percentage"):
                total_per = float(item.get("percentage") or 0)
                cgst_per = sgst_per = total_per / 2

            if not total_per and total_amt and base_amount:
                total_per = round((total_amt / base_amount) * 100, 2)
                cgst_per = sgst_per = total_per / 2

            if not total_amt and total_per:
                total_amt = round((base_amount * total_per) / 100, 2)
                cgst_amt = sgst_amt = total_amt / 2

            cgst_per = round(cgst_per, 2)
            sgst_per = round(sgst_per, 2)
            cgst_amt = round(cgst_amt, 2)
            sgst_amt = round(sgst_amt, 2)

        item["cgst_percentage"] = str(cgst_per)
        item["sgst_percentage"] = str(sgst_per)
        item["igst_percentage"] = str(igst_per)
        item["cgst_amount"] = str(cgst_amt)
        item["sgst_amount"] = str(sgst_amt)
        item["igst_amount"] = str(igst_amt)
        return item

    except Exception:
        return item

# ==========================
# STREAMLIT UI
# ==========================
st.set_page_config(
    page_title="Product Invoice Scanner", 
    layout="centered",
    page_icon="📦",
    initial_sidebar_state="collapsed"
)

# Custom CSS for professional styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.8rem;
        font-weight: 700;
        text-align: center;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 1rem;
        padding: 1rem;
    }
    .sub-header {
        font-size: 1.2rem;
        text-align: center;
        color: #6c757d;
        margin-bottom: 3rem;
        font-weight: 300;
    }
    .mode-button {
        height: 120px;
        border: none;
        border-radius: 15px;
        font-size: 1.3rem;
        font-weight: 600;
        transition: all 0.3s ease;
        box-shadow: 0 4px 15px rgba(0,0,0,0.1);
    }
    .mode-button:hover {
        transform: translateY(-2px);
        box-shadow: 0 8px 25px rgba(0,0,0,0.15);
    }
    .camera-btn {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
    }
    .import-btn {
        background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%);
        color: white;
    }
    .success-box {
        padding: 1.5rem;
        border-radius: 12px;
        background: linear-gradient(135deg, #d4edda 0%, #c3e6cb 100%);
        border: 1px solid #c3e6cb;
        color: #155724;
        margin: 1rem 0;
    }
    .info-box {
        padding: 1.5rem;
        border-radius: 12px;
        background: linear-gradient(135deg, #d1ecf1 0%, #bee5eb 100%);
        border: 1px solid #bee5eb;
        color: #0c5460;
        margin: 1rem 0;
    }
    .processing-box {
        padding: 1.5rem;
        border-radius: 12px;
        background: linear-gradient(135deg, #fff3cd 0%, #ffeaa7 100%);
        border: 1px solid #ffeaa7;
        color: #856404;
        margin: 1rem 0;
    }
    .stProgress > div > div > div > div {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    }
    .file-dropdown {
        margin: 1rem 0;
        border: 1px solid #e0e0e0;
        border-radius: 8px;
        padding: 0;
    }
    .remarks-section {
        background: #f8f9fa;
        padding: 2rem;
        border-radius: 12px;
        border: 1px solid #e9ecef;
        margin-top: 2rem;
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state
if 'current_mode' not in st.session_state:
    st.session_state.current_mode = None
if 'uploaded_files' not in st.session_state:
    st.session_state.uploaded_files = []
if 'current_file_index' not in st.session_state:
    st.session_state.current_file_index = 0
if 'extraction_results' not in st.session_state:
    st.session_state.extraction_results = []
if 'processing_started' not in st.session_state:
    st.session_state.processing_started = False
if 'file_data_cache' not in st.session_state:
    st.session_state.file_data_cache = {}
if 'user_remarks' not in st.session_state:
    st.session_state.user_remarks = ""
if 'show_remarks_section' not in st.session_state:
    st.session_state.show_remarks_section = False
if 'just_recorded' not in st.session_state:
    st.session_state.just_recorded = False

# Main header
st.markdown('<div class="main-header">📦 Product Invoice Scanner</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-header">AI-Powered Product Invoice Data Extraction with Voice Feedback</div>', unsafe_allow_html=True)

# Show mode selection if no mode is selected
if st.session_state.current_mode is None:
    col1, col2 = st.columns(2)
    
    with col1:
        if st.button(
            "📷 **Click Invoice**\n\nUse camera to capture invoice", 
            key="camera_mode", 
            use_container_width=True,
            help="Take a photo of your product invoice using your camera"
        ):
            st.session_state.current_mode = "camera"
            st.rerun()
    
    with col2:
        if st.button(
            "📁 **Import Invoice**\n\nUpload PDF or image files", 
            key="import_mode", 
            use_container_width=True,
            help="Upload multiple product invoice files for batch processing"
        ):
            st.session_state.current_mode = "import"
            st.rerun()
    
    # Add feature highlights
    st.markdown("""
    <div style='text-align: center; margin-top: 3rem; padding: 2rem; background: #f8f9fa; border-radius: 12px;'>
        <h4 style='color: #495057; margin-bottom: 1rem;'>✨ Product Invoice Features</h4>
        <div style='display: flex; justify-content: space-around; flex-wrap: wrap; gap: 1rem;'>
            <div style='flex: 1; min-width: 150px;'>
                <h5 style='color: #667eea;'>🔍 GSTIN Extraction</h5>
                <p style='font-size: 0.9rem; color: #6c757d;'>Accurate 15-digit GSTIN detection</p>
            </div>
            <div style='flex: 1; min-width: 150px;'>
                <h5 style='color: #667eea;'>📊 Product Items</h5>
                <p style='font-size: 0.9rem; color: #6c757d;'>Multi-item invoice processing</p>
            </div>
            <div style='flex: 1; min-width: 150px;'>
                <h5 style='color: #667eea;'>💾 Database Storage</h5>
                <p style='font-size: 0.9rem; color: #6c757d;'>Secure MongoDB storage</p>
            </div>
            <div style='flex: 1; min-width: 150px;'>
                <h5 style='color: #667eea;'>🎤 Voice Feedback</h5>
                <p style='font-size: 0.9rem; color: #6c757d;'>Speech-to-text remarks</p>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)

# Camera Mode
elif st.session_state.current_mode == "camera":
    st.header("📷 Capture Product Invoice")
    
    # Back button
    if st.button("← Back", key="camera_back"):
        st.session_state.current_mode = None
        st.session_state.show_remarks_section = False
        st.rerun()
    
    st.markdown("Position your product invoice clearly in the camera frame and capture the image.")
    
    camera_image = st.camera_input("Take a picture of your product invoice")
    
    if camera_image:
        col1, col2 = st.columns([1, 1])
        with col1:
            st.image(camera_image, caption="Captured Invoice", use_container_width=True)
        
        with col2:
            if st.button("🔍 Extract Data", type="primary", use_container_width=True):
                with st.spinner("🤖 Analyzing product invoice with AI..."):
                    result = extract_with_gemini(camera_image.getvalue(), "image/jpeg")
                
                if "error" in result:
                    st.error(f"❌ {result['error']}")
                else:
                    st.success("✅ Data extracted successfully!")
                    
                    # Calculate GST fields for all items
                    if "items" in result:
                        for item in result["items"]:
                            item = calculate_gst_fields(item)
                    
                    # Save to MongoDB
                    inserted_id = save_to_mongodb(result)
                    if inserted_id:
                        st.success(f"✅ Data saved to database")
                    
                    # Display results
                    st.subheader("📊 Extracted Product Data")
                    st.json(result)
                    
                    # Show items in a table
                    if result.get("items"):
                        st.subheader("🛍️ Product Items")
                        items_df = pd.DataFrame(result["items"])
                        st.dataframe(items_df)
                    
                    # Show remarks section
                    st.session_state.show_remarks_section = True
                    st.rerun()

# Import Mode
elif st.session_state.current_mode == "import":
    st.header("📁 Import Product Invoices")
    
    # Back button
    if st.button("← Back", key="import_back"):
        st.session_state.current_mode = None
        st.session_state.uploaded_files = []
        st.session_state.current_file_index = 0
        st.session_state.extraction_results = []
        st.session_state.processing_started = False
        st.session_state.file_data_cache = {}
        st.session_state.show_remarks_section = False
        st.rerun()
    
    uploaded_files = st.file_uploader(
        "Select product invoice files", 
        type=["pdf", "jpg", "jpeg", "png"], 
        accept_multiple_files=True,
        help="Supported formats: PDF, JPG, JPEG, PNG"
    )
    
    if uploaded_files and not st.session_state.uploaded_files:
        # Store files in session state for sequential processing
        st.session_state.uploaded_files = uploaded_files
        st.session_state.current_file_index = 0
        st.session_state.extraction_results = []
        st.session_state.processing_started = False
        
        # Cache file data for later display
        for file in uploaded_files:
            st.session_state.file_data_cache[file.name] = file.getvalue()
        
        st.success(f"✅ {len(uploaded_files)} product invoice file(s) selected")
    
    # Show extract button if files are uploaded but processing hasn't started
    if (st.session_state.uploaded_files and 
        not st.session_state.processing_started):
        
        if st.button("🚀 Start Processing All Files", type="primary", use_container_width=True):
            st.session_state.processing_started = True
            st.rerun()
    
    # Auto-process files when processing is started
    if (st.session_state.processing_started and 
        st.session_state.current_file_index < len(st.session_state.uploaded_files)):
        
        current_index = st.session_state.current_file_index
        current_file = st.session_state.uploaded_files[current_index]
        total_files = len(st.session_state.uploaded_files)
        
        # Progress bar and status
        progress = current_index / total_files
        st.progress(progress)
        st.write(f"**Processing:** {current_index + 1} of {total_files} files - {current_file.name}")
        
        # Process current file
        with st.spinner(f"🤖 Analyzing {current_file.name}..."):
            file_data = st.session_state.file_data_cache[current_file.name]
            result = extract_with_gemini(file_data, current_file.type)
        
        # Calculate GST fields if extraction successful
        if "error" not in result and "items" in result:
            for item in result["items"]:
                item = calculate_gst_fields(item)
        
        # Store result
        if "error" in result:
            status = "❌ Failed"
            st.error(f"Error processing {current_file.name}: {result['error']}")
        else:
            status = "✅ Success"
            # Save to MongoDB
            inserted_id = save_to_mongodb(result)
        
        st.session_state.extraction_results.append({
            'file_name': current_file.name,
            'file_type': current_file.type,
            'file_data': st.session_state.file_data_cache[current_file.name],
            'result': result,
            'mongo_id': inserted_id if 'error' not in result else None,
            'status': status
        })
        
        # Move to next file
        st.session_state.current_file_index += 1
        
        # Auto-refresh to process next file
        if st.session_state.current_file_index < len(st.session_state.uploaded_files):
            st.rerun()
        else:
            # All files processed
            st.session_state.processing_started = False
            st.session_state.show_remarks_section = True
            st.rerun()
    
    # Show results when all files are processed
    if (st.session_state.uploaded_files and 
        st.session_state.current_file_index >= len(st.session_state.uploaded_files) and
        len(st.session_state.extraction_results) > 0):
        
        st.success(f"🎉 Processing completed! {len(st.session_state.extraction_results)} product invoices processed.")
        
        # Show dropdowns for each file's results
        st.subheader("📊 Extraction Results")
        
        for i, result_data in enumerate(st.session_state.extraction_results):
            with st.expander(f"{result_data['status']} - {result_data['file_name']}", expanded=False):
                col1, col2 = st.columns([1, 1])
                
                with col1:
                    st.write("**Original File:**")
                    if result_data['file_type'].startswith('image'):
                        st.image(result_data['file_data'], caption=result_data['file_name'], use_container_width=True)
                    else:
                        st.info(f"📘 PDF Document: {result_data['file_name']}")
                
                with col2:
                    st.write("**Extracted Data:**")
                    if "error" in result_data['result']:
                        st.error(result_data['result']['error'])
                    else:
                        st.json(result_data['result'])
                        if result_data['mongo_id']:
                            st.success(f"✅ Saved to database")
                        
                        # Show items table for product invoices
                        if result_data['result'].get('items'):
                            st.write("**Product Items:**")
                            items_df = pd.DataFrame(result_data['result']['items'])
                            st.dataframe(items_df)

# ==========================
# REMARKS/FEEDBACK SECTION - WORKING VERSION
# ==========================
if st.session_state.show_remarks_section:
    st.markdown("---")
    st.markdown('<div class="remarks-section">', unsafe_allow_html=True)
    st.header("💬 User Remarks & Feedback")
    st.markdown("Please provide your feedback or remarks about the extraction results. You can type your message or use the microphone to speak.")
    
    # Handle recording if it just happened
    if st.session_state.just_recorded:
        st.session_state.just_recorded = False
        # Force update the text area
        st.rerun()
    
    # Text input area - use a unique key
    remarks_key = "remarks_textarea_" + str(hash(st.session_state.user_remarks))
    remarks_text = st.text_area(
        "🗒️ Your Remarks:",
        value=st.session_state.user_remarks,
        height=150,
        placeholder="Type your feedback here or use the microphone below to speak...",
        key=remarks_key
    )
    
    # Update session state with text input
    if remarks_text != st.session_state.user_remarks:
        st.session_state.user_remarks = remarks_text
    
    # Speech to text section
    st.subheader("🎤 Voice Input")
    col1, col2, col3 = st.columns([2, 1, 1])
    
    with col1:
        recording_duration = st.slider(
            "🎧 Recording Duration (seconds)",
            min_value=2,
            max_value=10,
            value=5,
            key="recording_duration"
        )
    
    with col2:
        # Use a form with a unique key for the record button
        record_form_key = "record_form_" + str(hash(st.session_state.user_remarks))
        with st.form(key=record_form_key):
            record_submitted = st.form_submit_button("🎙️ Record Voice", disabled=True)
            st.warning("Voice recording is not supported on this hosted version. Use text input instead.")
            
            if record_submitted:
                with st.spinner("🎤 Recording... Speak now!"):
                    transcribed_text = record_audio(recording_duration)
                
                if transcribed_text:
                    # Update the remarks directly
                    if st.session_state.user_remarks:
                        st.session_state.user_remarks += " " + transcribed_text
                    else:
                        st.session_state.user_remarks = transcribed_text
                    
                    st.session_state.just_recorded = True
                    st.success("✅ Voice recorded and text added!")
                    st.rerun()
    
    with col3:
        if st.button("🗑️ Clear", use_container_width=True, key="clear_remarks"):
            st.session_state.user_remarks = ""
            st.rerun()
    
    # Submit button
    submit_col1, submit_col2 = st.columns([1, 1])
    with submit_col1:
        if st.button("✅ Submit Remarks", type="primary", use_container_width=True, key="submit_remarks"):
            if st.session_state.user_remarks.strip():
                # Prepare remarks data
                remarks_data = {
                    'remarks': st.session_state.user_remarks.strip(),
                    'submission_type': 'camera' if st.session_state.current_mode == 'camera' else 'import',
                    'file_count': 1 if st.session_state.current_mode == 'camera' else len(st.session_state.extraction_results),
                    'extraction_results_count': len([r for r in st.session_state.extraction_results if 'error' not in r['result']]) if st.session_state.current_mode == 'import' else 1
                }
                
                # Save to MongoDB
                remarks_id = save_remarks_to_mongodb(remarks_data)
                if remarks_id:
                    st.success("✅ Your remarks have been submitted successfully!")
                    st.session_state.user_remarks = ""
                    st.session_state.show_remarks_section = False
                    st.rerun()
                else:
                    st.error("❌ Failed to save remarks to database.")
            else:
                st.warning("⚠️ Please enter some remarks before submitting.")
    
    with submit_col2:
        if st.button("❌ Cancel", use_container_width=True, key="cancel_remarks"):
            st.session_state.user_remarks = ""
            st.session_state.show_remarks_section = False
            st.rerun()
    
    st.markdown('</div>', unsafe_allow_html=True)

# Reset buttons for import mode after processing
if (st.session_state.current_mode == "import" and 
    st.session_state.uploaded_files and 
    st.session_state.current_file_index >= len(st.session_state.uploaded_files) and
    not st.session_state.show_remarks_section):
    
    col1, col2 = st.columns(2)
    with col1:
        if st.button("🔄 Process More Files", use_container_width=True, key="process_more"):
            st.session_state.uploaded_files = []
            st.session_state.current_file_index = 0
            st.session_state.extraction_results = []
            st.session_state.processing_started = False
            st.session_state.file_data_cache = {}
            st.session_state.show_remarks_section = False
            st.rerun()
    with col2:
        if st.button("🏠 Back to Main Menu", use_container_width=True, key="back_main"):
            st.session_state.current_mode = None
            st.session_state.uploaded_files = []
            st.session_state.current_file_index = 0
            st.session_state.extraction_results = []
            st.session_state.processing_started = False
            st.session_state.file_data_cache = {}
            st.session_state.show_remarks_section = False
            st.rerun()

# MongoDB connection status in sidebar
with st.sidebar:
    st.header("🔧 System Status")
    client = get_mongo_client()
    if client is not None:
        st.success("✅ **Database Connected**")
        
        # Show collections info
        db = client[DB_NAME]
        collections = db.list_collection_names()
        st.write("**Collections:**")
        for coll in collections:
            count = db[coll].count_documents({})
            st.write(f"- {coll}: {count} documents")
        
        client.close()
    else:
        st.error("❌ **Database Disconnected**")
    
    st.header("ℹ️ About")
    st.info("""
    **Product Invoice Scanner** uses advanced AI to automatically extract and validate data from PRODUCT invoices.
    
    **Extracts:**
    • GSTIN Numbers (15-digit)
    • Product Items & Quantities
    • Seller/Buyer Details  
    • Invoice Amounts & Taxes
    • HSN/SAC Codes
    • GST Calculations
    
    **New Features:**
    • 🎤 Voice-to-text remarks
    • 💬 User feedback system
    • 💾 Separate remarks database
    • 📊 Product item tables
    • 🔍 Advanced GST calculations

    """)

