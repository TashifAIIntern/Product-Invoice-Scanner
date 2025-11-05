# Python + FastAPI + MongoDB + Remark Text and Speech + Data Extraction
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from typing import List, Optional, Dict, Any
import google.generativeai as genai
import os
from dotenv import load_dotenv
from PIL import Image
from io import BytesIO
import tempfile
import pymongo
from datetime import datetime
import sounddevice as sd
from scipy.io.wavfile import write
import numpy as np
import speech_recognition as sr
import re
import json
import uvicorn
import traceback

# ==========================
# CONFIGURATION
# ==========================
load_dotenv()

# MongoDB Configuration
MONGODB_URI = os.getenv("MONGODB_URI", "mongodb://localhost:27017/product_Invoice_DB")
DB_NAME = "Product_Invoice_DB"
COLLECTION_NAME = "Product_Invoice"
REMARKS_COLLECTION_NAME = "user_remarks"

# ==========================
# FASTAPI APP
# ==========================
app = FastAPI(
    title="Product Invoice Scanner API",
    description="AI-Powered Product Invoice Data Extraction with Voice Feedback",
    version="1.0.0"
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ==========================
# MONGODB SETUP
# ==========================
def get_mongo_client():
    """Get MongoDB client connection."""
    try:
        client = pymongo.MongoClient(MONGODB_URI)
        # Test connection
        client.admin.command('ping')
        return client
    except Exception as e:
        print(f"Failed to connect to MongoDB: {str(e)}")
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
    """Save extracted product invoice data to MongoDB."""
    client, collection, _ = init_database()
    if collection is not None:
        try:
            # Add timestamp
            invoice_data['extraction_timestamp'] = datetime.now()
            result = collection.insert_one(invoice_data)
            print(f"✅ Product invoice saved to MongoDB with ID: {result.inserted_id}")
            return result.inserted_id
        except Exception as e:
            print(f"❌ Failed to save to MongoDB: {str(e)}")
            return None
    else:
        print("❌ Could not connect to MongoDB. Data not saved.")
        return None

def save_remarks_to_mongodb(remarks_data):
    """Save user remarks/feedback to MongoDB."""
    client, _, remarks_collection = init_database()
    if remarks_collection is not None:
        try:
            # Add timestamp
            remarks_data['submission_timestamp'] = datetime.now()
            result = remarks_collection.insert_one(remarks_data)
            print(f"✅ Remarks saved to MongoDB with ID: {result.inserted_id}")
            return result.inserted_id
        except Exception as e:
            print(f"❌ Failed to save remarks to MongoDB: {str(e)}")
            return None
    else:
        print("❌ Could not connect to MongoDB. Remarks not saved.")
        return None

# ==========================
# SPEECH TO TEXT FUNCTIONALITY
# ==========================
def record_audio(duration=5):
    """Record audio and return the transcribed text."""
    try:
        print(f"🎤 Recording audio for {duration} seconds...")
        
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
        
        print(f"📝 Transcribed text: {text}")
        return text
            
    except sr.UnknownValueError:
        print("❌ No speech detected")
        return ""
    except Exception as e:
        print(f"❌ Recording error: {e}")
        return ""

# ==========================
# JSON HANDLING
# ==========================
def safe_json_loads(json_str):
    """Cleanly parse JSON from Gemini response."""
    try:
        # Remove any markdown code blocks
        cleaned = re.sub(r"```(json)?", "", json_str.strip())
        # Remove any extra text before or after JSON
        json_match = re.search(r'\{.*\}', cleaned, re.DOTALL)
        if json_match:
            return json.loads(json_match.group(0))
        return None
    except Exception as e:
        print(f"❌ JSON parsing error: {e}")
        print(f"❌ Raw response: {json_str}")
        return None

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
# DATA CLEANING FUNCTIONS
# ==========================
def clean_amount_field(amount_str):
    """Clean amount fields by removing currency symbols and commas"""
    if not amount_str:
        return ""
    # Remove currency symbols, commas, and extra spaces
    cleaned = re.sub(r'[^\d.]', '', str(amount_str))
    try:
        # Ensure it's a valid float
        return str(float(cleaned))
    except ValueError:
        return ""

def clean_gstin_uin(gstin):
    """Clean and validate GSTIN/UIN"""
    if not gstin:
        return ""
    gstin = str(gstin).strip().upper()
    gstin = re.sub(r'[^A-Z0-9]', '', gstin)
    # GSTIN should be exactly 15 alphanumeric characters
    if len(gstin) == 15 and re.match(r'^\d{2}[A-Z]{5}\d{4}[A-Z]\d[A-Z0-9][A-Z0-9]$', gstin):
        return gstin
    return ""

# ==========================
# GEMINI EXTRACTION - PRODUCT INVOICE VERSION
# ==========================
def extract_with_gemini(file_data, file_type):
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        return {"error": "GEMINI_API_KEY not found in .env file."}

    try:
        genai.configure(api_key=api_key)
        
        # Use a more reliable model
        model = genai.GenerativeModel("gemini-2.5-flash")
        
        prompt = """
You are an expert OCR and data extraction system for PRODUCT invoices. Your PRIMARY FOCUS is to extract GSTIN/UIN, GSTIN fields ("seller_gstin_uin" and "buyer_gstin_uin") with ABSOLUTE ACCURACY, ensuring they are EXACTLY 15 alphanumeric characters matching the Indian GSTIN format (e.g., 22AAAAA0000A1Z5). For all other fields, extract data verbatim with high accuracy.

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
  "seller": {
    "name": "", 
    "address": "", 
    "telephone_number": "", 
    "gstin_uin": "", 
    "state": "", 
    "email": ""
  },
  "buyer": {
    "name": "", 
    "address": "", 
    "gstin_uin": "", 
    "state": ""
  },
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
        if file_type == 'application/pdf':
            with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as temp_file:
                temp_file.write(file_data)
                temp_path = temp_file.name
            uploaded = genai.upload_file(temp_path, mime_type="application/pdf")
            response = model.generate_content([uploaded, prompt])
            genai.delete_file(uploaded.name)
            os.unlink(temp_path)
        else:
            image = Image.open(BytesIO(file_data))
            response = model.generate_content([image, prompt])

        print(f"🔍 Raw Gemini response: {response.text}")

        # Parse the response
        extracted = safe_json_loads(response.text)
        
        if not extracted:
            return {"error": "Failed to parse Gemini response as JSON."}

        # Validate and clean the extracted data
        seller = extracted.get("seller", {})
        buyer = extracted.get("buyer", {})
        invoice = extracted.get("invoice", {})
        items = extracted.get("items", [])
        total_amount = extracted.get("total_amount", "")

        # Clean GSTIN/UIN numbers
        seller["gstin_uin"] = clean_gstin_uin(seller.get("gstin_uin"))
        buyer["gstin_uin"] = clean_gstin_uin(buyer.get("gstin_uin"))

        # Clean amount fields for items
        for item in items:
            amount_fields = ["rate", "discount", "amount", "cgst_amount", "sgst_amount", "igst_amount"]
            for field in amount_fields:
                if field in item:
                    item[field] = clean_amount_field(item[field])
            
            # Calculate GST fields
            item = calculate_gst_fields(item)

        # Clean total amount
        total_amount = clean_amount_field(total_amount)

        # Ensure all required fields exist
        result = {
            "seller": {
                "name": seller.get("name", ""),
                "address": seller.get("address", ""),
                "telephone_number": seller.get("telephone_number", ""),
                "gstin_uin": seller.get("gstin_uin", ""),
                "state": seller.get("state", ""),
                "email": seller.get("email", "")
            },
            "buyer": {
                "name": buyer.get("name", ""),
                "address": buyer.get("address", ""),
                "gstin_uin": buyer.get("gstin_uin", ""),
                "state": buyer.get("state", "")
            },
            "invoice": {
                "number": invoice.get("number", ""),
                "dated": invoice.get("dated", ""),
                "purchase_order_number": invoice.get("purchase_order_number", ""),
                "purchase_order_date": invoice.get("purchase_order_date", ""),
                "terms_of_delivery": invoice.get("terms_of_delivery", "")
            },
            "items": items,
            "total_amount": total_amount
        }

        print(f"✅ Successfully extracted product invoice data")
        return result
        
    except Exception as e:
        error_details = traceback.format_exc()
        print(f"❌ Gemini extraction failed: {str(e)}")
        print(f"❌ Error details: {error_details}")
        return {"error": f"Gemini extraction failed: {str(e)}"}

# ==========================
# PYDANTIC MODELS
# ==========================
class UnifiedRemarksRequest(BaseModel):
    text_remarks: Optional[str] = None
    record_audio: bool = False
    audio_duration: int = 5
    submission_type: str = "general"

# ==========================
# API ROUTES
# ==========================

@app.get("/")
async def root():
    """Root endpoint with API information"""
    return {
        "message": "Product Invoice Scanner API",
        "version": "1.0.0",
        "endpoints": {
            "extract_invoices": "/extract_invoices (POST)",
            "submit_remarks": "/submit_remarks (POST)",
            "database_status": "/database_status (GET)"
        }
    }

@app.post("/extract_invoices")
async def extract_invoices(files: List[UploadFile] = File(...)):
    """
    Extract data from product invoice files.
    Supports single or multiple files.
    Automatically saves to database.
    """
    try:
        results = []
        
        for file in files:
            print(f"📁 Processing file: {file.filename}")
            
            # Read file data
            file_data = await file.read()
            
            # Extract data using Gemini
            extraction_result = extract_with_gemini(file_data, file.content_type)
            
            # Always save to MongoDB (even if there's an error, save the error result)
            mongo_id = None
            if extraction_result:
                mongo_id = save_to_mongodb({
                    "file_name": file.filename,
                    "file_type": file.content_type,
                    "file_size": len(file_data),
                    "extraction_result": extraction_result,
                    "extraction_timestamp": datetime.now()
                })
            
            # Response structure
            file_result = {
                "file_name": file.filename,
                "file_type": file.content_type,
                "file_size": len(file_data),
                "extraction_result": extraction_result,
                "database_saved": mongo_id is not None,
                "mongo_id": str(mongo_id) if mongo_id else None
            }
            
            results.append(file_result)
        
        return {
            "success": True,
            "total_files": len(results),
            "processed_files": len(results),
            "results": results
        }
        
    except Exception as e:
        error_details = traceback.format_exc()
        print(f"❌ Error in extract_invoices: {error_details}")
        raise HTTPException(
            status_code=500,
            detail=f"Processing failed: {str(e)}"
        )

@app.post("/submit_remarks")
async def submit_remarks(request: UnifiedRemarksRequest):
    """
    Unified remarks endpoint - handles both text and voice remarks
    Returns the remarks data back to user
    """
    try:
        final_remarks = ""
        remarks_type = ""
        
        # Case 1: Record audio if requested
        if request.record_audio:
            print("🎤 Recording audio for remarks...")
            transcribed_text = record_audio(request.audio_duration)
            
            if transcribed_text:
                final_remarks = transcribed_text
                remarks_type = "audio"
                print(f"✅ Voice remarks recorded: {final_remarks}")
            else:
                return JSONResponse(
                    status_code=400,
                    content={
                        "success": False,
                        "message": "No speech detected during audio recording",
                        "remarks_text": "",
                        "remarks_type": "",
                        "remarks_id": None
                    }
                )
        
        # Case 2: Use text remarks if provided
        elif request.text_remarks and request.text_remarks.strip():
            final_remarks = request.text_remarks.strip()
            remarks_type = "text"
            print(f"✅ Text remarks received: {final_remarks}")
        
        # Case 3: Neither text nor successful audio recording
        else:
            return JSONResponse(
                status_code=400,
                content={
                    "success": False,
                    "message": "Please provide either text remarks or enable audio recording",
                    "remarks_text": "",
                    "remarks_type": "",
                    "remarks_id": None
                }
            )
        
        # Prepare remarks data for database
        remarks_data = {
            'remarks': final_remarks,
            'submission_type': request.submission_type,
            'remarks_type': remarks_type,
            'timestamp': datetime.now()
        }
        
        # Save to MongoDB
        remarks_id = save_remarks_to_mongodb(remarks_data)
        
        if remarks_id:
            return {
                "success": True,
                "remarks_id": str(remarks_id),
                "remarks_text": final_remarks,
                "remarks_type": remarks_type,
                "submission_type": request.submission_type,
                "message": "Remarks submitted successfully"
            }
        else:
            return JSONResponse(
                status_code=500,
                content={
                    "success": False,
                    "message": "Failed to save remarks to database",
                    "remarks_text": final_remarks,
                    "remarks_type": remarks_type,
                    "remarks_id": None
                }
            )
            
    except Exception as e:
        error_details = traceback.format_exc()
        print(f"❌ Error in submit_remarks: {error_details}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to submit remarks: {str(e)}"
        )

@app.get("/database_status")
async def get_database_status():
    """Get database connection status and statistics"""
    try:
        client = get_mongo_client()
        if client is not None:
            db = client[DB_NAME]
            invoices_collection = db[COLLECTION_NAME]
            remarks_collection = db[REMARKS_COLLECTION_NAME]
            
            invoices_count = invoices_collection.count_documents({})
            remarks_count = remarks_collection.count_documents({})
            
            client.close()
            
            return {
                "connected": True,
                "total_invoices": invoices_count,
                "total_remarks": remarks_count,
                "message": "Database connected successfully"
            }
        else:
            return {
                "connected": False,
                "total_invoices": 0,
                "total_remarks": 0,
                "message": "Database connection failed"
            }
            
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Error checking database status: {str(e)}"
        )

# ==========================
# MAIN APPLICATION
# ==========================
if __name__ == "__main__":
    print("🚀 Starting Product Invoice Scanner API...")
    print("📦 Product invoice extraction endpoint available at: POST /extract_invoices")
    print("📝 Unified remarks endpoint available at: POST /submit_remarks")
    uvicorn.run(
        "main:app",
        host="127.0.0.1",
        port=8002,  # Different port than service invoice API
        reload=True
    )