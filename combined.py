import os
import json
import time
import google.generativeai as genai
from PIL import Image
import pandas as pd
from dotenv import load_dotenv
from typing import Dict, Optional, List
import logging
import re

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# --- Configuration ---
load_dotenv()
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

if not GOOGLE_API_KEY:
    raise ValueError("GOOGLE_API_KEY not found in .env file. Please create a .env file.")

genai.configure(api_key=GOOGLE_API_KEY)

def build_expert_prompt() -> str:
    """
    Builds the advanced, structured Chain-of-Thought prompt for both Index 1 and Index 2 documents.
    Enhanced with West Bengal geographical knowledge and handwriting interpretation.
    """
    return """
    You are an expert document analyst specializing in extracting structured information from handwritten West Bengal government registration records from the 1960s era. Your task is to convert the content of this document into a well-structured JSON format.

    GEOGRAPHICAL CONTEXT - WEST BENGAL 1960s:
    Common districts: Howrah, Hooghly, Burdwan (now Purba Bardhaman), 24 Parganas, Nadia, Murshidabad, Birbhum, Bankura, Purulia, Jalpaiguri, Darjeeling, Cooch Behar, West Dinajpur, Malda
    Common towns/PS: Chinsurah, Serampore, Chandernagore, Rishra, Uttarpara, Bally, Barrackpore, Dum Dum, Barasat, Basirhat, Katwa, Kalna, Memari, Asansol, Durgapur, Raniganj
    Common police stations (PS): Chinsurah, Serampore, Chandernagore, Rishra, Uttarpara, Bally, Barrackpore, Galsi, Memari, Katwa, Kalna
    
    HANDWRITING INTERPRETATION GUIDELINES:
    - Old handwriting may have 'a' looking like 'o', 'n' like 'u', 'r' like 'v'
    - Numbers: '1' may look like 'l', '5' like 'S', '0' like 'O'
    - Common abbreviations: 'S/o' (son of), 'D/o' (daughter of), 'W/o' (wife of), 'PS' (Police Station), 'Dist' (District)
    - 'do', '〃', '"', ',,', '--', or any repetitive mark indicates "same as above"
    
    SPELLING CORRECTIONS:
    Auto-correct common misspellings using geographical knowledge:
    - "Burdwan" for variations like "Bardwan", "Burdhaman"
    - "Chinsurah" for "Chinsura", "Chunchura"
    - "Hooghly" for "Hugly", "Hugli"
    - "Howrah" for "Haora"
    - Apply similar corrections for other West Bengal locations

    IMPORTANT: First determine if this is an INDEX 1 or INDEX 2 document by looking at the header text.

    Follow this step-by-step reasoning process:

    Step 1: Document Type Identification
    - Look at the document header to identify if this is "INDEX No. I" or "INDEX No. II"
    - INDEX 1: Contains information about PERSONS involved in transactions
    - INDEX 2: Contains information about PROPERTIES involved in transactions

    Step 2: Document Analysis
    - Document type: Property registration index register
    - Government department: West Bengal Government
    - Time period: Look for year (typically 1950s-1960s)
    - Purpose: To index registered deeds

    Step 3: Field Extraction Based on Document Type

    FOR INDEX 1 DOCUMENTS (Person-based):
    Columns contain:
    1. Name of person (handle poor handwriting, common Bengali names)
    2. Father's name, residence, or other particulars for identification
    3. Nature and value of interest in the transaction (Vendor/Vendee/Donee/Donor)
    4. Where registered (correct spelling using geographical knowledge)
    5. Serial number
    6. Volume number
    7. Page numbers

    FOR INDEX 2 DOCUMENTS (Property-based):
    Columns contain:
    1. Name of property (can be address, village name)
    2. Town (correct using West Bengal geographical knowledge)
    3. District and sub-district (apply geographical corrections)
    4. Nature of transaction (Sale/Gift/Mortgage/etc.)
    5. Where registered
    6. Serial number
    7. Volume number
    8. Page numbers

    Step 4: Handwriting Interpretation and Ditto Mark Resolution
    - Provide best interpretation for unclear handwritten text
    - Mark uncertain readings as [UNCLEAR: possible_text]
    - Note completely illegible text as [ILLEGIBLE]
    - For "ditto" marks ('do', '〃', '"', '--', ',,'), inherit the corresponding value from the entry immediately above
    - Apply geographical spell-checking for place names
    - Correct common handwriting misinterpretations

    Step 5: JSON Output
    Create appropriate JSON structure based on document type identified in Step 1.

    REQUIRED JSON OUTPUT STRUCTURE:

    For INDEX 1 (Person-based) documents:
    ```json
    {
      "document_metadata": {
        "document_type": "Property Registration Index Register - Index 1",
        "index_type": "INDEX_1",
        "form_number": "string",
        "office_location": "string",
        "year": "string",
        "page_number_on_document": "string",
        "extraction_confidence": "high|medium|low"
      },
      "document_content": {
        "entries": [
          {
            "serial_number": "string",
            "entry_details": {
              "name_of_person": "string",
              "additional_information": "string",
              "interest_of_person_in_transaction": "string",
              "where_registered": "string",
              "book_1_volume": "string",
              "book_1_page": "string"
            },
            "confidence_notes": ["list of uncertain fields or corrections made"]
          }
        ]
      },
      "extraction_notes": {
        "unclear_sections": ["list of unclear text sections"],
        "missing_information": ["list of missing fields"],
        "interpretation_assumptions": ["list of assumptions made"],
        "ditto_resolutions": ["list of ditto marks resolved"],
        "spelling_corrections": ["list of geographical corrections made"]
      }
    }
    ```

    For INDEX 2 (Property-based) documents:
    ```json
    {
      "document_metadata": {
        "document_type": "Property Registration Index Register - Index 2",
        "index_type": "INDEX_2",
        "form_number": "string",
        "office_location": "string",
        "year": "string",
        "page_number_on_document": "string",
        "extraction_confidence": "high|medium|low"
      },
      "document_content": {
        "entries": [
          {
            "serial_number": "string",
            "entry_details": {
              "name_of_property": "string",
              "town": "string",
              "district_and_sub_district": "string",
              "nature_of_transaction": "string",
              "where_registered": "string",
              "book_1_volume": "string",
              "book_1_page": "string"
            },
            "confidence_notes": ["list of uncertain fields or corrections made"]
          }
        ]
      },
      "extraction_notes": {
        "unclear_sections": ["list of unclear text sections"],
        "missing_information": ["list of missing fields"],
        "interpretation_assumptions": ["list of assumptions made"],
        "ditto_resolutions": ["list of ditto marks resolved"],
        "spelling_corrections": ["list of geographical corrections made"]
      }
    }
    ```

    CRITICAL INSTRUCTIONS:
    1. This is a historical administrative document for academic research purposes
    2. Return ONLY the JSON object. No markdown backticks or additional text.
    3. When you see ditto marks, copy the value from the corresponding field in the entry above
    4. Apply geographical spell-checking for all place names
    5. Be aggressive in interpreting poor handwriting using context
    6. Always fill confidence_notes with any corrections or assumptions made
    7. Use your expertise to interpret unclear handwriting based on context and common names/places
    """

def build_fallback_prompt() -> str:
    """Simplified prompt for safety filter issues."""
    return """
    Extract information from this historical administrative document image for academic research purposes.

    This is a West Bengal government registration index from the 1960s. Look for:
    - Serial numbers
    - Names of persons or properties
    - Registration locations
    - Volume and page numbers
    - Any additional identifying information

    Return as JSON:

    {
      "document_metadata": {
        "document_type": "Registration Index",
        "index_type": "INDEX_1",
        "form_number": "Unknown",
        "office_location": "Unknown", 
        "year": "Unknown",
        "page_number_on_document": "Unknown",
        "extraction_confidence": "medium"
      },
      "document_content": {
        "entries": [
          {
            "serial_number": "number",
            "entry_details": {
              "name_of_person": "person name",
              "additional_information": "details",
              "interest_of_person_in_transaction": "role",
              "where_registered": "location",
              "book_1_volume": "volume",
              "book_1_page": "page"
            },
            "confidence_notes": ["notes"]
          }
        ]
      },
      "extraction_notes": {
        "unclear_sections": [],
        "missing_information": [],
        "interpretation_assumptions": [],
        "ditto_resolutions": [],
        "spelling_corrections": []
      }
    }

    Return only valid JSON.
    """

def validate_extracted_data(data: Dict) -> bool:
    """Validate the structure of extracted data."""
    try:
        if not isinstance(data, dict):
            return False
            
        required_keys = ['document_metadata', 'document_content', 'extraction_notes']
        if not all(key in data for key in required_keys):
            logger.error(f"Missing required keys. Found: {list(data.keys())}")
            return False
        
        if 'entries' not in data['document_content']:
            logger.error("Missing 'entries' in document_content")
            return False
            
        # Check and normalize index type
        index_type = data['document_metadata'].get('index_type', '')
        normalized_index_type = normalize_index_type(index_type)
        
        if normalized_index_type not in ['INDEX_1', 'INDEX_2']:
            logger.error(f"Invalid index_type: {index_type} (normalized: {normalized_index_type})")
            return False
        
        # Update the data with normalized index type
        data['document_metadata']['index_type'] = normalized_index_type
            
        # Validate entries structure
        entries = data['document_content']['entries']
        if not isinstance(entries, list):
            logger.error("Entries is not a list")
            return False
            
        for i, entry in enumerate(entries):
            if not isinstance(entry, dict):
                logger.error(f"Entry {i} is not a dict")
                return False
            if 'entry_details' not in entry:
                logger.error(f"Entry {i} missing entry_details")
                return False
                
        return True
    except Exception as e:
        logger.error(f"Validation error: {e}")
        return False

def normalize_index_type(index_type: str) -> str:
    """Normalize various index type formats to standard format."""
    if not index_type:
        return "INDEX_1"  # Default
    
    index_type_clean = index_type.upper().strip()
    
    # Handle various formats
    if "INDEX NO. I" in index_type_clean or "INDEX I" in index_type_clean:
        return "INDEX_1"
    elif "INDEX NO. II" in index_type_clean or "INDEX II" in index_type_clean:
        return "INDEX_2"
    elif "INDEX_1" in index_type_clean or "INDEX1" in index_type_clean:
        return "INDEX_1"
    elif "INDEX_2" in index_type_clean or "INDEX2" in index_type_clean:
        return "INDEX_2"
    else:
        # Try to extract roman numerals
        if "I" in index_type_clean and "II" not in index_type_clean:
            return "INDEX_1"
        elif "II" in index_type_clean:
            return "INDEX_2"
        else:
            logger.warning(f"Could not normalize index_type: {index_type}, defaulting to INDEX_1")
            return "INDEX_1"

# Also update the model names - Gemini 2.5 doesn't exist yet
def extract_data_from_image(image_path: str, max_retries: int = 3) -> Optional[Dict]:
    """Enhanced extraction with better error handling and safety filter workarounds."""
    logger.info(f"📄 Processing image: {image_path}")
    
    # Validate image file exists and is readable
    if not os.path.exists(image_path):
        logger.error(f"❌ Image file not found: {image_path}")
        return None
    
    try:
        # Load and validate image
        image = Image.open(image_path)
        image.verify()  # Check if image is corrupted
        image = Image.open(image_path)  # Reopen after verify
        
        # Convert to RGB if needed
        if image.mode != 'RGB':
            image = image.convert('RGB')
            
    except Exception as e:
        logger.error(f"❌ Error opening/processing image {image_path}: {e}")
        return None

    # Try different models and configurations - FIXED MODEL NAMES
    models_to_try = ['gemini-2.5-pro', 'gemini-2.5-flash']  # Fixed: 2.5 doesn't exist
    prompts_to_try = [build_expert_prompt(), build_fallback_prompt()]
    
    for model_name in models_to_try:
        model = genai.GenerativeModel(model_name)
        logger.info(f"🤖 Trying model: {model_name}")
        
        for prompt_idx, prompt in enumerate(prompts_to_try):
            logger.info(f"📝 Using prompt strategy {prompt_idx + 1}")
            
            for attempt in range(max_retries):
                try:
                    logger.info(f"🤖 API call attempt {attempt + 1}/{max_retries}")
                    
                    # More permissive generation config
                    generation_config = genai.types.GenerationConfig(
                        temperature=0.1,
                        max_output_tokens=8192,
                        response_mime_type="application/json",
                        candidate_count=1
                    )
                    
                    # Most permissive safety settings
                    safety_settings = [
                        {"category": "HARM_CATEGORY_HARASSMENT", "threshold": "BLOCK_NONE"},
                        {"category": "HARM_CATEGORY_HATE_SPEECH", "threshold": "BLOCK_NONE"},
                        {"category": "HARM_CATEGORY_SEXUALLY_EXPLICIT", "threshold": "BLOCK_NONE"},
                        {"category": "HARM_CATEGORY_DANGEROUS_CONTENT", "threshold": "BLOCK_NONE"},
                    ]
                    
                    response = model.generate_content(
                        [prompt, image],
                        generation_config=generation_config
                    )
                    
                    # Enhanced response validation
                    if not response.candidates:
                        logger.warning(f"No candidates in response on attempt {attempt + 1}")
                        if attempt < max_retries - 1:
                            time.sleep(2 ** attempt)
                            continue
                        else:
                            logger.error("No candidates returned in final attempt")
                            break
                    
                    candidate = response.candidates[0]
                    finish_reason = candidate.finish_reason
                    
                    logger.info(f"Response finish_reason: {finish_reason}")
                    
                    if finish_reason == 2:  # SAFETY
                        logger.warning("Response blocked by safety filters, trying next strategy...")
                        break  # Try next prompt/model
                    elif finish_reason != 1:  # 1 = STOP (normal completion)
                        logger.warning(f"Unusual finish_reason: {finish_reason}")
                        if finish_reason == 3:  # RECITATION
                            logger.error("Response blocked due to recitation")
                        elif finish_reason == 4:  # OTHER
                            logger.error("Response blocked for other reasons")
                        
                        if attempt < max_retries - 1:
                            time.sleep(2 ** attempt)
                            continue
                        else:
                            break
                    
                    if not hasattr(response, 'text') or not response.text:
                        logger.warning(f"Empty response.text on attempt {attempt + 1}")
                        if attempt < max_retries - 1:
                            time.sleep(2 ** attempt)
                            continue
                        else:
                            logger.error("Empty response in final attempt")
                            break
                    
                    # Clean and parse JSON
                    data = clean_and_parse_json(response.text)
                    
                    if data and validate_extracted_data(data):
                        doc_type = data.get('document_metadata', {}).get('index_type', 'UNKNOWN')
                        entries_count = len(data.get('document_content', {}).get('entries', []))
                        logger.info(f"✅ Successfully extracted {entries_count} entries from {doc_type} document using {model_name}")
                        return data
                    else:
                        logger.warning(f"Invalid data structure on attempt {attempt + 1}")
                        if data:
                            logger.error(f"Data structure: {list(data.keys()) if isinstance(data, dict) else type(data)}")
                            # Log the actual index_type for debugging
                            if 'document_metadata' in data:
                                actual_index_type = data['document_metadata'].get('index_type', 'NOT_FOUND')
                                logger.error(f"Actual index_type in response: '{actual_index_type}'")
                        
                        if attempt < max_retries - 1:
                            time.sleep(2 ** attempt)
                            continue
                            
                except Exception as e:
                    logger.error(f"❌ Error on attempt {attempt + 1}: {e}")
                    if attempt < max_retries - 1:
                        time.sleep(2 ** attempt)
                        continue
    
    logger.error("❌ All retry attempts with all models and prompts failed")
    return None

def clean_and_parse_json(response_text: str) -> Optional[Dict]:
    """Enhanced JSON cleaning and parsing with multiple fallback strategies."""
    if not response_text or not response_text.strip():
        logger.error("Empty response text")
        return None
        
    try:
        # Clean the response
        cleaned = response_text.strip()
        
        # Remove markdown code blocks
        if cleaned.startswith("```json"):
            cleaned = cleaned[7:]
        elif cleaned.startswith("```"):
            cleaned = cleaned[3:]
        if cleaned.endswith("```"):
            cleaned = cleaned[:-3]
        
        cleaned = cleaned.strip()
        
        # Log the first 500 chars for debugging
        logger.info(f"Attempting to parse JSON (first 500 chars): {cleaned[:500]}...")
        
        # Strategy 1: Try parsing as-is
        try:
            data = json.loads(cleaned)
            logger.info("✅ JSON parsed successfully with strategy 1")
            return data
        except json.JSONDecodeError as e:
            logger.warning(f"Strategy 1 failed: {e}")
        
        # Strategy 2: Try to extract JSON object using regex
        json_pattern = r'\{(?:[^{}]|{(?:[^{}]|{[^{}]*})*})*\}'
        matches = re.findall(json_pattern, cleaned, re.DOTALL)
        
        for match in matches:
            try:
                data = json.loads(match)
                logger.info("✅ JSON parsed successfully with strategy 2")
                return data
            except json.JSONDecodeError:
                continue
        
        # Strategy 3: Try with json5 for more lenient parsing
        try:
            import json5
            data = json5.loads(cleaned)
            logger.info("✅ JSON parsed successfully with strategy 3 (json5)")
            return data
        except Exception as e:
            logger.warning(f"Strategy 3 (json5) failed: {e}")
        
        # Strategy 4: Try to fix common JSON issues
        # Fix trailing commas
        fixed = re.sub(r',(\s*[}\]])', r'\1', cleaned)
        # Fix unescaped quotes in strings
        fixed = re.sub(r'(?<!\\)"(?=\w)', r'\\"', fixed)
        
        try:
            data = json.loads(fixed)
            logger.info("✅ JSON parsed successfully with strategy 4")
            return data
        except json.JSONDecodeError as e:
            logger.warning(f"Strategy 4 failed: {e}")
            
        logger.error(f"All JSON parsing strategies failed. Response: {cleaned[:1000]}...")
        return None
        
    except Exception as e:
        logger.error(f"Error in clean_and_parse_json: {e}")
        return None

def display_extracted_data(data: Dict):
    """Display the extracted data in a formatted way with error handling."""
    if not data:
        logger.warning("No data to display")
        return
    
    try:
        metadata = data.get('document_metadata', {})
        entries = data.get('document_content', {}).get('entries', [])
        notes = data.get('extraction_notes', {})
        
        print("\n" + "="*80)
        print("📋 EXTRACTED DATA SUMMARY")
        print("="*80)
        print(f"📄 Document Type: {metadata.get('document_type', 'Unknown')}")
        print(f"📊 Index Type: {metadata.get('index_type', 'Unknown')}")
        print(f"📅 Year: {metadata.get('year', 'Unknown')}")
        print(f"📍 Office: {metadata.get('office_location', 'Unknown')}")
        print(f"📃 Page: {metadata.get('page_number_on_document', 'Unknown')}")
        print(f"🎯 Confidence: {metadata.get('extraction_confidence', 'Unknown')}")
        print(f"📝 Total Entries: {len(entries)}")
        
        # Display extraction notes
        if notes.get('spelling_corrections'):
            print(f"🔧 Spelling Corrections: {len(notes['spelling_corrections'])}")
        if notes.get('ditto_resolutions'):
            print(f"🔄 Ditto Marks Resolved: {len(notes['ditto_resolutions'])}")
        
        if entries:
            print("\n" + "="*80)
            print("📋 DETAILED ENTRIES")
            print("="*80)
            
            for i, entry in enumerate(entries, 1):
                print(f"\n🔹 Entry {i}:")
                details = entry.get('entry_details', {})
                
                print(f"   👤 Name: {details.get('name_of_person', 'N/A')}")
                print(f"   ℹ️  Additional Info: {details.get('additional_information', 'N/A')}")
                print(f"   💼 Transaction Interest: {details.get('interest_of_person_in_transaction', 'N/A')}")
                print(f"   📍 Registered at: {details.get('where_registered', 'N/A')}")
                print(f"   🔢 Serial: {entry.get('serial_number', 'N/A')}")
                print(f"   📚 Volume: {details.get('book_1_volume', 'N/A')}")
                print(f"   📄 Page: {details.get('book_1_page', 'N/A')}")
                
                if entry.get('confidence_notes'):
                    print(f"   ⚠️  Notes: {', '.join(entry.get('confidence_notes', []))}")
        
        print("\n" + "="*80)
        
    except Exception as e:
        logger.error(f"Error displaying data: {e}")

def save_entries_to_excel(data: Dict, excel_path: str) -> bool:
    """Save extracted entries to Excel with error handling."""
    try:
        entries = data.get('document_content', {}).get('entries', [])
        if not entries:
            logger.warning("❌ No entries to save to Excel.")
            return False

        rows = []
        for entry in entries:
            row = {}
            row['serial_number'] = entry.get('serial_number', '')
            details = entry.get('entry_details', {})
            for k, v in details.items():
                row[k] = v
            row['confidence_notes'] = ', '.join(entry.get('confidence_notes', []))
            rows.append(row)

        df = pd.DataFrame(rows)
        df.to_excel(excel_path, index=False)
        logger.info(f"📊 Excel output successfully saved to {excel_path}")
        return True
        
    except Exception as e:
        logger.error(f"❌ Error saving Excel file: {e}")
        return False

def main():
    """Main function with robust error handling."""
    try:
        image_path = "Index I/INDEX_I_&II,_1960_&_1950[1]_pages-to-jpg-0006.jpg"
        output_path = "extracted_data.json"

        print("🚀 West Bengal Index Document Processor")
        print("="*80)

        extracted_data = extract_data_from_image(image_path)

        if extracted_data:
            display_extracted_data(extracted_data)
            
            try:
                json_output = json.dumps(extracted_data, indent=2, ensure_ascii=False)
                with open(output_path, 'w', encoding='utf-8') as f:
                    f.write(json_output)
                logger.info(f"💾 JSON output successfully saved to {output_path}")
            except IOError as e:
                logger.error(f"❌ Error saving JSON file: {e}")
            
            excel_path = "extracted_data.xlsx"
            save_entries_to_excel(extracted_data, excel_path)
        else:
            logger.error("❌ No data was extracted. Please check the image and try again.")
            
    except Exception as e:
        logger.error(f"❌ Unexpected error in main: {e}")

if __name__ == "__main__":
    main()