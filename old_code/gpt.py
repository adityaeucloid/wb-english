import os
import json
import time
import base64
from openai import OpenAI
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
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

if not OPENAI_API_KEY:
    raise ValueError("OPENAI_API_KEY not found in .env file. Please create a .env file.")

# Initialize OpenAI client
client = OpenAI(api_key=OPENAI_API_KEY)

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
    1. Return ONLY the JSON object. No markdown backticks or additional text.
    2. When you see ditto marks, copy the value from the corresponding field in the entry above
    3. Apply geographical spell-checking for all place names
    4. Be aggressive in interpreting poor handwriting using context
    5. Always fill confidence_notes with any corrections or assumptions made
    """

def encode_image_to_base64(image_path: str) -> str:
    """Encode image to base64 string for OpenAI API."""
    try:
        with open(image_path, "rb") as image_file:
            return base64.b64encode(image_file.read()).decode('utf-8')
    except Exception as e:
        logger.error(f"Error encoding image to base64: {e}")
        raise

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
            
        # Check if it's a valid index type
        index_type = data['document_metadata'].get('index_type')
        if index_type not in ['INDEX_1', 'INDEX_2']:
            logger.error(f"Invalid index_type: {index_type}")
            return False
            
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

def extract_data_from_image_gpt(image_path: str, max_retries: int = 3) -> Optional[Dict]:
    """Enhanced extraction with OpenAI GPT-4 Vision."""
    logger.info(f"📄 Processing image with GPT-4: {image_path}")
    
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
            # Save as JPEG for better compatibility
            temp_path = image_path.replace(os.path.splitext(image_path)[1], '_temp.jpg')
            image.save(temp_path, 'JPEG')
            image_path = temp_path
            
    except Exception as e:
        logger.error(f"❌ Error opening/processing image {image_path}: {e}")
        return None

    # Encode image to base64
    try:
        base64_image = encode_image_to_base64(image_path)
    except Exception as e:
        logger.error(f"❌ Error encoding image: {e}")
        return None

    prompt = build_expert_prompt()
    
    for attempt in range(max_retries):
        try:
            logger.info(f"🤖 GPT-4 API call attempt {attempt + 1}/{max_retries}")
            
            response = client.chat.completions.create(
                model="gpt-4o",  # Updated to the latest GPT-4 Vision model
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "text",
                                "text": prompt
                            },
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": f"data:image/jpeg;base64,{base64_image}",
                                    "detail": "high"
                                }
                            }
                        ]
                    }
                ],
                max_tokens=4096,
                temperature=0.1
            )
            
            if not response.choices:
                logger.warning(f"No choices in response on attempt {attempt + 1}")
                if attempt < max_retries - 1:
                    time.sleep(2 ** attempt)
                    continue
                else:
                    logger.error("No choices returned in final attempt")
                    return None
            
            choice = response.choices[0]
            finish_reason = choice.finish_reason
            
            logger.info(f"Response finish_reason: {finish_reason}")
            
            if finish_reason not in ['stop', 'length']:
                logger.warning(f"Unusual finish_reason: {finish_reason}")
                if finish_reason == 'content_filter':
                    logger.error("Response blocked by content filter")
                
                if attempt < max_retries - 1:
                    time.sleep(2 ** attempt)
                    continue
                else:
                    return None
            
            content = choice.message.content
            if not content:
                logger.warning(f"Empty content on attempt {attempt + 1}")
                if attempt < max_retries - 1:
                    time.sleep(2 ** attempt)
                    continue
                else:
                    logger.error("Empty content in final attempt")
                    return None
            
            # Clean and parse JSON
            data = clean_and_parse_json(content)
            
            if data and validate_extracted_data(data):
                doc_type = data.get('document_metadata', {}).get('index_type', 'UNKNOWN')
                entries_count = len(data.get('document_content', {}).get('entries', []))
                logger.info(f"✅ Successfully extracted {entries_count} entries from {doc_type} document with GPT-4")
                return data
            else:
                logger.warning(f"Invalid data structure on attempt {attempt + 1}")
                if data:
                    logger.error(f"Data structure: {list(data.keys()) if isinstance(data, dict) else type(data)}")
                
                if attempt < max_retries - 1:
                    time.sleep(2 ** attempt)
                    continue
                    
        except Exception as e:
            # Handle different types of OpenAI errors
            error_msg = str(e)
            if "rate limit" in error_msg.lower():
                logger.warning(f"Rate limit exceeded on attempt {attempt + 1}: {e}")
                if attempt < max_retries - 1:
                    time.sleep(60)  # Wait 1 minute for rate limit
                    continue
                else:
                    logger.error("Rate limit exceeded in final attempt")
                    return None
            elif "api" in error_msg.lower():
                logger.error(f"OpenAI API error on attempt {attempt + 1}: {e}")
                if attempt < max_retries - 1:
                    time.sleep(2 ** attempt)
                    continue
                else:
                    return None
            else:
                logger.error(f"❌ Error on attempt {attempt + 1}: {e}")
                if attempt < max_retries - 1:
                    time.sleep(2 ** attempt)
                    continue
    
    logger.error("❌ All retry attempts failed")
    return None

def display_extracted_data_gpt(data: Dict):
    """Display the extracted data in a formatted way with error handling."""
    if not data:
        logger.warning("No data to display")
        return
    
    try:
        metadata = data.get('document_metadata', {})
        entries = data.get('document_content', {}).get('entries', [])
        notes = data.get('extraction_notes', {})
        
        print("\n" + "="*80)
        print("📋 EXTRACTED DATA SUMMARY (GPT-4)")
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

def save_entries_to_excel_gpt(data: Dict, excel_path: str) -> bool:
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
        output_path = "extracted_data_gpt.json"

        print("🚀 West Bengal Index Document Processor (GPT-4)")
        print("="*80)

        extracted_data = extract_data_from_image_gpt(image_path)

        if extracted_data:
            display_extracted_data_gpt(extracted_data)
            
            try:
                json_output = json.dumps(extracted_data, indent=2, ensure_ascii=False)
                with open(output_path, 'w', encoding='utf-8') as f:
                    f.write(json_output)
                logger.info(f"💾 JSON output successfully saved to {output_path}")
            except IOError as e:
                logger.error(f"❌ Error saving JSON file: {e}")
            
            excel_path = "extracted_data_gpt.xlsx"
            save_entries_to_excel_gpt(extracted_data, excel_path)
        else:
            logger.error("❌ No data was extracted. Please check the image and try again.")
            
    except Exception as e:
        logger.error(f"❌ Unexpected error in main: {e}")

if __name__ == "__main__":
    main()