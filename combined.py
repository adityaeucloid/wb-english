import os
import json
import google.generativeai as genai
from PIL import Image
import pandas as pd  # Add at the top if not already present

# --- Configuration ---
GOOGLE_API_KEY = "AIzaSyCxS-z-pncy8ryzIPJ7N0TLh4dgw-p3PXk"  # <-- Put your API key here
genai.configure(api_key=GOOGLE_API_KEY)

def build_expert_prompt() -> str:
    """
    Builds the advanced, structured Chain-of-Thought prompt for both Index 1 and Index 2 documents.
    """
    return """
    You are an expert document analyst specializing in extracting structured information from handwritten West Bengal government registration records. Your task is to convert the content of this document into a well-structured JSON format.

    IMPORTANT: First determine if this is an INDEX 1 or INDEX 2 document by looking at the header text.

    Follow this step-by-step reasoning process:

    Step 1: Document Type Identification
    - Look at the document header to identify if this is "INDEX No. I" or "INDEX No. II"
    - INDEX 1: Contains information about PERSONS involved in transactions
    - INDEX 2: Contains information about PROPERTIES involved in transactions

    Step 2: Document Analysis
    - Document type: Property registration index register
    - Government department: West Bengal Government
    - Time period: Look for year (like 1960)
    - Purpose: To index registered deeds

    Step 3: Field Extraction Based on Document Type

    FOR INDEX 1 DOCUMENTS (Person-based):
    Columns contain:
    1. Name of person
    2. Father's name, residence, or other particulars for identification
    3. Nature and value of interest in the transaction
    4. Where registered
    5. Serial number
    6. Volume
    7. Page

    FOR INDEX 2 DOCUMENTS (Property-based):
    Columns contain:
    1. Name of property (can be address)
    2. Town (can be town, thana or pargana)
    3. District and sub-district
    4. Nature of transaction (gifted, sale, etc.)
    5. Where registered
    6. Serial number
    7. Volume
    8. Page

    Step 4: Handwriting Interpretation
    - Provide best interpretation for unclear text
    - Mark uncertain readings as [UNCLEAR: possible_text]
    - Note illegible text as [ILLEGIBLE]
    - For "ditto" marks ('do', '""'), inherit value from entry above

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
        "extraction_confidence": "string"
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
            "confidence_notes": []
          }
        ]
      },
      "extraction_notes": {
        "unclear_sections": [],
        "missing_information": [],
        "interpretation_assumptions": []
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
        "extraction_confidence": "string"
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
            "confidence_notes": []
          }
        ]
      },
      "extraction_notes": {
        "unclear_sections": [],
        "missing_information": [],
        "interpretation_assumptions": []
      }
    }
    ```

    FIELD DEFINITIONS:

    For INDEX 1:
    - name_of_person: Primary name in first column
    - additional_information: Father's name, residence, profession combined
    - interest_of_person_in_transaction: Usually 'Vendor' or 'Vendee'
    - where_registered: Registration office location
    - book_1_volume: Volume number
    - book_1_page: Page numbers in original format

    For INDEX 2:
    - name_of_property: Property address or description
    - town: Town, thana, or pargana name
    - district_and_sub_district: District information
    - nature_of_transaction: Type like 'gifted', 'sale', etc.
    - where_registered: Registration office location
    - book_1_volume: Volume number
    - book_1_page: Page numbers in original format

    CRITICAL: Return ONLY the JSON object. No markdown backticks or additional text.
    """

def extract_data_from_image(image_path: str) -> dict | None:
    """
    Uses Gemini 2.5 Pro with an expert prompt to extract structured data from both Index 1 and Index 2 documents.

    Args:
        image_path: The file path to the image.

    Returns:
        A dictionary containing the structured data, or None on error.
    """
    print(f"📄 Processing image: {image_path}")

    model = genai.GenerativeModel('gemini-2.5-pro')
    
    try:
        image = Image.open(image_path)
    except FileNotFoundError:
        print(f"❌ Error: Image file not found at '{image_path}'")
        return None

    prompt = build_expert_prompt()
    
    try:
        print("🤖 Calling Gemini API... (This may take a moment)")
        response = model.generate_content([prompt, image])
        
        # Clean and parse the JSON response
        cleaned_response = response.text.strip()
        if cleaned_response.startswith("```json"):
            cleaned_response = cleaned_response[7:]
        if cleaned_response.endswith("```"):
            cleaned_response = cleaned_response[:-3]
        cleaned_response = cleaned_response.strip()

        data = json.loads(cleaned_response)
        
        # Identify document type from response
        doc_type = data.get('document_metadata', {}).get('index_type', 'UNKNOWN')
        print(f"✅ Successfully extracted data from {doc_type} document")
        
        return data

    except json.JSONDecodeError as e:
        print(f"❌ JSON Parsing Error: The model output was not valid JSON. {e}")
        print("\n--- Raw Gemini Response ---")
        if 'response' in locals():
            print(response.text[:500] + "..." if len(response.text) > 500 else response.text)
        print("---------------------------\n")
        return None
    except Exception as e:
        print(f"❌ An error occurred during API call or processing: {e}")
        return None

def display_extracted_data(data: dict):
    """Display the extracted data in a formatted way"""
    if not data:
        return
    
    metadata = data.get('document_metadata', {})
    entries = data.get('document_content', {}).get('entries', [])
    
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
    
    if entries:
        print("\n" + "="*80)
        print("📋 DETAILED ENTRIES")
        print("="*80)
        
        for i, entry in enumerate(entries, 1):
            print(f"\n🔹 Entry {i}:")
            details = entry.get('entry_details', {})
            
            if metadata.get('index_type') == 'INDEX_1':
                print(f"   👤 Name: {details.get('name_of_person', 'N/A')}")
                print(f"   ℹ️  Additional Info: {details.get('additional_information', 'N/A')}")
                print(f"   💼 Transaction Interest: {details.get('interest_of_person_in_transaction', 'N/A')}")
            
            elif metadata.get('index_type') == 'INDEX_2':
                print(f"   🏠 Property: {details.get('name_of_property', 'N/A')}")
                print(f"   🏘️  Town: {details.get('town', 'N/A')}")
                print(f"   🗺️  District: {details.get('district_and_sub_district', 'N/A')}")
                print(f"   📋 Transaction Type: {details.get('nature_of_transaction', 'N/A')}")
            
            print(f"   📍 Registered at: {details.get('where_registered', 'N/A')}")
            print(f"   🔢 Serial: {entry.get('serial_number', 'N/A')}")
            print(f"   📚 Volume: {details.get('book_1_volume', 'N/A')}")
            print(f"   📄 Page: {details.get('book_1_page', 'N/A')}")
            
            if entry.get('confidence_notes'):
                print(f"   ⚠️  Notes: {', '.join(entry.get('confidence_notes', []))}")
    
    print("\n" + "="*80)

def save_entries_to_excel(data: dict, excel_path: str):
    """Save extracted entries to an Excel file in a structured manner."""
    entries = data.get('document_content', {}).get('entries', [])
    if not entries:
        print("❌ No entries to save to Excel.")
        return

    # Flatten entry_details for each entry
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
    print(f"📊 Excel output successfully saved to {excel_path}")

def main():
    """Main function to run the script."""
    # Image path - update this to your image
    image_path = "Index II\\INDEX_I_&II,_1960_&_1950[1]_pages-to-jpg-0001.jpg"
    output_path = "extracted_data.json"

    print("🚀 West Bengal Index Document Processor")
    print("Supports both Index 1 (Person-based) and Index 2 (Property-based) documents")
    print("="*80)

    extracted_data = extract_data_from_image(image_path)

    if extracted_data:
        # Display the extracted data in terminal
        display_extracted_data(extracted_data)
        
        # Save to JSON file
        json_output = json.dumps(extracted_data, indent=2, ensure_ascii=False)
        try:
            with open(output_path, 'w', encoding='utf-8') as f:
                f.write(json_output)
            print(f"\n💾 JSON output successfully saved to {output_path}")
        except IOError as e:
            print(f"❌ Error saving file to {output_path}: {e}")
        
        # Save to Excel file
        excel_path = "extracted_data.xlsx"
        save_entries_to_excel(extracted_data, excel_path)
    else:
        print("❌ No data was extracted. Please check the image and try again.")

if __name__ == "__main__":
    main()