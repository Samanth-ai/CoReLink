# resource_finder_script.py
import os
import logging
import sys
import re
from dotenv import load_dotenv
from typing import List, Dict, Any, Optional, Tuple
import json
import time

# Pydantic for structured output
from pydantic import BaseModel, Field, ValidationError

# LlamaIndex imports
from llama_index.core.tools import FunctionTool
from llama_index.core.program import LLMTextCompletionProgram
from llama_index.core.output_parsers import PydanticOutputParser
from llama_index.core.bridge.pydantic import PrivateAttr

# Use the recommended Google GenAI integration
from llama_index.llms.google_genai import GoogleGenAI

# Tooling specific imports
try: from duckduckgo_search import DDGS
except ImportError: print("duckduckgo-search library not found..."); sys.exit(1)
try: from googleapiclient.discovery import build; Google_Search_available = True
except ImportError: print("google-api-python-client library not found..."); Google_Search_available = False
try: import requests; from bs4 import BeautifulSoup; web_reader_available = True
except ImportError: print("requests or beautifulsoup4 not found..."); web_reader_available = False

# --- Configuration ---
# Load environment variables at the start
load_dotenv()
logging.basicConfig(stream=sys.stdout, level=logging.WARNING) # Keep logging level manageable
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("googleapiclient").setLevel(logging.WARNING) # Suppress Google API client info logs

# --- API Key Checks ---
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
CUSTOM_SEARCH_ENGINE_ID = os.getenv("CUSTOM_SEARCH_ENGINE_ID")

if not GEMINI_API_KEY:
    print("Error: GEMINI_API_KEY not set in .env file.")

if Google_Search_available and (not GOOGLE_API_KEY or not CUSTOM_SEARCH_ENGINE_ID):
    print("Warning: GOOGLE_API_KEY/CSE_ID not set. Google Search disabled.")
    Google_Search_available = False

# --- Pydantic Models (Updated Descriptions) ---
class ResourceInfo(BaseModel):
    """Structured information about a single resource or service provider for patient or caregiver needs."""
    problem_need: str = Field(..., description="The specific patient or caregiver problem/need identified.")
    service_type: str = Field(..., description="Category of service (e.g., 'Caregiver Support', 'Transportation', 'Meal Delivery', 'Home Modification', 'Legal/Financial Planning', 'Language Assistance', 'Prescription Assistance').")
    provider: str = Field(..., description="The name of the organization providing the service.")
    service_description: str = Field(..., description="A brief description of the relevant service offered.")
    contact_info: str = Field(..., description="Contact information (phone/website).")
    service_details: str = Field(..., description="Specific practical details: Cost (Free, Sliding Scale, Private Pay), Insurance Accepted (Medicare, Medicaid), Discounts, Key Eligibility, Language Support (e.g., Spanish available, Bilingual staff).")

class ResourceList(BaseModel):
    """A list of identified resources relevant to patient and caregiver needs from the case study."""
    resources: List[ResourceInfo] = Field(..., description="A list of relevant resources found. Key MUST be 'resources'.")


# --- Tool Definitions (Functions Only - Location Context Re-added) ---
# Global counter for webpage reads to limit requests per run
webpage_read_counter = 0
MAX_WEBPAGE_READS = 150 # Limit per run

def search_web_ddg_func(query: str, location_context: str, num_results: int = 3) -> List[Dict[str, str]]:
    """Searches DuckDuckGo with local focus."""
    search_query = f"{query} near {location_context}"
    print(f"\n--- Searching DDG: {search_query} ---")
    results_list = []
    try:
        with DDGS() as ddgs:
            results_generator = ddgs.text(search_query, region="us-en", max_results=num_results)
            for i, result in enumerate(results_generator):
                if i >= num_results: break
                res = {"title": result.get('title', ''), "href": result.get('href', ''), "body": result.get('body', '')}
                results_list.append(res)
                print(f"  - DDG Result {i+1}: {res['title']} ({res['href']})")
        print("--- DDG Search Complete ---")
    except Exception as e: print(f"Error during DDG search: {e}"); results_list.append({"error": f"Failed search: {e}"})
    return results_list

def search_web_google_func(query: str, location_context: str, num_results: int = 3) -> List[Dict[str, str]]:
    """Searches Google using Custom Search JSON API with local focus."""
    if not Google_Search_available: return [{"error": "Google Search not configured."}]
    search_query = f"{query} in {location_context}"
    print(f"\n--- Searching Google: {search_query} ---")
    results_list = []
    try:
        # Ensure API keys are available here as well
        if not GOOGLE_API_KEY or not CUSTOM_SEARCH_ENGINE_ID:
             return [{"error": "Google Search API Key or CSE ID missing."}]
        service = build("customsearch", "v1", developerKey=GOOGLE_API_KEY)
        res = service.cse().list(q=search_query, cx=CUSTOM_SEARCH_ENGINE_ID, num=num_results).execute()
        items = res.get('items', [])
        for i, item in enumerate(items):
             res_item = {"title": item.get('title', ''), "href": item.get('link', ''), "body": item.get('snippet', '')}
             results_list.append(res_item)
             print(f"  - Google Result {i+1}: {res_item['title']} ({res_item['href']})")
        print("--- Google Search Complete ---")
    except Exception as e: print(f"Error during Google search: {e}"); results_list.append({"error": f"Failed Google search: {e}"})
    return results_list

def read_webpage_func(url: str) -> str:
    """Fetches and extracts the main text content from a given URL."""
    global webpage_read_counter # Use the global counter
    if not web_reader_available: return "Error: Web Reader tool not available."
    if not url or not url.startswith(('http://', 'https://')): return "Error: Invalid URL."

    if webpage_read_counter >= MAX_WEBPAGE_READS:
         print(f"--- Skipping Webpage Read (Limit Reached): {url} ---")
         return "Info: Webpage read skipped due to limit."
    webpage_read_counter += 1

    print(f"\n--- Reading Webpage ({webpage_read_counter}/{MAX_WEBPAGE_READS}): {url} ---")
    try:
        headers = {'User-Agent': 'Mozilla/5.0 (compatible; Googlebot/2.1; +http://www.google.com/bot.html)'}
        # Increased timeout slightly
        response = requests.get(url, headers=headers, timeout=20, allow_redirects=True)
        response.raise_for_status()
        content_type = response.headers.get('content-type', '').lower()
        if 'html' not in content_type: return f"Error: Content not HTML ({content_type})."
        soup = BeautifulSoup(response.text, 'html.parser')
        # Remove common non-content elements
        for element in soup(["script", "style", "nav", "footer", "header", "aside", "form", "button", "iframe", "img", "link", "meta"]):
             element.decompose()
        # Try to find main content areas
        main_content = soup.find('main') or soup.find('article') or soup.find('div', id='content') or soup.find('div', class_='content') or soup.find('div', role='main')
        # Fallback to body if no specific main area found
        text = (main_content or soup.body or soup).get_text(separator=' ', strip=True)
        # Clean up whitespace
        text = re.sub(r'\s{2,}', ' ', text).strip()
        max_length = 6000 # Keep truncation
        if len(text) > max_length: text = text[:max_length] + "... (truncated)"
        print(f"--- Reading Complete (Length: {len(text)}) ---")
        return text if text else "Error: Could not extract meaningful text."
    except requests.exceptions.Timeout: return f"Error: Timeout fetching URL {url}"
    except requests.exceptions.RequestException as e: return f"Error: Could not fetch URL {url} ({type(e).__name__})"
    except Exception as e: return f"Error: Could not process webpage {url} ({type(e).__name__})"


# --- LLM Definition ---
# Define LLM instance globally or pass it into the main function if preferred
llm = None
try:
    if GEMINI_API_KEY:
        model_to_use = "models/gemini-pro" # Ensure Pro model is used
        llm = GoogleGenAI(model_name=model_to_use, api_key=GEMINI_API_KEY)
        print(f"--- Successfully initialized LLM with model: {model_to_use} ---")
    else:
        print("--- Gemini LLM not initialized due to missing API key ---")
except Exception as e:
    print(f"Error initializing Gemini LLM: {e}")
    # Don't exit, let the main function handle the lack of LLM

# --- Step 1: Analysis Prompt ---
analysis_prompt_template = """
Analyze the following Case Study context to identify key information for finding relevant support resources for BOTH the patient and the caregiver.

**Case Study Context:**
{context}

**Tasks:**
1.  **Identify Location:** Determine the primary geographic location (City, County, State, Zip Code if available) where services are needed. If multiple are mentioned, determine the most relevant one for service seeking. If none, state 'Unknown'.
2.  **Identify Insurance Status:** Determine the patient's insurance situation (e.g., "Medicare only", "Medicaid eligible", "Medicare and Medicaid", "Private Insurance", "Uninsured", "Unknown"). Note limitations like "limited income". If none, state 'Unknown'.
3.  **Identify Language:** Note any primary language other than English mentioned for the patient or caregiver (e.g., "Spanish", "None specified"). If none, state 'English'.
4.  **Identify ALL Needs:** List the specific, distinct needs and challenges for BOTH patient and caregiver mentioned or implied in the context. Be comprehensive. Include needs like: Caregiver support (stress, isolation, education, respite, health concerns), Transportation, In-Home Support (personal care, meal prep, dementia behavior management), Meals, Home Environment (safety modifications, ramps), Financial/Benefits Assistance (Medicaid application, affordable meds), Legal/Financial Planning, Language Assistance, Socialization, etc. If context is too vague, state 'Needs unclear from context'.

**Output Format:**
Provide the output strictly as a JSON object with four keys: "location" (string), "insurance_status" (string), "language" (string), and "all_needs" (list of strings).

Example Output:
{{
  "location": "Travis County, TX, 78744",
  "insurance_status": "Medicare only, Limited Income",
  "language": "Spanish",
  "all_needs": [
    "Caregiver: Overwhelmed and isolated",
    "Caregiver: Uncertainty about navigating dementia care",
    "Caregiver: Concern about ability to continue providing care alone",
    "Caregiver: Language barrier (limited English)",
    "Caregiver: Poor physical health",
    "Patient: Transportation from hospital",
    "Patient: Assistance accessing affordable medications",
    "Patient: Home safety modifications/Fall risk",
    "Patient: Personal care support (meals, bathing, behavior)",
    "Patient & Caregiver: Language barrier affecting service access"
  ]
}}
"""

# --- Step 4: Synthesis Program Prompt ---
synthesis_program_prompt_template = """
Synthesize the provided information to generate a structured list of relevant resources for the patient and caregiver, considering location, insurance, language, and specific needs.

**Original Case Study Context:**
{context}

**Identified Location:** {location}
**Patient Insurance Status:** {insurance_status}
**Primary Language (if not English):** {language}
**Identified Needs (Patient & Caregiver):**
{needs_list}

**Collected Information from Search and Web Reading:**
{collected_data}

**Instructions:**
1.  Review all provided information.
2.  For each identified need in the `needs_list`, find the most relevant resources within the `collected_data`. Focus on resources explicitly addressing the need within the specified `location`.
3.  **Prioritize based on Insurance Status:**
    * If "Medicare only" or "Limited Income", prioritize free, low-cost, sliding scale, or Medicare-accepting resources. Note private pay options clearly.
    * If "Medicaid", highlight Medicaid-compatible services or application assistance.
4.  **Prioritize based on Language:**
    * If a `language` (e.g., "Spanish") is specified, prioritize resources offering services in that language or providing interpreters. Explicitly mention language support availability in `service_details`. If language support is unknown after checking, state 'Language support unknown'.
5.  Extract the required information for each relevant resource *accurately* from the `collected_data`. Do not infer information not present.
6.  **Populate `service_details` Thoroughly:** For EACH resource, include details found on: Cost, Insurance Accepted, Key Eligibility, **Language Support**, and Format (online/in-person, etc.), based *only* on information found in the `collected_data`. If specific details (like cost or exact insurance) are not found in the collected text, state 'Details not specified'.
7.  Format the output strictly as a JSON object matching the Pydantic schema provided below. Ensure the JSON is valid.

```json
{{
  "title": "ResourceList",
  "description": "A list of identified resources relevant to patient and caregiver needs from the case study.",
  "type": "object",
  "properties": {{
    "resources": {{
      "title": "Resources",
      "description": "A list of relevant resources found. Key MUST be 'resources'.",
      "type": "array",
      "items": {{
        "title": "ResourceInfo",
        "description": "Structured information about a single resource or service provider for patient or caregiver needs.",
        "type": "object",
        "properties": {{
          "problem_need": {{ "title": "Problem Need", "description": "The specific patient or caregiver problem/need identified.", "type": "string" }},
          "service_type": {{ "title": "Service Type", "description": "Category of service (e.g., 'Caregiver Support', 'Transportation', 'Meal Delivery', 'Home Modification', 'Legal/Financial Planning', 'Language Assistance', 'Prescription Assistance').", "type": "string" }},
          "provider": {{ "title": "Provider", "description": "The name of the organization providing the service.", "type": "string" }},
          "service_description": {{ "title": "Service Description", "description": "A brief description of the relevant service offered.", "type": "string" }},
          "contact_info": {{ "title": "Contact Info", "description": "Contact information (phone/website).", "type": "string" }},
          "service_details": {{ "title": "Service Details", "description": "Specific practical details: Cost (Free, Sliding Scale, Private Pay), Insurance Accepted (Medicare, Medicaid), Discounts, Key Eligibility, Language Support (e.g., Spanish available, Bilingual staff).", "type": "string" }}
        }},
        "required": ["problem_need", "service_type", "provider", "service_description", "contact_info", "service_details"]
      }}
    }}
  }},
  "required": ["resources"]
}}
```
Output ONLY the JSON object adhering to this schema. Do not include any other text before or after the JSON. If no relevant resources matching the criteria are found, output {{\"resources\": []}}.
"""

#--- Main Function to be called by Streamlit ---
def find_resources(context_str: str, progress_callback=None) -> Tuple[Optional[ResourceList], Optional[str]]:
    """
    Analyzes context, searches for resources, and synthesizes results.

    Args:
        context_str: The user-provided context.
        progress_callback: Optional function to report progress updates (e.g., to Streamlit UI).

    Returns:
        A tuple containing:
        - ResourceList object (or None if error/no results).
        - An error message string (or None if successful).
    """
    global webpage_read_counter # Access the global counter
    webpage_read_counter = 0 # Reset counter for each new run

    def report_progress(message):
        print(message) # Log to console
        if progress_callback:
            progress_callback(message) # Send to UI if callback provided

    if not llm:
        return None, "LLM (Gemini) is not initialized. Check GEMINI_API_KEY."
    if not context_str:
        return None, "Input context cannot be empty."

    # --- Step 1: Analyze Context (LLM Call 1) ---
    report_progress("Step 1: Analyzing Context for Location, Insurance, Language, and Needs...")
    analysis_prompt = analysis_prompt_template.format(context=context_str)
    location = "Unknown"
    insurance_status = "Unknown"
    language = "English" # Default language
    all_needs = []
    analysis_error = None

    try:
        analysis_response = llm.complete(analysis_prompt)
        analysis_result_text = analysis_response.text.strip()
        report_progress(f"Analysis Raw Response:\n{analysis_result_text}") # Log raw response

        # Clean potential markdown code fences
        if analysis_result_text.startswith("```json"): analysis_result_text = analysis_result_text[7:-3].strip()
        elif analysis_result_text.startswith("```"): analysis_result_text = analysis_result_text[3:-3].strip()

        try:
            analysis_data = json.loads(analysis_result_text)
            location = analysis_data.get("location", "Unknown")
            insurance_status = analysis_data.get("insurance_status", "Unknown")
            language = analysis_data.get("language", "English")
            all_needs = analysis_data.get("all_needs", [])

            if not isinstance(all_needs, list): # Handle case where LLM returns string instead of list
                report_progress(f"Warning: 'all_needs' was not a list, attempting to fix. Value: {all_needs}")
                if isinstance(all_needs, str) and all_needs.lower() != 'needs unclear from context':
                    # Simple split if it looks like a comma/newline separated list
                    all_needs = [n.strip() for n in re.split(r'[,\n]', all_needs) if n.strip()]
                else:
                    all_needs = [] # Treat as no specific needs identified

            if location == "Unknown" or not all_needs or all_needs == ['Needs unclear from context']:
                 report_progress("Warning: Location or specific needs could not be reliably extracted from the context.")
                 # Decide if we should stop or proceed with broad searches (currently proceeding)

            report_progress(f"  - Identified Location: {location}")
            report_progress(f"  - Identified Insurance Status: {insurance_status}")
            report_progress(f"  - Identified Language: {language}")
            report_progress(f"  - Identified Needs: {all_needs}")

        except (json.JSONDecodeError, ValueError, KeyError) as e:
            analysis_error = f"Error parsing analysis results: {e}. Raw text: '{analysis_result_text}'"
            report_progress(analysis_error)
            # Fallback if parsing fails - maybe try a broader search? For now, return error.
            return None, analysis_error

    except Exception as e:
        analysis_error = f"An error occurred during context analysis LLM call: {e}"
        report_progress(analysis_error)
        logging.error("Context analysis error", exc_info=True)
        return None, analysis_error

    # --- Step 2 & 3: Execute Tools and Collect Data ---
    report_progress("\nStep 2 & 3: Searching for Resources and Reading Webpages...")
    collected_data_text = ""

    if not all_needs or all_needs == ['Needs unclear from context']:
         report_progress("No specific needs identified to search for.")
         # Return empty results gracefully
         return ResourceList(resources=[]), None

    if location == "Unknown":
        report_progress("Location is unknown, cannot perform targeted search.")
        # Return empty results gracefully
        return ResourceList(resources=[]), None

    # Limit the number of needs to search to avoid excessive API calls/time
    MAX_NEEDS_TO_SEARCH = 15
    if len(all_needs) > MAX_NEEDS_TO_SEARCH:
        report_progress(f"Warning: Too many needs identified ({len(all_needs)}). Searching for the first {MAX_NEEDS_TO_SEARCH}.")
        needs_to_search = all_needs[:MAX_NEEDS_TO_SEARCH]
    else:
        needs_to_search = all_needs

    for i, need in enumerate(needs_to_search):
        report_progress(f"\n-- Searching for need {i+1}/{len(needs_to_search)}: '{need}' in location: '{location}' --")
        collected_data_text += f"\n## Resources Found for Need: {need} (Location: {location}, Insurance: {insurance_status}, Language: {language}) ##\n"

        # Clean the need for search query
        clean_need = need.split(":")[-1].strip() # Removes potential "Patient:" or "Caregiver:"

        # --- Formulate Search Queries ---
        queries_to_run = set() # Use set to avoid duplicate queries
        queries_to_run.add(f"{clean_need}") # Base need

        # Add language if not English
        if language.lower() != 'english':
            queries_to_run.add(f"{language}-speaking {clean_need}")
            queries_to_run.add(f"{clean_need} {language} language support")

        # Add low-cost/insurance qualifiers if relevant
        if "limited income" in insurance_status.lower() or "uninsured" in insurance_status.lower():
            queries_to_run.add(f"low cost or free {clean_need}")
            if language.lower() != 'english':
                queries_to_run.add(f"low cost or free {language}-speaking {clean_need}")
        elif "medicaid" in insurance_status.lower():
             queries_to_run.add(f"{clean_need} medicaid accepted")
        elif "medicare" in insurance_status.lower():
             queries_to_run.add(f"{clean_need} medicare accepted")


        report_progress(f"   Running searches for: {list(queries_to_run)}")
        all_search_results = []
        search_delay = 5 # Increased delay between different search queries

        for query in list(queries_to_run): # Iterate over a list copy
            # Add location context to the tool call
            ddg_results = search_web_ddg_func(query=query, location_context=location)
            all_search_results.extend(ddg_results)
            time.sleep(1) # Small delay between engines
            if Google_Search_available:
                google_results = search_web_google_func(query=query, location_context=location)
                all_search_results.extend(google_results)
            time.sleep(search_delay) # Longer delay between query variations

        # --- Process Results & Read Webpages ---
        unique_urls = set()
        processed_results_count = 0

        # De-duplicate results based on URL before processing
        unique_results_for_need = []
        seen_hrefs = set()
        for result in all_search_results:
            # Filter out obvious non-results or errors before processing
            if not isinstance(result, dict): continue
            if "error" in result:
                collected_data_text += f"- Search Error: {result['error']}\n"
                continue
            href = result.get('href')
            # Basic filtering of non-useful URLs (adjust as needed)
            if href and href.startswith('http') and not any(domain in href for domain in ['youtube.com', 'amazon.com', 'facebook.com']):
                 # Normalize URL slightly (remove trailing slash)
                 normalized_href = href.rstrip('/')
                 if normalized_href not in seen_hrefs:
                    unique_results_for_need.append(result)
                    seen_hrefs.add(normalized_href)
            # Optional: Keep results without URLs if snippet is useful?
            # elif not href and result.get('body'):
            #     unique_results_for_need.append(result)

        report_progress(f"   (Found {len(all_search_results)} raw results, {len(unique_results_for_need)} unique/valid URLs to process for '{need}')")

        # Limit webpage reads per need as well
        MAX_READS_PER_NEED = 4
        reads_for_this_need = 0

        for result in unique_results_for_need:
            if reads_for_this_need >= MAX_READS_PER_NEED:
                report_progress("   (Reached max webpage reads for this need)")
                break
            if webpage_read_counter >= MAX_WEBPAGE_READS:
                 report_progress("   (Reached total webpage read limit for this run)")
                 break # Stop reading pages entirely if global limit hit

            title = result.get('title', 'N/A'); href = result.get('href', None); body = result.get('body', '')
            collected_data_text += f"\n* **Result:** {title}\n  * **URL:** {href}\n  * **Snippet:** {body}\n"

            if href: # Already filtered for http start and basic domains
                 # Check unique_urls again (global across needs) to avoid re-reading same page for different needs
                 normalized_href = href.rstrip('/')
                 if normalized_href not in unique_urls:
                     unique_urls.add(normalized_href)
                     page_content = read_webpage_func(href) # This function handles the global counter
                     collected_data_text += f"  * **Webpage Content:** {page_content}\n"
                     processed_results_count += 1
                     reads_for_this_need += 1
                     time.sleep(1) # Delay between page reads
                 else:
                     collected_data_text += f"  * **Webpage Content:** (URL already processed/read in this run)\n"

        if reads_for_this_need == 0 and not any("error" in r for r in all_search_results if isinstance(r, dict)):
             collected_data_text += "- No new webpages read for this need (check limits or search results).\n"
        elif not any(r for r in all_search_results if isinstance(r, dict) and "error" not in r):
             collected_data_text += "- No search results found for this need.\n"


    report_progress("\n--- Data Collection Complete ---")
    if not collected_data_text.strip():
        report_progress("No data was collected from web searches or reading.")
        # Return empty results if nothing was found
        return ResourceList(resources=[]), None

    # --- Step 4: Synthesize and Format (LLM Call 2 - Program) ---
    report_progress("\nStep 4: Synthesizing Results into Structured JSON...")

    # Ensure collected data isn't excessively long for the prompt
    MAX_DATA_LENGTH = 100000 # Adjust based on model context window limits
    if len(collected_data_text) > MAX_DATA_LENGTH:
        report_progress(f"Warning: Collected data is too long ({len(collected_data_text)} chars), truncating to {MAX_DATA_LENGTH}.")
        collected_data_text = collected_data_text[:MAX_DATA_LENGTH] + "\n... (data truncated)"

    program = LLMTextCompletionProgram.from_defaults(
        output_parser=PydanticOutputParser(output_cls=ResourceList),
        prompt_template_str=synthesis_program_prompt_template,
        llm=llm,
        verbose=False, # Set to False to avoid duplicate LlamaIndex logging if using report_progress
    )

    synthesis_output = None
    synthesis_error = None
    try:
        synthesis_output = program(
            context=context_str,
            location=location,
            insurance_status=insurance_status,
            language=language,
            needs_list="\n- ".join(all_needs), # Provide the original full list of needs for context
            collected_data=collected_data_text
        )

        report_progress("\n--- Synthesis Program Executed ---")

        # Validate the output against the Pydantic model
        if isinstance(synthesis_output, ResourceList):
            report_progress("Output successfully validated as ResourceList.")
            return synthesis_output, None
        elif isinstance(synthesis_output, dict):
             report_progress("Output is a dict, attempting Pydantic validation...")
             try:
                 validated_output = ResourceList(**synthesis_output)
                 report_progress("Dict output successfully validated.")
                 return validated_output, None
             except ValidationError as val_err:
                 synthesis_error = f"Pydantic validation failed for dict output: {val_err}. Raw dict: {json.dumps(synthesis_output, indent=2)}"
                 report_progress(synthesis_error)
                 return None, synthesis_error
             except Exception as e:
                 synthesis_error = f"Unexpected error validating dict output: {e}. Raw dict: {json.dumps(synthesis_output, indent=2)}"
                 report_progress(synthesis_error)
                 return None, synthesis_error
        else:
            # Handle cases where the LLM might return a string (e.g., error message or non-JSON)
            synthesis_error = f"Synthesis program returned an unexpected type: {type(synthesis_output)}. Value: {repr(synthesis_output)}"
            report_progress(synthesis_error)
            # Attempt to parse if it looks like JSON string
            if isinstance(synthesis_output, str):
                try:
                    # Clean potential markdown
                    if synthesis_output.strip().startswith("```json"):
                         json_str = synthesis_output.strip()[7:-3].strip()
                    elif synthesis_output.strip().startswith("```"):
                         json_str = synthesis_output.strip()[3:-3].strip()
                    else:
                         json_str = synthesis_output.strip()

                    parsed_dict = json.loads(json_str)
                    validated_output = ResourceList(**parsed_dict)
                    report_progress("Successfully parsed and validated string output.")
                    return validated_output, None
                except (json.JSONDecodeError, ValidationError) as parse_err:
                    synthesis_error += f"\nAttempted to parse as JSON failed: {parse_err}"
                    report_progress(synthesis_error)
                    return None, synthesis_error
                except Exception as e:
                    synthesis_error += f"\nUnexpected error parsing string output: {e}"
                    report_progress(synthesis_error)
                    return None, synthesis_error
            else:
                 return None, synthesis_error


    except Exception as e:
        synthesis_error = f"An error occurred during program execution/synthesis: {e}"
        report_progress(synthesis_error)
        logging.error("Program synthesis error", exc_info=True)
        report_progress("\n--- Collected Data Sent to Synthesis Program (First 2000 chars) ---")
        report_progress(collected_data_text[:2000] + "...")
        return None, synthesis_error


#--- Helper to format results table (Optional - Streamlit will use DataFrame) ---
# Keep this function for potential direct script use or debugging
def format_results_table(output_object: ResourceList):
    """Formats the ResourceList Pydantic object as a Markdown table."""
    if not output_object or not isinstance(output_object, ResourceList) or not output_object.resources:
        return "No specific resources were identified or the output was invalid."

    resources = output_object.resources
    header = "| Problem/Need | Service Type | Provider | Service Description | Contact Info | Service Details (Cost, Insurance, Eligibility, Language) |\n"
    separator = "|---|---|---|---|---|---|\n"
    body = ""
    for res in resources:
        # Basic sanitization for Markdown table
        pn = str(res.problem_need).replace('|', '\\|').replace('\n', ' ')
        st = str(res.service_type).replace('|', '\\|').replace('\n', ' ')
        pv = str(res.provider).replace('|', '\\|').replace('\n', ' ')
        sd = str(res.service_description).replace('|', '\\|').replace('\n', ' ')
        ci = str(res.contact_info).replace('|', '\\|').replace('\n', ' ')
        dt = str(res.service_details).replace('|', '\\|').replace('\n', ' ')
        body += f"| {pn} | {st} | {pv} | {sd} | {ci} | {dt} |\n"
    return header + separator + body


#--- Example Execution Block (for testing the script directly) ---
if __name__ == "__main__":
    print("\n--- AI Dementia Resource Finder (Direct Script Test) ---")

    # Example context (replace with your test case)
    example_context = """
    Patient: 80-year-old male, diagnosed with moderate Alzheimer's disease, living in Austin, Travis County, TX 78744.
    He lives with his 75-year-old wife who is his primary caregiver.
    Insurance: Medicare only. They have a very limited fixed income.
    Needs:
    - Caregiver (wife) is feeling very overwhelmed, isolated, and needs support/respite. She mentioned her own health isn't great.
    - Patient needs transportation to doctor appointments twice a month.
    - Wife is struggling with managing patient's occasional agitation, especially in the evenings. Needs advice/strategies.
    - They need help finding affordable prescription options.
    - Wife mentioned needing grab bars installed in the bathroom for safety.
    - Primary language is English, but wife mentioned neighbor speaks Spanish and sometimes helps translate medical info.
    """
    print("\n--- Using Example Context ---")
    print(example_context)
    print("-----------------------------")

    # Define a simple progress callback for testing
    def simple_progress(msg):
        print(f"[PROGRESS] {msg}")

    # Call the main function
    results_object, error_message = find_resources(example_context, progress_callback=simple_progress)

    print("\n--- Processing Finished ---")

    if error_message:
        print(f"\n--- Error Occurred ---")
        print(error_message)
    elif results_object:
        print("\n--- Formatted Results Table ---")
        print(format_results_table(results_object))

        # Example of accessing data for other uses (like DataFrame conversion)
        if results_object.resources:
            print(f"\n--- Found {len(results_object.resources)} resources (raw data example) ---")
            print(results_object.resources[0].model_dump_json(indent=2)) # Pydantic v2
        else:
            print("\n--- No resources found in the final list. ---")
    else:
        print("\n--- No results object returned and no specific error message. ---")

    print("\n--- Script Test Ended ---")

