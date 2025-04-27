# app.py
import streamlit as st
import pandas as pd
import re
import io
import os # To check for API key existence

# Import the main function and Pydantic models from your script
# Ensure resource_finder_script.py is in the same directory or accessible via PYTHONPATH
try:
    from resource_finder_script import find_resources, ResourceList, ResourceInfo, GEMINI_API_KEY
except ImportError as e:
    st.error(f"Error importing 'resource_finder_script.py'. Make sure it's in the same directory. Details: {e}")
    st.stop() # Stop execution if the backend script can't be imported

# --- Basic PHI Masking Function ---
# WARNING: This is a very basic regex-based masker and is NOT reliable for real PHI.
# It's prone to false positives and false negatives. Use with extreme caution.
# A robust solution requires NLP models (e.g., spaCy, Presidio) or dedicated services.
def mask_phi(text):
    """Attempts to mask potential PHI using basic regex. Highly experimental."""
    # Mask potential names (simple pattern: consecutive capitalized words, at least two)
    text = re.sub(r'\b([A-Z][a-z]+(?:\s+[A-Z][a-z]+)+)\b', '[NAME MASKED]', text)
    # Mask potential phone numbers (various common North American formats)
    text = re.sub(r'\b(\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4})\b', '[PHONE MASKED]', text)
    # Mask potential simple addresses (Number Street Name Suffix) - Very basic
    #text = re.sub(r'\b(\d+\s+[A-Z][a-zA-Z]+\s+(?:Street|St|Avenue|Ave|Road|Rd|Lane|Ln|Drive|Dr|Court|Ct|Boulevard|Blvd))\b', '[ADDRESS MASKED]', text)
    # Mask potential email addresses
    text = re.sub(r'\b([A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,})\b', '[EMAIL MASKED]', text)
    # Mask potential SSN-like numbers
    text = re.sub(r'\b(\d{3}-\d{2}-\d{4})\b', '[SSN MASKED]', text)
    # Mask potential Zip codes (5 digit or 5+4)
    #text = re.sub(r'\b(\d{5}(?:-\d{4})?)\b', '[ZIP MASKED]', text)
    return text

# --- Function to convert ResourceList to Excel ---
def convert_to_excel(resource_list_obj: ResourceList):
    """Converts the ResourceList Pydantic object to an Excel file in memory."""
    if not resource_list_obj or not resource_list_obj.resources:
        return None

    try:
        # Convert list of Pydantic objects to list of dicts
        data_for_df = [res.model_dump() for res in resource_list_obj.resources] # Use model_dump() for Pydantic v2
        df = pd.DataFrame(data_for_df)

        # Reorder and rename columns for better readability
        column_map = {
            'problem_need': 'Problem/Need',
            'service_type': 'Service Type',
            'provider': 'Provider',
            'service_description': 'Service Description',
            'contact_info': 'Contact Info',
            'service_details': 'Service Details (Cost, Insurance, Eligibility, Language)'
        }
        # Ensure only expected columns are present and ordered correctly
        df = df[list(column_map.keys())]
        df = df.rename(columns=column_map)

        output = io.BytesIO()
        with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
            df.to_excel(writer, index=False, sheet_name='Resources')
            # Auto-adjust column widths for better readability
            worksheet = writer.sheets['Resources']
            for idx, col in enumerate(df):
                series = df[col]
                max_len = max((
                    series.astype(str).map(len).max(),
                    len(str(series.name)) # Header length
                )) + 2 # Add a little padding
                worksheet.set_column(idx, idx, max_len)

        processed_data = output.getvalue()
        return processed_data
    except Exception as e:
        st.error(f"Error creating Excel file: {e}")
        return None

# --- Streamlit UI Setup ---
st.set_page_config(layout="wide") # Use wider layout for table display
st.title("AI-Driven Navigation Support for Persons Living with Dementia and Their Caregivers (CoReLink)")

# --- Disclaimer and Warnings ---
st.warning(
    """
    **Disclaimer:** This tool uses AI (including Large Language Models and web searches) to identify potential resources based on the context provided.
    - **Information Accuracy:** The information is gathered from public web sources and AI interpretation, and may not always be accurate, complete, or up-to-date. **Always verify information directly with the resource provider.**
    - **Not Medical/Legal Advice:** This tool does not provide medical, legal, or financial advice. Consult qualified professionals for specific advice.
    - **PHI Masking:** A basic attempt is made to mask potential Protected Health Information (PHI) like names, phones, SSN before processing. **This masking is NOT guaranteed to be effective.** Do NOT enter highly sensitive details if you are concerned about privacy. Review the 'Masked Context' below before proceeding if unsure.
    """
)

# Check for API Key and display warning if missing
if not GEMINI_API_KEY:
    st.error("🔴 **Configuration Error:** The `GEMINI_API_KEY` is missing. The core AI functionality will not work. Please set it in your `.env` file and restart Streamlit.")
    # Optionally disable the input/button if the key is missing
    # st.stop()

# --- Input Area ---
st.header("1. Enter Patient Context")
st.caption("Provide details about the patient's situation, location (City, State/County), insurance, language, and specific needs for both patient and caregiver. More detail yields better results.")
context_input = st.text_area("Paste or type context here:", height=250, key="context_input_area")

# --- PHI Masking Option and Display ---
st.header("2. Review Masked Context (Optional)")
st.caption("This is the context that will be sent for analysis after basic PHI masking. Review it to ensure sensitive information is adequately masked according to your comfort level.")

masked_context = ""
if context_input:
    masked_context = mask_phi(context_input)
    st.text_area("Masked Context (for review):", value=masked_context, height=150, key="masked_context_display", disabled=True)
else:
    st.info("Enter context above to see the masked version here.")

# --- Processing Trigger ---
st.header("3. Find Resources")
find_button = st.button("Find Resources", key="find_button", type="primary", disabled=(not context_input or not GEMINI_API_KEY))

# --- Results Area ---
st.header("4. Results")

# Use session state to store results and avoid re-running on widget interactions
if 'results_object' not in st.session_state:
    st.session_state.results_object = None
if 'error_message' not in st.session_state:
    st.session_state.error_message = None
if 'run_complete' not in st.session_state:
    st.session_state.run_complete = False

# --- Progress Updates ---
# Create a placeholder for progress messages
progress_placeholder = st.empty()

def display_progress(message):
    # Use markdown to allow for some formatting if needed
    progress_placeholder.info(f"⏳ {message}")

# --- Main Logic ---
if find_button and masked_context:
    st.session_state.results_object = None # Clear previous results
    st.session_state.error_message = None
    st.session_state.run_complete = False
    with st.spinner("Analyzing context, searching the web, and synthesizing results... This may take a few minutes."):
        try:
            # Call the backend function, passing the progress callback
            results_obj, error_msg = find_resources(masked_context, progress_callback=display_progress)
            st.session_state.results_object = results_obj
            st.session_state.error_message = error_msg
            st.session_state.run_complete = True
            progress_placeholder.empty() # Clear progress messages on completion
        except Exception as e:
            st.session_state.error_message = f"An unexpected error occurred in the Streamlit app: {e}"
            st.session_state.results_object = None
            st.session_state.run_complete = True
            progress_placeholder.empty() # Clear progress messages on error
            st.exception(e) # Show detailed traceback in Streamlit

# --- Display Results or Errors ---
if st.session_state.run_complete:
    if st.session_state.error_message:
        st.error(f"An error occurred: {st.session_state.error_message}")
        st.info("Please check the console where Streamlit is running for more detailed logs from the backend script.")
    elif st.session_state.results_object and st.session_state.results_object.resources:
        results_obj = st.session_state.results_object
        st.success(f"Found {len(results_obj.resources)} potential resources.")

        # Convert to DataFrame for display and download
        try:
            data_for_df = [res.model_dump() for res in results_obj.resources]
            df = pd.DataFrame(data_for_df)
             # Reorder and rename columns for display
            column_map = {
                'problem_need': 'Problem/Need',
                'service_type': 'Service Type',
                'provider': 'Provider',
                'service_description': 'Service Description',
                'contact_info': 'Contact Info',
                'service_details': 'Service Details (Cost, Insurance, Eligibility, Language)'
            }
            # Filter to expected columns and handle potential missing ones gracefully
            display_columns = [col for col in column_map.keys() if col in df.columns]
            df_display = df[display_columns].rename(columns=column_map)

            st.dataframe(df_display, use_container_width=True) # Display as interactive table

            # --- Excel Download Button ---
            excel_data = convert_to_excel(results_obj)
            if excel_data:
                st.download_button(
                    label="📥 Download Results as Excel",
                    data=excel_data,
                    file_name="resource_finder_results.xlsx",
                    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                    key="download_button"
                )
            else:
                st.warning("Could not generate Excel file for download.")

        except Exception as e:
            st.error(f"Error displaying results or creating download link: {e}")
            # Display raw data if DataFrame conversion fails
            st.subheader("Raw Results Data (JSON)")
            st.json(results_obj.model_dump_json(indent=2))

    elif st.session_state.results_object and not st.session_state.results_object.resources:
        st.info("The process completed, but no specific resources were identified based on the provided context and web search.")
    else:
        # This case might occur if find_resources returns (None, None) unexpectedly
        st.warning("Processing finished, but no results or error message were returned.")

elif not find_button and not st.session_state.run_complete:
    st.info("Enter context and click 'Find Resources' to begin.")


# Add footer or additional info if needed
st.markdown("---")
st.caption("Developed using Streamlit, LlamaIndex, and Google Gemini.")
