# CoReLink (Community Resource Link )

**AI-Enhanced Dementia Resource Navigator: Connecting Patients and Caregivers with Essential Support Services**

---

## Project Overview

This project uses AI (Google Gemini, LlamaIndex) and web search to help identify local resources for dementia patients and caregivers based on user-provided context.

---

## Setup Instructions

### 1. Prerequisites

- **Python 3.11** must be installed.  
  Check with:
  ```bash
  python3 --version
  
  If not installed, download it from python.org.

- **pip** (Python package manager) should be available.

---

### 2. Create and Activate a Virtual Environment

```bash
python3.11 -m venv venv
source venv/bin/activate
```

---

### 3. Install Dependencies

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

---

### 4. Configure Environment Variables

Create a `.env` file in the project root with the following content:

```
GEMINI_API_KEY=your_gemini_api_key
GOOGLE_API_KEY=your_google_api_key
CUSTOM_SEARCH_ENGINE_ID=your_custom_search_engine_id
```
Replace the values with your actual API keys and search engine ID.

---

#### How to Obtain Google Gemini API Key

1. Go to the [Google AI Studio](https://aistudio.google.com/app/apikey).
2. Sign in with your Google account.
3. Click "Create API Key" and follow the prompts.
4. Copy the generated API key.

Add this value to your `.env` file as `GEMINI_API_KEY=your_gemini_api_key`.

---

#### How to Obtain Google API Key and Custom Search Engine ID

1. **Google API Key**
   - Go to the [Google Cloud Console](https://console.cloud.google.com/).
   - Create or select a project.
   - Enable the "Custom Search API" for your project.
   - Go to "APIs & Services" > "Credentials".
   - Click "Create Credentials" > "API key". Copy the generated key.

2. **Custom Search Engine ID**
   - Visit the [Custom Search Engine](https://cse.google.com/cse/all).
   - Click "Add" to create a new search engine.
   - Set up your search engine (you can use `www.google.com` to search the entire web).
   - After creation, go to "Control Panel" and copy the "Search engine ID".

Add both values to your `.env` file as shown above.

---

### 5. Run the Application

Start the Streamlit app:

```bash
streamlit run app.py
```

The app will open in your browser.  
Follow the on-screen instructions to enter patient/caregiver context and find resources.


---

### 6. Enter Patient Context

Copy sample context from the `input.txt` file and paste it under "Enter Patient Context" in the UI.

---

### 7. Review Masked Context

Review the masked context displayed in the UI.

---

### 8. Find Resources

Click on the "Find Resources" button to display the output.

---

## Notes

- Do **not** share your `.env` file or API keys publicly.
- For best results, provide detailed context in the input area.
- This tool does **not** provide medical, legal, or financial advice.

---

## Troubleshooting

- If you see errors about missing packages, ensure your virtual environment is activated and all dependencies are installed.
- If API keys are missing or incorrect, the app will display a configuration error.

---

## License

For research and educational use only.
```

 
