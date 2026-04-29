import os
import google.generativeai as genai
from dotenv import load_dotenv

# Load API Key from .env
load_dotenv()
api_key = os.getenv("GEMINI_API_KEY")

# Configure Gemini
if api_key:
    genai.configure(api_key=api_key)
    # Using 'gemini-flash-latest' as confirmed by your API test
    model = genai.GenerativeModel('gemini-flash-latest')

def get_ai_interpretation(metrics):
    """
    Sends HRV metrics to Google Gemini and returns a clinical interpretation.
    """
    if not api_key:
        return "⚠️ Gemini API Key not found. Please check your .env file."

    # Construct the clinical prompt
    prompt = f"""
    Act as a Senior Cardiologist and Autonomic Nervous System Expert. 
    Analyze the following patient HRV (Heart Rate Variability) metrics and provide a professional clinical interpretation.
    
    PATIENT METRICS:
    - Heart Rate: {metrics.get('HR', 'N/A')} BPM
    - SDNN (Overall Variability): {metrics.get('SDNN', 'N/A')} ms
    - RMSSD (Vagal Tone): {metrics.get('RMSSD', 'N/A')} ms
    - LF/HF Ratio (Autonomic Balance): {metrics.get('LF_HF', 'N/A')}
    - Sample Entropy (Complexity): {metrics.get('SampEn', 'N/A')}
    - Skewness (Distribution): {metrics.get('Skew', 'N/A')}
    
    Please provide your analysis in the following sections:
    1. Overall Autonomic Status:
    2. Vagal & Sympathetic Insights:
    3. Potential Clinical Risks (if any):
    4. Recommended Physiological Interventions:
    
    Keep the report concise, academic, and highly professional.
    """

    try:
        response = model.generate_content(prompt)
        return response.text
    except Exception as e:
        return f"❌ Gemini AI Analysis Failed: {str(e)}"
