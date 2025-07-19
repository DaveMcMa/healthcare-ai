import os
import gradio as gr
import logging
import sys
import requests
import urllib3
import json
import base64
import pymysql
from datetime import datetime

hpe_theme = gr.themes.Soft(
    primary_hue="emerald",
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger(__name__)

# Disable SSL warnings for self-signed certificates
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

# Default configurations
DEFAULTS = {
    "medreason": {
        "url": os.getenv("MEDREASON_URL", ""),
        "token": os.getenv("MEDREASON_TOKEN", "")
    },
    "whisper": {
        "url": os.getenv("WHISPER_URL", ""),
        "token": os.getenv("WHISPER_TOKEN", "")
    },
    "nllb": {
        "url": os.getenv("NLLB_URL", ""),
        "token": os.getenv("NLLB_TOKEN", "")
    },
    "medgemma": {
        "url": os.getenv("MEDGEMMA_URL", ""),
        "token": os.getenv("MEDGEMMA_TOKEN", "")
    }
}

DB_CONFIG = {
    'host': os.getenv('DB_HOST', ''),
    'user': os.getenv('DB_USER', ''),
    'password': os.getenv('DB_PASSWORD', ''),
    'database': os.getenv('DB_NAME', '')
}

def check_model(model_type, url, token):
    """Generic health check for model APIs"""
    try:
        headers = {
            "Authorization": f"Bearer {token}",
            "Content-Type": "application/json"
        }
        
        # Different payload structures for different model types
        if model_type == "medreason":
            payload = {
                "model": "UCSC-VLAA/MedReason-8B",
                "prompt": "Hello",
                "max_tokens": 10
            }
        elif model_type == "whisper":
            payload = {"model": "openai/whisper-large-v3"}
        elif model_type == "nllb":
            payload = {
                "instances": [{
                    "text": "hello",
                    "source_language": "english",
                    "target_language": "french"
                }]
            }
        elif model_type == "medgemma":
            # Use the specialized MedGemma check function
            return check_medgemma_model(url, token)
        
        logger.info(f"Checking {model_type} at: {url}")
        response = requests.post(
            url, 
            headers=headers, 
            data=json.dumps(payload), 
            verify=False,
            timeout=10
        )
        
        # Different success conditions for different models
        if model_type == "whisper" and response.status_code == 400:
            return True, f"{model_type.capitalize()} API is available (expected 400 error for test)"
        elif response.status_code == 200:
            return True, f"{model_type.capitalize()} API is available and responding"
        else:
            return False, f"{model_type.capitalize()} API returned status code: {response.status_code}"
            
    except Exception as e:
        return False, f"{model_type.capitalize()} API error: {str(e)}"

def encode_image_to_base64(image_path):
    """Convert local image to base64 string"""
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')

def check_medgemma_model(url, token):
    """Health check specifically for MedGemma model"""
    try:
        headers = {
            "Authorization": f"Bearer {token}",
            "Content-Type": "application/json"
        }
        
        # Simple test payload for MedGemma
        payload = {
            "model": "google/medgemma-4b-it",
            "messages": [
                {
                    "role": "system",
                    "content": "You are an expert radiologist."
                },
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": "Hello, can you help with medical imaging analysis?"
                        }
                    ]
                }
            ],
            "max_tokens": 10,
            "temperature": 0.1
        }
        
        logger.info(f"Checking MedGemma at: {url}/v1/chat/completions")
        response = requests.post(
            f"{url}/v1/chat/completions", 
            headers=headers, 
            data=json.dumps(payload), 
            verify=False,
            timeout=10
        )
        
        if response.status_code == 200:
            return True, "MedGemma API is available and responding"
        else:
            return False, f"MedGemma API returned status code: {response.status_code}"
            
    except Exception as e:
        return False, f"MedGemma API error: {str(e)}"

def analyze_xray_with_medgemma(image_path, medgemma_url, medgemma_token):
    """Analyze X-ray image using MedGemma"""
    try:
        # Convert image to base64
        base64_image = encode_image_to_base64(image_path)
        
        headers = {
            "Authorization": f"Bearer {medgemma_token}",
            "Content-Type": "application/json"
        }
        
        # Prepare the request payload
        payload = {
            "model": "google/medgemma-4b-it",
            "messages": [
                {
                    "role": "system",
                    "content": "You are an expert radiologist. Analyze the provided X-ray image and provide a detailed medical assessment."
                },
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": "Please analyze this X-ray image. Describe any abnormalities, potential diagnoses, and recommendations for further evaluation if needed. Provide a structured analysis including: 1) Image quality assessment, 2) Anatomical structures visible, 3) Abnormal findings (if any), 4) Differential diagnoses, 5) Recommendations."
                        },
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/png;base64,{base64_image}"
                            }
                        }
                    ]
                }
            ],
            "max_tokens": 1000,
            "temperature": 0.1
        }
        
        logger.info(f"Sending X-ray analysis request to MedGemma")
        response = requests.post(
            f"{medgemma_url}/v1/chat/completions",
            headers=headers,
            data=json.dumps(payload),
            verify=False,
            timeout=120
        )
        
        if response.status_code == 200:
            result = response.json()
            if 'choices' in result and len(result['choices']) > 0:
                analysis = result['choices'][0]['message']['content']
                return analysis
            else:
                return f"Unexpected response format: {result}"
        else:
            logger.error(f"MedGemma API error: {response.status_code}, {response.text}")
            return f"Error: MedGemma API returned status code {response.status_code}. Response: {response.text}"
            
    except Exception as e:
        logger.error(f"Error analyzing X-ray with MedGemma: {str(e)}")
        return f"Error: {str(e)}"


def save_diagnosis_to_db(json_data):
    """Save diagnosis data to MySQL database"""
    if not json_data or json_data == "No JSON found in response" or json_data == "Invalid JSON format":
        return False, "No valid data to save to database."
    
    try:
        # Parse the JSON string
        try:
            diagnosis_data = json.loads(json_data)
        except json.JSONDecodeError:
            return False, "Failed to parse JSON data."
        
        # Connect to the database
        connection = pymysql.connect(**DB_CONFIG)
        cursor = connection.cursor()
        
        # Validate and clean data
        patient_name = diagnosis_data.get("patient_name", "N/A")
        
        # Handle date_of_birth - ensure YYYY-MM-DD format
        date_of_birth = diagnosis_data.get("date_of_birth", "N/A")
        if date_of_birth == "N/A" or not date_of_birth:
            date_of_birth = "N/A"
        else:
            try:
                # Try to parse the date to validate format
                datetime.strptime(date_of_birth, "%Y-%m-%d")
            except ValueError:
                date_of_birth = "N/A"
        
        # Handle visit_time - ensure YYYY-MM-DD HH:MM:SS format
        visit_time = diagnosis_data.get("visit_time", "N/A")
        if visit_time == "N/A" or not visit_time:
            visit_time = "N/A"
        else:
            try:
                # First, replace T with space if it's in ISO format
                if "T" in visit_time:
                    visit_time = visit_time.replace("T", " ")
                
                # Check if this is just a date (YYYY-MM-DD) without time
                if len(visit_time.strip()) == 10 and visit_time.count("-") == 2:
                    # Add a default time (00:00:00)
                    visit_time = f"{visit_time} 00:00:00"
                    
                # Try to parse the date to validate format
                datetime.strptime(visit_time, "%Y-%m-%d %H:%M:%S")
            except ValueError:
                visit_time = "N/A"
        
        # Get remaining fields with default value "N/A" if missing
        severity = diagnosis_data.get("severity", "N/A")
        primary_diagnosis = diagnosis_data.get("primary_diagnosis", "N/A")
        secondary_diagnoses = diagnosis_data.get("secondary_diagnoses", "N/A")
        recommended_tests = diagnosis_data.get("recommended_tests", "N/A")
        recommended_treatment = diagnosis_data.get("recommended_treatment", "N/A")
        follow_up = diagnosis_data.get("follow_up", "N/A")
        
        # Prepare SQL query - Note we're not including medical_reasoning as per requirements
        sql = """
        INSERT INTO triage (
            patient_name, date_of_birth, visit_time, severity, 
            primary_diagnosis, secondary_diagnoses, recommended_tests, 
            recommended_treatment, follow_up
        ) VALUES (
            %s, %s, %s, %s, %s, %s, %s, %s, %s
        )
        """
        
        # Use special handling for dates
        cursor.execute(sql, (
            patient_name,
            None if date_of_birth == "N/A" else date_of_birth,
            None if visit_time == "N/A" else visit_time,
            severity,
            primary_diagnosis,
            secondary_diagnoses,
            recommended_tests,
            recommended_treatment,
            follow_up
        ))
        
        # Commit the transaction
        connection.commit()
        
        # Get the ID of the inserted record
        record_id = cursor.lastrowid
        
        # Close the connection
        cursor.close()
        connection.close()
        
        return True, f"Diagnosis saved successfully to database with ID: {record_id}"
    
    except pymysql.MySQLError as e:
        logger.error(f"Database error: {str(e)}")
        return False, f"Database error: {str(e)}"
    except Exception as e:
        logger.error(f"Error saving to database: {str(e)}")
        return False, f"Error: {str(e)}"

def save_diagnosis_to_db_button(json_data):
    """Handle save to database button click"""
    if not json_data:
        return "No data to save. Please run a diagnosis first."
    
    success, message = save_diagnosis_to_db(json_data)
    return message

def get_language_code(language_name):
    """Convert language name to code used by NLLB model"""
    # Normalize language name to handle different formats from Whisper
    normalized_name = language_name.lower() if language_name else "english"
    
    # Handle ISO codes that might be returned directly
    if normalized_name in ["bg", "bulgarian"]:
        return "bulgarian"
    elif normalized_name in ["en", "english"]:
        return "english"
    elif normalized_name in ["de", "german"]:
        return "german"
    elif normalized_name in ["pl", "polish"]:
        return "polish"
    elif normalized_name in ["cs", "czech"]:
        return "czech"
    elif normalized_name in ["sk", "slovak"]:
        return "slovak"
    elif normalized_name in ["uk", "ukrainian"]:
        return "ukrainian"
    elif normalized_name in ["fi", "finnish"]:
        return "finnish"
    
    # Default to English if no match
    return "english"

def get_iso_language_code(language_name):
    """Convert language name to ISO code for Whisper API"""
    iso_language_map = {
        "English": "en",
        "German": "de",
        "Polish": "pl",
        "Czech": "cs",
        "Slovak": "sk",
        "Ukrainian": "uk",
        "Bulgarian": "bg",
        "Finnish": "fi"
    }
    return iso_language_map.get(language_name, "en")

def normalize_language_name(language_code):
    """Convert various language codes/names to standard language names"""
    code_map = {
        "bg": "Bulgarian",
        "en": "English",
        "de": "German", 
        "pl": "Polish",
        "cs": "Czech",
        "sk": "Slovak",
        "uk": "Ukrainian",
        "fi": "Finnish",
        "bulgarian": "Bulgarian",
        "english": "English",
        "german": "German",
        "polish": "Polish", 
        "czech": "Czech",
        "slovak": "Slovak",
        "ukrainian": "Ukrainian",
        "finnish": "Finnish"
    }
    
    # Try to normalize the input
    normalized = language_code.lower() if language_code else "en"
    return code_map.get(normalized, "Unknown")

def transcribe_audio(audio_path, auto_detect, expected_language, whisper_url, whisper_token):
    """Transcribe audio using Whisper API"""
    try:
        headers = {
            "Authorization": f"Bearer {whisper_token}"
        }
        
        # Create form data with audio file - Fixed model reference
        files = {
            'file': open(audio_path, 'rb'),
            'model': (None, 'openai/whisper-large-v3')  # Fixed model reference
        }
        
        # Add language if not auto-detecting
        if not auto_detect:
            # Use ISO language code instead of full name
            iso_lang_code = get_iso_language_code(expected_language)
            files['language'] = (None, iso_lang_code)
            logger.info(f"Using specified language: {expected_language} (code: {iso_lang_code})")
        
        # Request transcription
        response = requests.post(
            whisper_url,
            headers=headers,
            files=files,
            verify=False,
            timeout=300  # Increase timeout for longer audio files
        )
        
        if response.status_code == 200:
            result = response.json()
            transcription = result.get('text', '')
            
            # Get the detected language and normalize it
            detected_language = result.get('language', 'Unknown')
            logger.info(f"Raw detected language from Whisper: {detected_language}")
            
            # Normalize the language name for display
            normalized_language = normalize_language_name(detected_language)
            
            return normalized_language, transcription
        else:
            logger.error(f"Whisper API error: {response.status_code}, {response.text}")
            return "Unknown", f"Transcription failed: API returned status {response.status_code}, message: {response.text}"
            
    except Exception as e:
        logger.error(f"Error transcribing audio: {str(e)}")
        return "Unknown", f"Error: {str(e)}"
    finally:
        # Close the file properly
        try:
            files['file'].close()
        except:
            pass

def translate_text(text, source_lang, target_lang, nllb_url, nllb_token):
    """Translate text using NLLB API"""
    try:
        headers = {
            "Authorization": f"Bearer {nllb_token}",
            "Content-Type": "application/json"
        }
        
        logger.info(f"Translating from {source_lang} to {target_lang}")
        
        payload = {
            "instances": [
                {
                    "text": text,
                    "source_language": source_lang,
                    "target_language": target_lang
                }
            ]
        }
        
        response = requests.post(
            nllb_url, 
            headers=headers, 
            data=json.dumps(payload), 
            verify=False,
            timeout=300
        )
        
        if response.status_code == 200:
            result = response.json()
            logger.info(f"Translation response format: {type(result)}")
            
            # Extract just the translated text from the response
            if "predictions" in result:
                if isinstance(result["predictions"][0], dict) and "translated_text" in result["predictions"][0]:
                    return result["predictions"][0]["translated_text"]
                else:
                    return result["predictions"][0]
            elif "outputs" in result:
                if isinstance(result["outputs"][0], dict) and "translated_text" in result["outputs"][0]:
                    return result["outputs"][0]["translated_text"]
                else:
                    return result["outputs"][0]
            else:
                # Try to extract from any top-level field that seems to have the translation
                for key, value in result.items():
                    if isinstance(value, list) and len(value) > 0:
                        if isinstance(value[0], str):
                            return value[0]
                        elif isinstance(value[0], dict) and "translated_text" in value[0]:
                            return value[0]["translated_text"]
                        elif isinstance(value[0], dict) and "translation" in value[0]:
                            return value[0]["translation"]
                
                # If we got a direct dictionary response with translated_text
                if isinstance(result, dict) and "translated_text" in result:
                    return result["translated_text"]
                
                return str(result)  # Return the whole result if we can't extract the translation
        else:
            logger.error(f"NLLB API error: {response.status_code}, {response.text}")
            return f"Translation failed: API returned status {response.status_code}"
            
    except Exception as e:
        logger.error(f"Error translating text: {str(e)}")
        return f"Error: {str(e)}"


def parse_medreason_response(result):
    """Parse MedReason response into structured sections"""
    response_text = ""
    
    # Extract text from different possible response formats
    if "choices" in result and len(result["choices"]) > 0:
        if "text" in result["choices"][0]:
            response_text = result["choices"][0]["text"]
        elif "message" in result["choices"][0]:
            response_text = result["choices"][0]["message"]["content"]
        else:
            response_text = json.dumps(result["choices"][0], indent=2)
    elif "response" in result:
        response_text = result["response"]
    elif "generations" in result:
        response_text = result["generations"][0]["text"]
    else:
        response_text = json.dumps(result, indent=2)
    
    # Extract sections from the response
    sections = {
        "thinking": "",
        "reasoning": "",
        "conclusion": "",
        "final_answer": ""
    }
    
    # Check for ## Thinking section
    if "## Thinking" in response_text:
        thinking_section = response_text.split("## Thinking")[1]
        if "## Final Answer" in thinking_section:
            sections["thinking"] = thinking_section.split("## Final Answer")[0].strip()
        elif "## Triage Summary" in thinking_section:
            sections["thinking"] = thinking_section.split("## Triage Summary")[0].strip()
        else:
            sections["thinking"] = thinking_section.strip()
    
    # Check for ### Reasoning Process section
    if "### Reasoning Process" in response_text:
        reasoning_section = response_text.split("### Reasoning Process")[1]
        if "---" in reasoning_section:
            sections["reasoning"] = reasoning_section.split("---")[0].strip()
        elif "### Conclusion" in reasoning_section:
            sections["reasoning"] = reasoning_section.split("### Conclusion")[0].strip()
        else:
            sections["reasoning"] = reasoning_section.strip()
    
    # Check for ### Conclusion section
    if "### Conclusion" in response_text:
        conclusion_section = response_text.split("### Conclusion")[1]
        if "## Final Answer" in conclusion_section:
            sections["conclusion"] = conclusion_section.split("## Final Answer")[0].strip()
        elif "## Triage Summary" in conclusion_section:
            sections["conclusion"] = conclusion_section.split("## Triage Summary")[0].strip()
        else:
            sections["conclusion"] = conclusion_section.strip()
    
    # Check for ## Final Answer or ## Triage Summary section
    if "## Triage Summary" in response_text:
        sections["final_answer"] = response_text.split("## Triage Summary")[1].strip()
    elif "## Final Answer" in response_text:
        sections["final_answer"] = response_text.split("## Final Answer")[1].strip()
    
    # Try to extract JSON from the final answer
    try:
        if "{" in sections["final_answer"] and "}" in sections["final_answer"]:
            start = sections["final_answer"].find("{")
            end = sections["final_answer"].rfind("}") + 1
            json_str = sections["final_answer"][start:end]
            
            # Format the JSON nicer if possible
            try:
                json_data = json.loads(json_str)
                sections["final_answer"] = json.dumps(json_data, indent=2)
            except:
                # Keep the original if parsing fails
                pass
    except Exception as e:
        logger.error(f"Error extracting JSON: {str(e)}")
    
    return sections
def get_medreason_diagnosis(transcription, medreason_url, medreason_token):
    """Get diagnosis from MedReason using the transcribed notes"""
    try:
        headers = {
            "Authorization": f"Bearer {medreason_token}",
            "Content-Type": "application/json"
        }
        
        # Create JSON prompt template with Triage Summary heading
        json_prompt = f"""
{transcription}

Analyze the medical information above. After your analysis, provide your conclusion in valid JSON format with the following fields:
{{
  "patient_name": "Full Name",
  "date_of_birth": "MM/DD/YYYY",
  "visit_time": "Date and Time",
  "severity": "Mild/Moderate/Severe",
  "primary_diagnosis": "Primary diagnosis",
  "secondary_diagnoses": "Comma-separated list of secondary diagnoses or 'None'",
  "recommended_tests": "Comma-separated list of recommended tests",
  "recommended_treatment": "Treatment plan",
  "follow_up": "Follow-up recommendations",
  "medical_reasoning": "Brief summary of your medical reasoning"
}}

Analyze the case carefully step by step. Include your thinking process and medical reasoning, following this structure:

## Thinking
Systematically explore possible diagnoses based on symptoms, findings, and medical history.

### Reasoning Process
Explain your diagnostic reasoning in detail, considering differential diagnoses and their likelihood.

### Conclusion
Summarize your findings and medical assessment.

## Triage Summary
Return your final answer in valid JSON format with all the fields mentioned above. Each field must contain a string value - no arrays allowed. Return ONLY valid JSON with no additional text.
"""
        
        # Create payload for the request
        payload = {
            "model": "UCSC-VLAA/MedReason-8B",
            "prompt": json_prompt,
            "max_tokens": 4096,
            "temperature": 0.1
        }
        
        logger.info(f"Sending request to MedReason at: {medreason_url}")
        response = requests.post(
            medreason_url,
            headers=headers,
            data=json.dumps(payload),
            verify=False,
            timeout=60  # Longer timeout for MedReason responses
        )
        
        if response.status_code == 200:
            result = response.json()
            logger.info(f"Received successful response from MedReason")
            return parse_medreason_response(result)
        else:
            logger.error(f"MedReason API error: {response.status_code}, {response.text}")
            return {
                "thinking": "",
                "reasoning": "",
                "conclusion": "",
                "final_answer": f"Error: MedReason API returned status code {response.status_code}"
            }
            
    except Exception as e:
        logger.error(f"Error getting diagnosis from MedReason: {str(e)}")
        return {
            "thinking": "",
            "reasoning": "",
            "conclusion": "",
            "final_answer": f"Error: {str(e)}"
        }

def save_config(config):
    """Save configuration to file"""
    try:
        with open('api_config.json', 'w') as f:
            json.dump(config, f)
        return "Configuration saved successfully"
    except Exception as e:
        return f"Failed to save configuration: {str(e)}"

def load_config():
    """Load configuration from file or use defaults"""
    try:
        if os.path.exists('api_config.json'):
            with open('api_config.json', 'r') as f:
                loaded_config = json.load(f)
                # Merge with defaults to ensure all services are present
                for service in DEFAULTS:
                    if service not in loaded_config:
                        loaded_config[service] = DEFAULTS[service]
                return loaded_config
        return DEFAULTS
    except Exception as e:
        logger.error(f"Failed to load configuration: {str(e)}")
        return DEFAULTS

def create_interface():
    """Create Gradio interface"""
    config = load_config()
    
    # Define the supported languages
    LANGUAGES = [
        "English", "German", "Polish", "Czech", 
        "Slovak", "Ukrainian", "Bulgarian", "Finnish"
    ]
    
    with gr.Blocks(theme=hpe_theme) as demo:
        logo_path = "logo.png"
        with open(logo_path, "rb") as f:
            logo_data = base64.b64encode(f.read()).decode("utf-8")
        
        gr.Markdown(f"""
        <div style="display: flex; align-items: center; gap: 10px">
            <img src="data:image/png;base64,{logo_data}" alt="Triage AI Logo" 
                 style="max-height: 30px; max-width: 30x; width: auto; height: auto; object-fit: contain;" />
            <h1 style="margin: 0">HealthcareAI powered by HPE Private Cloud AI</h1>
        </div>
        """)
        
        with gr.Tabs() as tabs:
            with gr.TabItem("About"):
                gr.Markdown("""
                ## About This Tool
                
                This tool is designed to support hospitals and clinics in the pursuit of providing superior healthcare treatment to patients.

                In this scenario we support the healthcare process by augmenting existing tools & expertise with AI powered intelligence.

                4 open-source models (served by HPE Private Cloud AI) provide the core functionality:

                **Models:**
                - NLLB ("No Language Left Behind"): Provides multi-lingual support for text generation/translation
                - Whisper: Provides speech-to-text capabilities
                - MedReason: Provides domain-specific (healthcare) reasoning capabilities for text such as doctors written diagnosis
                - MedGemma: Provides domain-specific (healthcare) reasoning capabilities for images such as submitted XRays
                
                **Features:**
                - Perform health check of 4 models.
                - Transcribe patient description of medical issues in core or niche languages
                - Perform live translations between patient language and native language of medical practitioner
                - Transcribe notes of the medical practitioner
                - Submit notes of medical practitioner for AI augmented analysis, diagnosis & treatment plan
                - Save results to database for automated recordkeeping

                
                Switch between tabs to view & experiment with all functionality. Please note audio can be recorded live or with recorded WAV files.
                """)
            
            with gr.TabItem("Model Health"):
                with gr.Row():
                    with gr.Column():
                        # MedReason Configuration
                        gr.Markdown("### MedReason LLM")
                        medreason_url = gr.Textbox(
                            label="API Endpoint", 
                            value=config["medreason"]["url"]
                        )
                        medreason_token = gr.Textbox(
                            label="API Token", 
                            value=config["medreason"]["token"],
                            type="password"
                        )
                        
                        # Whisper Configuration
                        gr.Markdown("### Whisper Speech-to-Text")
                        whisper_url = gr.Textbox(
                            label="API Endpoint", 
                            value=config["whisper"]["url"]
                        )
                        whisper_token = gr.Textbox(
                            label="API Token", 
                            value=config["whisper"]["token"],
                            type="password"
                        )
                        
                        # NLLB Configuration
                        gr.Markdown("### NLLB Translator")
                        nllb_url = gr.Textbox(
                            label="API Endpoint", 
                            value=config["nllb"]["url"]
                        )
                        nllb_token = gr.Textbox(
                            label="API Token", 
                            value=config["nllb"]["token"],
                            type="password"
                        )
                        
                        # ADD THIS NEW SECTION FOR MEDGEMMA:
                        gr.Markdown("### MedGemma Multimodal")
                        medgemma_url = gr.Textbox(
                            label="API Endpoint", 
                            value=config["medgemma"]["url"]
                        )
                        medgemma_token = gr.Textbox(
                            label="API Token", 
                            value=config["medgemma"]["token"],
                            type="password"
                        )
                        
                        # Buttons
                        with gr.Row():
                            check_btn = gr.Button("Check All Services", variant="primary")
                            save_btn = gr.Button("Save Configuration")
                    
                    with gr.Column():
                        gr.Markdown("### Model Health")
                        status_box = gr.Textbox(
                            label="Status Results", 
                            value="Click 'Check All Services' to verify connections",
                            lines=18,
                            show_label=False
                        )
            
            # Replace the entire Patient Translation tab with this code

            with gr.TabItem("TranslateAI"):
                with gr.Row():
                    with gr.Column():
                        gr.Markdown("### Audio Input")
                        audio_mode = gr.Radio(
                            ["Upload Audio File", "Record Live Audio"], 
                            label="Input Mode", 
                            value="Upload Audio File"
                        )
                        
                        # Dynamic inputs based on selection
                        with gr.Row(visible=True) as file_input_row:
                            audio_file = gr.Audio(
                                type="filepath", 
                                label="Upload Audio File",
                                show_label=False
                            )
                        
                        with gr.Row(visible=False) as record_input_row:
                            audio_recorder = gr.Audio(
                                type="filepath",
                                label="Record Audio",
                                sources=["microphone"],
                                show_label=False
                            )
                    
                        patient_language = gr.Dropdown(
                            LANGUAGES,
                            label="Patient Language",
                            value="English",
                            show_label=False
                        )
                        
                        doctor_language = gr.Dropdown(
                            LANGUAGES,
                            label="Doctor Language",
                            value="English",
                            show_label=False
                        )
                        
                        # Action buttons - now both in left column
                        with gr.Row():
                            transcribe_btn = gr.Button("Transcribe Audio", variant="primary")
                        
                        with gr.Row():
                            translate_btn = gr.Button("Translate Text", variant="primary")
                    
                    with gr.Column():
                        gr.Markdown("### Transcription & Translation")
                        
                        transcription_output = gr.Textbox(
                            label="Transcription",
                            value="",
                            lines=8
                        )
                        
                        translation_output = gr.Textbox(
                            label="Translation",
                            value="",
                            lines=8
                        )
                        
                        processing_status = gr.Textbox(
                            label="Processing Status",
                            value=""
                        )


            with gr.TabItem("TriageAI"):
                with gr.Row():
                    with gr.Column():
                        gr.Markdown("### Audio Input")
                        doctor_audio_mode = gr.Radio(
                            ["Upload Audio File", "Record Live Audio"], 
                            label="Input Mode", 
                            value="Upload Audio File",
                            show_label=False
                        )
                        
                        # Dynamic inputs based on selection
                        with gr.Row(visible=True) as doctor_file_input_row:
                            doctor_audio_file = gr.Audio(
                                type="filepath", 
                                label="Upload Doctor's Notes Audio"
                            )
                        
                        with gr.Row(visible=False) as doctor_record_input_row:
                            doctor_audio_recorder = gr.Audio(
                                type="filepath",
                                label="Record Doctor's Notes",
                                sources=["microphone"]
                            )
                        
                        with gr.Row():
                            doctor_transcribe_btn = gr.Button("Transcribe", variant="primary")
                            doctor_diagnose_btn = gr.Button("Diagnose", variant="primary")
                    
                    with gr.Column():
                        gr.Markdown("### Transcription")
                        doctor_transcription_output = gr.Textbox(
                            label="Transcribed Notes",
                            value="",
                            lines=5,
                            show_label=False
                        )
                        
                        doctor_processing_status = gr.Textbox(
                            label="Processing Status",
                            value="",
                            lines=2,
                            show_label=False
                        )
                        
                        # Moved the Save to Database button here, after the processing status
                        with gr.Row():
                            save_to_db_btn = gr.Button("Save to Database", variant="primary")
                        
                        # Removed the separate save_status box - we'll use doctor_processing_status instead
                
                with gr.Row():
                    with gr.Column():
                        gr.Markdown("### Complete Medreason Diagnosis")
                        doctor_raw_output = gr.Textbox(
                            label="Reasoning",
                            value="",
                            lines=20,
                            show_label=False
                        )
                    
                    with gr.Column():
                        gr.Markdown("### Structured Triage Summary")
                        doctor_json_output = gr.Textbox(
                            label="Summary",
                            value="",
                            lines=20,
                            show_label=False
                        )
            
            with gr.TabItem("XrayAI"):
                with gr.Row():
                    with gr.Column():
                        gr.Markdown("### X-ray Image Upload")
                        xray_image = gr.Image(
                            type="filepath",
                            label="Upload X-ray Image",
                            height=400,
                            show_label=False
                        )
                        
                        with gr.Row():
                            diagnose_xray_btn = gr.Button("Diagnose Patient", variant="primary")
                        
                        xray_status = gr.Textbox(
                            label="Processing Status",
                            value="",
                            lines=2,
                            show_label=False
                        )
                    
                    with gr.Column():
                        gr.Markdown("### X-ray Analysis Results")
                        xray_analysis_output = gr.Textbox(
                            label="Medical Analysis",
                            value="",
                            lines=25,
                            show_label=False
                        )
        
        # Health Check tab functions
        def check_all_services(medreason_url, medreason_token, whisper_url, whisper_token, nllb_url, nllb_token, medgemma_url, medgemma_token):
            """Check all services and return formatted status"""
            results = []
            
            # Check each model
            medreason_ok, medreason_msg = check_model("medreason", medreason_url, medreason_token)
            whisper_ok, whisper_msg = check_model("whisper", whisper_url, whisper_token)
            nllb_ok, nllb_msg = check_model("nllb", nllb_url, nllb_token)
            medgemma_ok, medgemma_msg = check_model("medgemma", medgemma_url, medgemma_token)
            
            # Format results
            status = lambda ok: "✅" if ok else "❌"
            results.append(f"{status(medreason_ok)} MedReason LLM: {medreason_msg}")
            results.append(f"{status(whisper_ok)} Whisper STT: {whisper_msg}")
            results.append(f"{status(nllb_ok)} NLLB Translator: {nllb_msg}")
            results.append(f"{status(medgemma_ok)} MedGemma Multimodal: {medgemma_msg}")
            
            return "\n\n".join(results)
        
        def save_config_and_check(medreason_url, medreason_token, whisper_url, whisper_token, nllb_url, nllb_token, medgemma_url, medgemma_token):
            """Save configuration and then check services"""
            config = {
                "medreason": {"url": medreason_url, "token": medreason_token},
                "whisper": {"url": whisper_url, "token": whisper_token},
                "nllb": {"url": nllb_url, "token": nllb_token},
                "medgemma": {"url": medgemma_url, "token": medgemma_token}
            }
            
            save_msg = save_config(config)
            check_msg = check_all_services(medreason_url, medreason_token, whisper_url, whisper_token, nllb_url, nllb_token, medgemma_url, medgemma_token)
            
            return f"{check_msg}\n\n{save_msg}"


        def analyze_xray_image(image_path, medgemma_url, medgemma_token):
            """Analyze uploaded X-ray image"""
            if not image_path:
                return "No image uploaded. Please upload an X-ray image first.", ""
            
            try:
                # Perform X-ray analysis with MedGemma
                analysis = analyze_xray_with_medgemma(image_path, medgemma_url, medgemma_token)
                return "Analysis complete.", analysis
                    
            except Exception as e:
                logger.error(f"Error in analyze_xray_image: {str(e)}")
                return f"Error: {str(e)}", ""
                
        # Patient Translation tab functions
        def toggle_audio_input(choice):
            """Toggle between file upload and microphone recording"""
            if choice == "Upload Audio File":
                return gr.update(visible=True), gr.update(visible=False)
            else:
                return gr.update(visible=False), gr.update(visible=True)
        
        def transcribe_audio_only(audio_mode, audio_file, audio_recorder, patient_language, whisper_url, whisper_token):
            """Process audio through Whisper for transcription in specified language"""
            # Select the appropriate audio path based on the mode
            audio_path = audio_file if audio_mode == "Upload Audio File" else audio_recorder
            
            if not audio_path:
                return "", "No audio provided. Please upload or record audio first."
            
            try:
                # Transcribe with Whisper using specified language
                iso_lang_code = get_iso_language_code(patient_language)
                headers = {
                    "Authorization": f"Bearer {whisper_token}"
                }
                
                files = {
                    'file': open(audio_path, 'rb'),
                    'model': (None, 'openai/whisper-large-v3'),
                    'language': (None, iso_lang_code)
                }
                
                logger.info(f"Transcribing in {patient_language} (code: {iso_lang_code})")
                
                response = requests.post(
                    whisper_url,
                    headers=headers,
                    files=files,
                    verify=False,
                    timeout=300
                )
                
                if response.status_code == 200:
                    result = response.json()
                    transcription = result.get('text', '')
                    
                    if not transcription or transcription.strip() == "":
                        return "", f"No transcription returned. Please try again with a clearer recording."
                    
                    return transcription, f"Transcribed in {patient_language}."
                        
                else:
                    logger.error(f"Whisper API error: {response.status_code}, {response.text}")
                    return "", f"Transcription failed: API returned status {response.status_code}"
                        
            except Exception as e:
                logger.error(f"Error in transcribe_audio_only: {str(e)}")
                return "", f"Error: {str(e)}"
            finally:
                # Close the file properly
                try:
                    files['file'].close()
                except:
                    pass
        
        # Replace the translate_text_only function with:
        def translate_text_only(transcription, patient_language, doctor_language, nllb_url, nllb_token):
            """Translate text using NLLB API"""
            if not transcription.strip():
                return "", "No text to translate. Please transcribe audio first."
            
            try:
                # Skip translation if source and target languages are the same
                if doctor_language.lower() == patient_language.lower():
                    return transcription, f"Translation skipped (both languages are {patient_language})."
                
                # Get language codes for both source and target
                source_lang_code = get_language_code(patient_language)
                target_lang_code = get_language_code(doctor_language)
                
                logger.info(f"Translating from {patient_language} to {doctor_language}")
                
                # Perform translation
                translation = translate_text(
                    transcription, 
                    source_lang_code, 
                    target_lang_code,
                    nllb_url,
                    nllb_token
                )
                
                return translation, f"Translated from {patient_language} to {doctor_language}."
                    
            except Exception as e:
                logger.error(f"Error in translate_text_only: {str(e)}")
                return "", f"Error: {str(e)}"
        
        # Connect Health Check tab buttons
        check_btn.click(
            fn=check_all_services,
            inputs=[medreason_url, medreason_token, whisper_url, whisper_token, nllb_url, nllb_token, medgemma_url, medgemma_token],
            outputs=status_box
        )
        
        save_btn.click(
            fn=save_config_and_check,
            inputs=[medreason_url, medreason_token, whisper_url, whisper_token, nllb_url, nllb_token, medgemma_url, medgemma_token],
            outputs=status_box
        )

        
        # Connect Patient Translation tab components
        audio_mode.change(
            fn=toggle_audio_input,
            inputs=audio_mode,
            outputs=[file_input_row, record_input_row]
        )
        
        transcribe_btn.click(
            fn=transcribe_audio_only,
            inputs=[
                audio_mode,
                audio_file,
                audio_recorder,
                patient_language,
                whisper_url,
                whisper_token
            ],
            outputs=[
                transcription_output,
                processing_status
            ]
        )
        
        translate_btn.click(
            fn=translate_text_only,
            inputs=[
                transcription_output,
                patient_language,
                doctor_language,
                nllb_url,
                nllb_token
            ],
            outputs=[
                translation_output,
                processing_status
            ]
        )

        # Connect XrayAI tab button
        diagnose_xray_btn.click(
            fn=analyze_xray_image,
            inputs=[
                xray_image,
                medgemma_url,
                medgemma_token
            ],
            outputs=[
                xray_status,
                xray_analysis_output
            ]
        )
        
        # Toggle doctor audio input based on mode
        doctor_audio_mode.change(
            fn=toggle_audio_input,  # Reuse the same function from Patient Translation
            inputs=doctor_audio_mode,
            outputs=[doctor_file_input_row, doctor_record_input_row]
        )
        
        
        def transcribe_doctor_notes(audio_mode, audio_file, audio_recorder, whisper_url, whisper_token):
            """Transcribe doctor's audio notes"""
            # Select the appropriate audio path based on the mode
            audio_path = audio_file if audio_mode == "Upload Audio File" else audio_recorder
            
            if not audio_path:
                return "", "No audio provided. Please upload or record audio first."
            
            try:
                # Transcribe with Whisper
                status_msg = "Transcribing..."
                detected_lang, transcription = transcribe_audio(audio_path, False, "English", whisper_url, whisper_token)
                
                if not transcription or transcription.strip() == "":
                    return "", "Failed to transcribe audio. Please try again with a clearer recording."
                
                return transcription, "Transcription complete."
                    
            except Exception as e:
                logger.error(f"Error in transcribe_doctor_notes: {str(e)}")
                return "", f"Error: {str(e)}"
        
        def diagnose_doctor_notes(transcription, medreason_url, medreason_token):
            """Process doctor's notes through MedReason"""
            if not transcription or transcription.strip() == "":
                return "", "", "No transcription to analyze. Please transcribe audio first."
            
            try:
                # Process with MedReason
                status_msg = "Analyzing with MedReason..."
                
                # Call MedReason API directly without using parse_medreason_response first
                headers = {
                    "Authorization": f"Bearer {medreason_token}",
                    "Content-Type": "application/json"
                }
                
                # Create JSON prompt template with Triage Summary heading
                # Create JSON prompt template with Triage Summary heading with standardized date formats
                json_prompt = f"""
                {transcription}
                
                Analyze the medical information above. After your analysis, provide your conclusion in valid JSON format with the following fields:
                {{
                  "patient_name": "Full Name",
                  "date_of_birth": "YYYY-MM-DD",  // Format birth date as YYYY-MM-DD for database compatibility
                  "visit_time": "YYYY-MM-DD HH:MM:SS",  // Format visit time as YYYY-MM-DD HH:MM:SS for database compatibility
                  "severity": "Mild/Moderate/Severe",
                  "primary_diagnosis": "Primary diagnosis",
                  "secondary_diagnoses": "Comma-separated list of secondary diagnoses or 'None'",
                  "recommended_tests": "Comma-separated list of recommended tests",
                  "recommended_treatment": "Treatment plan",
                  "follow_up": "Follow-up recommendations",
                  "medical_reasoning": "Brief summary of your medical reasoning"
                }}
                
                Analyze the case carefully step by step. Include your thinking process and medical reasoning, following this structure:
                
                ## Thinking
                Systematically explore possible diagnoses based on symptoms, findings, and medical history.
                
                ### Reasoning Process
                Explain your diagnostic reasoning in detail, considering differential diagnoses and their likelihood.
                
                ### Conclusion
                Summarize your findings and medical assessment.
                
                ## Triage Summary
                Return your final answer in valid JSON format with all the fields mentioned above. Each field must contain a string value - no arrays allowed.
                
                IMPORTANT: Format dates and times as follows:
                - date_of_birth: Use YYYY-MM-DD format (e.g., 1978-01-10)
                - visit_time: Use YYYY-MM-DD HH:MM:SS format (e.g., 2025-04-23 14:30:00)
                
                Return ONLY valid JSON with no additional text.
                """
                
                # Create payload for the request
                payload = {
                    "model": "UCSC-VLAA/MedReason-8B",
                    "prompt": json_prompt,
                    "max_tokens": 4096,
                    "temperature": 0.7
                }
                
                logger.info(f"Sending request to MedReason at: {medreason_url}")
                response = requests.post(
                    medreason_url,
                    headers=headers,
                    data=json.dumps(payload),
                    verify=False,
                    timeout=60  # Longer timeout for MedReason responses
                )
                
                if response.status_code == 200:
                    result = response.json()
                    logger.info(f"Received successful response from MedReason")
                    
                    # Extract text from different possible response formats
                    if "choices" in result and len(result["choices"]) > 0:
                        if "text" in result["choices"][0]:
                            raw_response = result["choices"][0]["text"]
                        elif "message" in result["choices"][0]:
                            raw_response = result["choices"][0]["message"]["content"]
                        else:
                            raw_response = json.dumps(result["choices"][0], indent=2)
                    elif "response" in result:
                        raw_response = result["response"]
                    elif "generations" in result:
                        raw_response = result["generations"][0]["text"]
                    else:
                        raw_response = json.dumps(result, indent=2)
                    
                    # Extract JSON from raw response - IMPROVED VERSION
                    json_output = ""
                    try:
                        # Look for JSON between curly braces that appears right after "## Triage Summary" or "## Final Answer"
                        if "## Triage Summary" in raw_response:
                            after_marker = raw_response.split("## Triage Summary")[1].strip()
                        elif "## Final Answer" in raw_response:
                            after_marker = raw_response.split("## Final Answer")[1].strip()
                        else:
                            after_marker = raw_response
                        
                        # Find the first curly brace and the last curly brace
                        if "{" in after_marker and "}" in after_marker:
                            start = after_marker.find("{")
                            # Find the last balanced closing brace
                            open_count = 0
                            end = -1
                            for i, char in enumerate(after_marker[start:]):
                                if char == '{':
                                    open_count += 1
                                elif char == '}':
                                    open_count -= 1
                                    if open_count == 0:
                                        end = start + i + 1
                                        break
                            
                            if end > start:
                                json_str = after_marker[start:end]
                                
                                # Format the JSON nicer if possible
                                try:
                                    json_data = json.loads(json_str)
                                    json_output = json.dumps(json_data, indent=2)
                                except:
                                    json_output = "Invalid JSON format"
                            else:
                                json_output = "No valid JSON found in response"
                        else:
                            json_output = "No JSON found in response"
                    except Exception as e:
                        logger.error(f"Error extracting JSON: {str(e)}")
                        json_output = f"Error extracting JSON: {str(e)}"
                    
                    return raw_response, json_output, "Analysis complete."
                else:
                    logger.error(f"MedReason API error: {response.status_code}, {response.text}")
                    return "", f"Error: MedReason API returned status code {response.status_code}", f"Error: API returned status {response.status_code}"
                    
            except Exception as e:
                logger.error(f"Error in diagnose_doctor_notes: {str(e)}")
                return "", f"Error: {str(e)}", f"Error: {str(e)}"
        
        # Connect doctor diagnose button

        doctor_transcribe_btn.click(
            fn=transcribe_doctor_notes,
            inputs=[
                doctor_audio_mode,
                doctor_audio_file,
                doctor_audio_recorder,
                whisper_url,
                whisper_token
            ],
            outputs=[
                doctor_transcription_output,
                doctor_processing_status
            ]
        )
        
        doctor_diagnose_btn.click(
            fn=diagnose_doctor_notes,
            inputs=[
                doctor_transcription_output,
                medreason_url,
                medreason_token
            ],
            outputs=[
                doctor_raw_output,
                doctor_json_output,
                doctor_processing_status
            ]
        )

        save_to_db_btn.click(
            fn=save_diagnosis_to_db_button,
            inputs=[doctor_json_output],
            outputs=[doctor_processing_status]
        )

    return demo

if __name__ == "__main__":
    logger.info("Creating Gradio interface")
    demo = create_interface()
    logger.info("Launching Gradio server")
    demo.launch(server_name="0.0.0.0", server_port=7860)