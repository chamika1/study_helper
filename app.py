import os
import json
import requests
import uuid
import re
from flask import Flask, render_template, request, jsonify, session
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Configure the Gemini API
API_KEY = os.getenv("GEMINI_API_KEY", "#API-KEY")  # Using the provided API key with fallback
API_URL = "https://generativelanguage.googleapis.com/v1beta/models/gemini-2.5-flash:generateContent"  # Using gemini-2.5-flash model

# Configure GetImg.ai API for image generation
GETIMG_API_KEY = os.getenv("GETIMG_API_KEY", "#IMGKEY")
GETIMG_API_URL = "https://api.getimg.ai/v1/stable-diffusion-xl/text-to-image"
AVAILABLE_IMAGE_MODELS = ["stable-diffusion-xl-v1-0", "realistic-vision-v5-1", "dreamshaper-8", "anything-v5", "openjourney-v4", "realvis-xl-v4"]
DEFAULT_IMAGE_MODEL = "stable-diffusion-xl-v1-0"

# Initialize Flask app
app = Flask(__name__)
app.secret_key = os.getenv("SECRET_KEY", "your-secret-key")  # Fixed: Properly evaluate the function
app.config['JSON_AS_ASCII'] = False  # Enable UTF-8 support for JSON responses

# Store conversation history
conversations = {}

# Function to call the Gemini API
def call_gemini_api(prompt, conversation_id=None):
    try:
        # Prepare the request payload
        payload = {
            "contents": [],
            "generationConfig": {
                "temperature": 0.7,
                "topK": 40,
                "topP": 0.95,
                "maxOutputTokens": 8192,
            }
        }
        
        # Add conversation history if available
        if conversation_id and conversation_id in conversations:
            payload["contents"] = conversations[conversation_id].copy()
        
        # Add the current prompt
        payload["contents"].append({
            "role": "user",
            "parts": [{"text": prompt}]
        })
        
        # Make the API request
        response = requests.post(
            f"{API_URL}?key={API_KEY}",
            json=payload,
            headers={"Content-Type": "application/json"}
        )
        
        # Check if the request was successful
        if response.status_code == 200:
            response_json = response.json()
            
            # Update conversation history
            if conversation_id:
                if conversation_id not in conversations:
                    conversations[conversation_id] = []
                
                # Add user message to history
                conversations[conversation_id].append({
                    "role": "user",
                    "parts": [{"text": prompt}]
                })
                
                # Add model response to history - FIX HERE
                try:
                    model_response = response_json["candidates"][0]["content"]
                    conversations[conversation_id].append({
                        "role": "model",
                        "parts": [{"text": model_response["parts"][0]["text"]}]  # Fixed structure
                    })
                    
                    # Limit history to last 20 exchanges to prevent context overflow
                    if len(conversations[conversation_id]) > 20:
                        conversations[conversation_id] = conversations[conversation_id][-20:]
                except (KeyError, IndexError) as e:
                    print(f"Error updating conversation history: {e}")
            
            return response_json
        else:
            error_message = f"API call failed with status code {response.status_code}"
            try:
                error_details = response.json()
                error_message += f": {json.dumps(error_details)}"
            except:
                error_message += f": {response.text}"
            
            print(error_message)  # Log the error for debugging
            return {"error": error_message}
    except Exception as e:
        error_message = f"Exception during API call: {str(e)}"
        print(error_message)  # Log the error for debugging
        return {"error": error_message}

# Load physics topics from file
def load_physics_topics():
    try:
        with open("physics.txt", "r", encoding="utf-8") as file:
            content = file.read()
            
        # Parse the content to extract units and topics
        units = {}
        current_unit = None
        
        for line in content.split('\n'):
            line = line.strip()
            if line.startswith("ඒකකය"):
                current_unit = line
                units[current_unit] = []
            elif line and current_unit and not line[0].isdigit() and ":" not in line:
                units[current_unit].append(line)
                
        return units
    except Exception as e:
        print(f"Error loading physics topics: {e}")
        return {"Error": ["Failed to load topics"]}

# Load physics topics
physics_units = load_physics_topics()

# Function to generate images using GetImg.ai API
def generate_image(prompt, model=None):
    """Generate an image using GetImg.ai API"""
    try:
        if model is None or model not in AVAILABLE_IMAGE_MODELS:
            model = DEFAULT_IMAGE_MODEL
        
        # Enhance the prompt for physics diagrams
        enhanced_prompt = f"Scientific educational diagram: {prompt}. Clear, labeled, educational style, high quality scientific illustration"
        
        headers = {
            "accept": "application/json",
            "content-type": "application/json",
            "authorization": f"Bearer {GETIMG_API_KEY}"
        }
        
        payload = {
            "prompt": enhanced_prompt,
            "model": model,
            "negative_prompt": "Disfigured, cartoon, blurry, low quality, bad anatomy, bad proportions, deformed, distorted, watermark, signature, text",
            "width": 1024,
            "height": 1024,
            "steps": 40,
            "guidance": 8.0,
            "scheduler": "euler",
            "output_format": "png",
            "response_format": "url",
            "seed": 42
        }
        
        response = requests.post(
            GETIMG_API_URL,
            headers=headers,
            json=payload,
            timeout=30
        )
        
        if response.status_code == 200:
            result = response.json()
            return {"success": True, "url": result.get("url")}
        else:
            return {"success": False, "error": f"API returned status {response.status_code}: {response.text}"}
            
    except Exception as e:
        return {"success": False, "error": str(e)}

# Function to process response text and replace [IMAGE:...] tags with actual images
def process_image_tags(text):
    """Process [IMAGE:description] tags and replace with markdown image syntax"""
    # Find all [IMAGE:...] tags
    pattern = r'\[IMAGE:(.*?)\]'
    matches = re.findall(pattern, text)
    
    for match in matches:
        # Generate image for each tag
        image_result = generate_image(match)
        
        if image_result.get("success"):
            # Replace [IMAGE:description] with markdown image syntax
            image_url = image_result.get("url")
            markdown_image = f"![{match}]({image_url})"
            text = text.replace(f"[IMAGE:{match}]", markdown_image)
        else:
            # If image generation failed, replace with error message
            error_msg = f"[Image generation failed: {match}]"
            text = text.replace(f"[IMAGE:{match}]", error_msg)
    
    return text

# Function to fix common LaTeX formatting issues in questions
def fix_latex_formatting(text):
    """Fix common LaTeX formatting issues to ensure proper MathJax rendering"""
    if not text:
        return text
    
    # First, normalize existing math delimiters - extract math content to protect it
    math_blocks = []
    def save_math(match):
        math_blocks.append(match.group(1))
        return f"###MATH{len(math_blocks)-1}###"
    
    # Save existing inline math $...$ 
    text = re.sub(r'\$([^\$]+)\$', save_math, text)
    
    # Now fix problematic patterns in non-math text
    
    # Fix \degree C or \degree F -> proper degree symbol
    text = re.sub(r'\\degree\s*([CF])', r'°\1', text)
    
    # Fix standalone degree symbol ° followed by C or F
    text = re.sub(r'(\d+)\s*°\s*([CF])', r'\1$^\\circ$\2', text)
    
    # Fix \textsuperscript{...} 
    text = re.sub(r'\\textsuperscript\{([^}]+)\}', r'^{\1}', text)
    
    # Fix \textsubscript{...}
    text = re.sub(r'\\textsubscript\{([^}]+)\}', r'_{\1}', text)
    
    # Fix patterns like "20^°C" or "20^^C" (double exponents)
    text = re.sub(r'(\d+)\s*\^\s*°\s*([CF])', r'\1$^\\circ$\2', text)
    text = re.sub(r'(\d+)\s*\^\s*\^\s*([CF])', r'\1$^\\circ$\2', text)
    
    # Restore saved math blocks and fix any issues within them
    for i, math_content in enumerate(math_blocks):
        # Fix double exponents in saved math: ^^ -> ^
        math_content = re.sub(r'\^\s*\^', '^', math_content)
        # Fix \degree in math mode
        math_content = re.sub(r'\\degree', r'^\\circ', math_content)
        # Replace placeholder
        text = text.replace(f"###MATH{i}###", f"${math_content}$")
    
    # Clean up multiple consecutive $ signs
    text = re.sub(r'\$+', '$', text)
    text = re.sub(r'\$\s*\$', '', text)
    
    return text

@app.route('/')
def index():
    # Generate a unique conversation ID for this session if not exists
    if 'conversation_id' not in session:
        session['conversation_id'] = str(uuid.uuid4())
    
    return render_template('index.html', 
                          units=physics_units,
                          modes=["Learn Step-by-Step", "Practice Questions", "Exam Tips", "Study Plan", "Short Notes"],
                          languages=["English", "Sinhala"])


@app.route('/practice')
def practice():
    # Generate a unique conversation ID for this session if not exists
    if 'conversation_id' not in session:
        session['conversation_id'] = str(uuid.uuid4())
    
    return render_template('practice.html',
                          units=physics_units,
                          languages=["English", "Sinhala"])

@app.route('/exam_tips')
def exam_tips():
    # Generate a unique conversation ID for this session if not exists
    if 'conversation_id' not in session:
        session['conversation_id'] = str(uuid.uuid4())
    
    return render_template('exam_tips.html', 
                          units=physics_units, 
                          languages=["English", "Sinhala"])


@app.route('/about')
def about():
    # Generate a unique conversation ID for this session if not exists
    if 'conversation_id' not in session:
        session['conversation_id'] = str(uuid.uuid4())
    
    return render_template('about.html')


@app.route('/study_plan')
def study_plan():
    # Generate a unique conversation ID for this session if not exists
    if 'conversation_id' not in session:
        session['conversation_id'] = str(uuid.uuid4())
    
    return render_template('study_plan.html', 
                          units=physics_units, 
                          languages=["English", "Sinhala"])

@app.route('/short_notes')
def short_notes():
    # Generate a unique conversation ID for this session if not exists
    if 'conversation_id' not in session:
        session['conversation_id'] = str(uuid.uuid4())
    
    return render_template('short_notes.html', 
                          units=physics_units, 
                          languages=["English", "Sinhala"])

@app.route('/get_topics', methods=['POST'])
def get_topics():
    unit = request.form.get('unit')
    if unit in physics_units:
        return jsonify({"topics": physics_units[unit]})
    return jsonify({"topics": []})

# In the start_teaching function
@app.route('/start_teaching', methods=['POST'])
def start_teaching():
    unit = request.form.get('unit')
    topic = request.form.get('topic')
    language = request.form.get('language')
    conversation_id = session.get('conversation_id')
    
    # Reset conversation history for this session
    if conversation_id in conversations:
        conversations[conversation_id] = []
    
    # Create a prompt to start teaching the unit with improved translation instructions
    if language == "Sinhala":
        lang_instruction = "Respond in Sinhala. Use proper Sinhala physics terminology. For physics terms, use standard academic Sinhala terminology used in Sri Lankan A/L textbooks. "
    else:
        lang_instruction = ""
    
    # Add image generation instruction for GetImg.ai API
    image_instruction = "When appropriate, you can request images by using this special format: [IMAGE:description of the physics concept]. For example: [IMAGE:electric field lines around a positive charge]. The system will generate a high-quality scientific diagram. Use this sparingly for important visual concepts only."
    
    # Create a prompt based on whether a specific topic is selected
    if topic and topic != "All Topics":
        prompt = f"{lang_instruction}{image_instruction}I want to learn about the topic '{topic}' in the unit '{unit}' for Sri Lankan A/L Physics. Please start teaching me this specific topic step by step, explaining the fundamental concepts first. Break down the content into manageable parts and ask me if I understand before moving to the next part."
    else:
        prompt = f"{lang_instruction}{image_instruction}I want to learn about {unit} for Sri Lankan A/L Physics. Please start teaching me step by step, explaining the fundamental concepts first. Break down the content into manageable parts and ask me if I understand before moving to the next part."
    
    # Make API call
    response_json = call_gemini_api(prompt, conversation_id)
    
    # Handle the response
    if "error" in response_json:
        return jsonify({"error": response_json["error"]})
    else:
        try:
            # Extract text from the response
            response_text = response_json["candidates"][0]["content"]["parts"][0]["text"]
            # Process any [IMAGE:...] tags in the response
            response_text = process_image_tags(response_text)
            return jsonify({
                "response": response_text,
                "conversation_id": conversation_id
            })
        except (KeyError, IndexError) as e:
            return jsonify({"error": f"Error parsing API response: {e}", "raw_response": response_json})

@app.route('/continue_conversation', methods=['POST'])
def continue_conversation():
    user_message = request.form.get('message')
    language = request.form.get('language')
    conversation_id = session.get('conversation_id')
    
    if not user_message:
        return jsonify({"error": "No message provided"})
    
    # Make API call with conversation history
    response_json = call_gemini_api(user_message, conversation_id)
    
    # Handle the response
    if "error" in response_json:
        return jsonify({"error": response_json["error"]})
    else:
        try:
            # Extract text from the response
            response_text = response_json["candidates"][0]["content"]["parts"][0]["text"]
            # Process any [IMAGE:...] tags in the response
            response_text = process_image_tags(response_text)
            return jsonify({
                "response": response_text,
                "conversation_id": conversation_id
            })
        except (KeyError, IndexError) as e:
            return jsonify({"error": f"Error parsing API response: {e}", "raw_response": response_json})

# In the generate function
@app.route('/generate', methods=['POST'])
def generate():
    unit = request.form.get('unit')
    topic = request.form.get('topic')
    mode = request.form.get('mode')
    language = request.form.get('language')
    user_query = request.form.get('user_query')
    conversation_id = session.get('conversation_id')
    
    # Create a prompt based on the selected topic and mode
    topic_text = topic if topic != "All Topics" else f"all topics in {unit}"
    
    # Add language preference with improved translation instructions
    if language == "Sinhala":
        lang_instruction = "Respond in Sinhala. Use proper Sinhala physics terminology. For physics terms, use standard academic Sinhala terminology used in Sri Lankan A/L textbooks. "
    else:
        lang_instruction = ""
    
    # Add image generation instruction for GetImg.ai API
    image_instruction = "When appropriate, you can request images by using this special format: [IMAGE:description of the physics concept]. For example: [IMAGE:electric field lines around a positive charge]. The system will generate a high-quality scientific diagram. Use this sparingly for important visual concepts only."
    
    if mode == "Learn Step-by-Step":
        prompt = f"{lang_instruction}{image_instruction}Explain the following concept in Physics for Sri Lankan A/L students: {topic_text}. Break it down step by step with clear explanations and examples."
    elif mode == "Practice Questions":
        prompt = f"{lang_instruction}Generate 3 practice questions about {topic_text} for Sri Lankan A/L Physics students. Include detailed solutions."
    elif mode == "Exam Tips":
        prompt = f"{lang_instruction}Provide exam tips for Sri Lankan A/L Physics students on the topic: {topic_text}."
    elif mode == "Study Plan":
        prompt = f"{lang_instruction}Create a study plan for Sri Lankan A/L Physics students focusing on: {topic_text}."
    elif mode == "Short Notes":
        prompt = f"{lang_instruction}{image_instruction}Create concise short notes for quick revision on {topic_text} for Sri Lankan A/L Physics students."
    else:  # Default to explanation
        prompt = f"{lang_instruction}{image_instruction}Explain the concept of {topic_text} in Sri Lankan A/L Physics curriculum."
    
    if user_query:
        prompt += f" The student specifically asked: {user_query}"
    
    # Make API call
    response_json = call_gemini_api(prompt, conversation_id)
    
    # Handle the response
    if "error" in response_json:
        return jsonify({"error": response_json["error"]})
    else:
        try:
            # Extract text from the response
            response_text = response_json["candidates"][0]["content"]["parts"][0]["text"]
            # Process any [IMAGE:...] tags in the response
            response_text = process_image_tags(response_text)
            return jsonify({
                "response": response_text,
                "conversation_id": conversation_id
            })
        except (KeyError, IndexError) as e:
            return jsonify({"error": f"Error parsing API response: {e}", "raw_response": response_json})

@app.route('/clear_conversation', methods=['POST'])
def clear_conversation():
    conversation_id = session.get('conversation_id')
    
    if conversation_id in conversations:
        conversations[conversation_id] = []
        
    return jsonify({"status": "success", "message": "Conversation cleared"})

@app.route('/generate_image', methods=['POST'])
def generate_image_endpoint():
    """Endpoint to generate images using GetImg.ai API"""
    try:
        data = request.get_json()
        prompt = data.get('prompt')
        model = data.get('model', DEFAULT_IMAGE_MODEL)
        
        if not prompt:
            return jsonify({"success": False, "error": "No prompt provided"})
        
        result = generate_image(prompt, model)
        return jsonify(result)
        
    except Exception as e:
        return jsonify({"success": False, "error": str(e)})

@app.route('/generate_questions', methods=['POST'])
def generate_questions():
    unit = request.form.get('unit')
    topic = request.form.get('topic')
    question_type = request.form.get('question_type')
    difficulty = request.form.get('difficulty')
    language = request.form.get('language')
    num_questions = request.form.get('num_questions', '3')
    conversation_id = session.get('conversation_id')
    
    # Create a prompt based on the selected parameters
    topic_text = topic if topic != "All Topics" else f"all topics in {unit}"
    
    # Add language preference with improved translation instructions
    if language == "Sinhala":
        lang_instruction = "Respond in Sinhala. Use proper Sinhala physics terminology. For physics terms, use standard academic Sinhala terminology used in Sri Lankan A/L textbooks. "
    else:
        lang_instruction = ""
    
    # Create specific prompts based on question type
    if question_type == "MCQ":
        prompt = f"{lang_instruction}Generate {num_questions} multiple-choice questions (MCQs) about {topic_text} for Sri Lankan A/L Physics students at {difficulty} difficulty level. For each question, provide 4 options (A, B, C, D), clearly mark the correct answer, and include a brief explanation of why it's correct.\n\nIMPORTANT FORMATTING RULES:\n1. Wrap ALL mathematical expressions in dollar signs: $expression$ for inline or $$expression$$ for display\n2. Use proper LaTeX syntax: $\\\\frac{{a}}{{b}}$, $\\\\theta$, $^{{2}}$ for superscripts, $_{{2}}$ for subscripts\n3. For degrees: $^\\\\circ$C not \\\\degree C\n4. For units: $\\\\text{{kg}}$ or $\\\\mathrm{{kg}}$\n5. In JSON, use double backslashes: \\\\frac, \\\\theta, \\\\circ\n6. Example: $100 \\\\text{{ g}}$ at $20^\\\\circ\\\\text{{C}}$\n\nFormat the response as a JSON array with each question having these fields: 'question', 'options' (array), 'correct_answer' (index number), and 'explanation'."
    elif question_type == "Structured":
        prompt = (f"{lang_instruction}Generate {num_questions} structured questions about {topic_text} for Sri Lankan A/L Physics students at {difficulty} difficulty level. Each structured question should:\n"
                 f"1. Start with a detailed scenario or practical experiment description (5-8 sentences)\n"
                 f"2. Include 5-10 sub-questions that relate to different aspects of the scenario\n"
                 f"3. Require application of physics concepts to analyze the given situation\n"
                 f"4. Include a comprehensive model answer with clear step-by-step explanations for each sub-question\n\n"
                 f"IMPORTANT FORMATTING RULES:\n"
                 f"1. Wrap ALL mathematical expressions in dollar signs: $expression$ for inline or $$expression$$ for display\n"
                 f"2. Use proper LaTeX syntax: $\\\\frac{{a}}{{b}}$, $\\\\theta$, $^{{2}}$ for superscripts, $_{{2}}$ for subscripts\n"
                 f"3. For degrees: $^\\\\circ$C not \\\\degree C\n"
                 f"4. For units: $\\\\text{{kg}}$ or $\\\\mathrm{{kg}}$\n"
                 f"5. In JSON, use double backslashes: \\\\frac, \\\\theta, \\\\circ\n"
                 f"6. Example: $100 \\\\text{{ g}}$ at $20^\\\\circ\\\\text{{C}}$\n\n"
                 f"Format the response as a JSON array with each question having these fields: \n"
                 f"- 'scenario': the detailed practical context or scenario description\n"
                 f"- 'scenario_title': a brief title for the scenario\n"
                 f"- 'sub_questions': an array of objects, each with 'question' and 'answer' fields\n"
                 f"- 'answer': a comprehensive model answer covering all aspects with numbered steps for each calculation or explanation")
    else:  # Essay
        prompt = (f"{lang_instruction}Generate {num_questions} essay questions about {topic_text} for Sri Lankan A/L Physics students at {difficulty} difficulty level. Each essay question should:\n"
                 f"1. Be a complex question requiring detailed explanations\n"
                 f"2. Include calculations and critical thinking components\n"
                 f"3. Test deeper understanding of physics principles\n"
                 f"4. Have a comprehensive model answer with step-by-step calculations\n\n"
                 f"IMPORTANT FORMATTING RULES:\n"
                 f"1. Wrap ALL mathematical expressions in dollar signs: $expression$ for inline or $$expression$$ for display\n"
                 f"2. Use proper LaTeX syntax: $\\\\frac{{a}}{{b}}$, $\\\\theta$, $^{{2}}$ for superscripts, $_{{2}}$ for subscripts\n"
                 f"3. For degrees: $^\\\\circ$C not \\\\degree C\n"
                 f"4. For units: $\\\\text{{kg}}$ or $\\\\mathrm{{kg}}$\n"
                 f"5. In JSON, use double backslashes: \\\\frac, \\\\theta, \\\\circ\n"
                 f"6. Example: $100 \\\\text{{ g}}$ at $20^\\\\circ\\\\text{{C}}$\n\n"
                 f"Format the response as a JSON array with each question having these fields: 'question' and 'answer'.")
    
    # Make API call
    response_json = call_gemini_api(prompt, conversation_id)
    
    # Handle the response
    if "error" in response_json:
        return jsonify({"error": response_json["error"], "raw_response": response_json})
    else:
        try:
            # Extract text from the response
            response_text = response_json["candidates"][0]["content"]["parts"][0]["text"]
            
            # Extract JSON from the response text
            # Find JSON content between ```json and ``` or just parse the entire text
            json_match = None
            import re
            json_matches = re.findall(r'```(?:json)?\s*([\s\S]*?)\s*```', response_text)
            
            if json_matches:
                json_content = json_matches[0]
            else:
                json_content = response_text
            
            # Clean up the JSON content
            json_content = json_content.strip()
            
            # Try to parse the JSON with error handling
            try:
                questions = json.loads(json_content)
                
                # Fix LaTeX formatting in all questions after parsing
                for q in questions:
                    if 'question' in q:
                        q['question'] = fix_latex_formatting(q['question'])
                    if 'options' in q:
                        q['options'] = [fix_latex_formatting(opt) for opt in q['options']]
                    if 'explanation' in q:
                        q['explanation'] = fix_latex_formatting(q['explanation'])
                    if 'answer' in q:
                        q['answer'] = fix_latex_formatting(q['answer'])
                    if 'scenario' in q:
                        q['scenario'] = fix_latex_formatting(q['scenario'])
                    if 'sub_questions' in q:
                        for sq in q['sub_questions']:
                            if 'question' in sq:
                                sq['question'] = fix_latex_formatting(sq['question'])
                            if 'answer' in sq:
                                sq['answer'] = fix_latex_formatting(sq['answer'])
                
            except json.JSONDecodeError as e:
                # If parsing fails, try to fix common issues with LaTeX in JSON
                # Replace single backslashes with double backslashes for LaTeX commands
                # But be careful not to double-escape already escaped sequences
                print(f"Initial JSON parse failed: {e}")
                print(f"Attempting to fix LaTeX escape sequences...")
                
                # Common LaTeX commands that might cause issues
                latex_commands = [
                    r'\frac', r'\theta', r'\alpha', r'\beta', r'\gamma', r'\delta',
                    r'\epsilon', r'\sigma', r'\pi', r'\omega', r'\lambda', r'\mu',
                    r'\int', r'\sum', r'\sqrt', r'\cdot', r'\times', r'\div',
                    r'\partial', r'\nabla', r'\infty', r'\rightarrow', r'\leftarrow',
                    r'\Delta', r'\Omega', r'\Sigma', r'\Pi', r'\Gamma',
                    r'\text', r'\mathrm', r'\mathbf', r'\prime'
                ]
                
                # Try to fix the JSON by escaping LaTeX commands
                fixed_content = json_content
                for cmd in latex_commands:
                    # Replace \command with \\command (but not \\command)
                    fixed_content = re.sub(
                        r'(?<!\\)' + re.escape(cmd),
                        cmd.replace('\\', '\\\\'),
                        fixed_content
                    )
                
                try:
                    questions = json.loads(fixed_content)
                    print("Successfully parsed JSON after fixing escape sequences")
                except json.JSONDecodeError as e2:
                    print(f"Still failed after fixes: {e2}")
                    return jsonify({
                        "error": f"Failed to parse JSON response: {str(e)}",
                        "raw_response": response_text,
                        "json_content": json_content
                    })
            
            # Process the questions based on type
            if question_type == "MCQ":
                # Validate MCQ format
                for q in questions:
                    if 'options' not in q or 'correct_answer' not in q:
                        return jsonify({"error": "Invalid question format", "raw_response": response_text})
            elif question_type == "Structured":
                # Ensure structured questions have the right format
                for q in questions:
                    # If scenario is missing, use question as scenario
                    if 'scenario' not in q and 'question' in q:
                        q['scenario'] = q['question']
                    
                    # If sub_questions is missing, create a placeholder
                    if 'sub_questions' not in q:
                        q['sub_questions'] = []
                        # Try to extract sub-questions from the answer if possible
                        if 'answer' in q:
                            # Simple extraction based on numbered points
                            sub_q_matches = re.findall(r'(\d+[\)\.]\s*[^\n]+)', q['answer'])
                            for i, sq in enumerate(sub_q_matches):
                                q['sub_questions'].append({
                                    'question': sq,
                                    'answer': f"See the comprehensive answer for part {i+1}."
                                })
                    
                    # Ensure answers have step-by-step format
                    if 'answer' in q and not re.search(r'Step \d+:|^\d+\.\s', q['answer'], re.MULTILINE):
                        # If answer doesn't already have steps, try to format it with steps
                        paragraphs = re.split(r'\n\s*\n', q['answer'])
                        if len(paragraphs) > 1:
                            formatted_answer = ""
                            for i, para in enumerate(paragraphs):
                                if para.strip():
                                    formatted_answer += f"Step {i+1}: {para.strip()}\n\n"
                            q['answer'] = formatted_answer
                    
                    # Ensure each sub-question answer has steps
                    if 'sub_questions' in q:
                        for sub_q in q['sub_questions']:
                            if 'answer' in sub_q and not re.search(r'Step \d+:|^\d+\.\s', sub_q['answer'], re.MULTILINE):
                                # Format the answer with steps if it doesn't have them
                                lines = sub_q['answer'].split('\n')
                                if len(lines) > 1:
                                    formatted_answer = ""
                                    for i, line in enumerate(lines):
                                        if line.strip():
                                            formatted_answer += f"Step {i+1}: {line.strip()}\n"
                                    sub_q['answer'] = formatted_answer

            return jsonify({"questions": questions})
        except json.JSONDecodeError as e:
            return jsonify({
                "error": f"JSON parsing error: {str(e)}. The AI response may contain unescaped special characters.",
                "raw_response": response_text,
                "hint": "Try generating questions again. If the error persists, the mathematical formulas in the response may have formatting issues."
            })
        except Exception as e:
            return jsonify({
                "error": f"Error processing questions: {str(e)}",
                "raw_response": response_text
            })

# Modify the app.run() call at the end of the file
if __name__ == "__main__":
    # This is used when running locally
    app.run(host="0.0.0.0", port=8080, debug=True)
    
# No need to call app.run() when deployed to Vercel
