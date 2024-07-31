import os
import json
import io
import base64
import requests
import numpy as np
import cv2
import matplotlib.pyplot as plt
from flask import Flask, render_template, request, jsonify
from werkzeug.utils import secure_filename
from dotenv import load_dotenv
from PIL import Image
from keras.models import load_model
import markdown

from Test import get_defect_locations, predict_defects, process_image

# Load environment variables from .env file
load_dotenv()

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max-limit

# Ensure the upload folder exists
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# Load the model
model_path = 'D:\\PCB Defect Classification\\Model\\Model.h5'
model = load_model(model_path)

# ... (keep all other functions as they are)

def analyze_with_groq(defect_names, user_query=None):
    api_key = os.getenv('GROQ_API_KEY')
    if not api_key:
        raise ValueError("GROQ_API_KEY not found in environment variables")
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json"
    }
    
    if user_query:
        prompt = f"""You are an expert in PCB (Printed Circuit Board) manufacturing and quality control. 
        The following defects were detected in a PCB: {json.dumps(defect_names, indent=2)}
        
        The user has asked the following question: {user_query}
        
        Please provide a detailed answer to the user's question, focusing on PCB manufacturing, 
        defects, and quality control. If the question is not related to PCBs, politely inform 
        the user that you can only answer questions related to PCB manufacturing and defects."""
    else:
        prompt = f"""You are an expert in PCB (Printed Circuit Board) manufacturing and quality control. 
        Analyze the following defect types and provide detailed explanations on how to fix each issue:

        {json.dumps(defect_names, indent=2)}

        For each defect type, explain:
        1. What causes this type of defect?
        2. How to fix this specific issue in the manufacturing process?
        3. Preventive measures to avoid such defects in future production.

        Format your response in Markdown, with separate sections for each defect type."""
    
    data = {
        "model": "mixtral-8x7b-32768",
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0.7,
        "max_tokens": 1000
    }

    try:
        response = requests.post("https://api.groq.com/openai/v1/chat/completions", headers=headers, json=data)
        response.raise_for_status()
        return response.json()['choices'][0]['message']['content']
    except requests.exceptions.RequestException as e:
        print(f"Error communicating with Groq API: {e}")
        return "I apologize, but I'm having trouble accessing my knowledge base right now. Please try again later."

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return jsonify({"error": "No file part"}), 400
    file = request.files['file']
    if file.filename == '':
        return jsonify({"error": "No selected file"}), 400
    if file:
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)
        
        try:
            # Process the image to find defects
            contours, image = process_image(filepath)
            CX, CY, C = get_defect_locations(contours)
            pred, confidence, classes = predict_defects(model, filepath, C)

            # Deduplicate defect types
            unique_defect_types = list(set(classes[p] for p in pred))
            
            # Generate annotated image
            plt.switch_backend('Agg')  
            plt.figure(figsize=(12, 8))
            plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
            plt.scatter(CX, CY, c='r', s=40)
            for i, txt in enumerate(pred):
                plt.annotate(f"{classes[txt]}\n{confidence[i][0][txt]:.2f}", (CX[i], CY[i]), color='r', fontsize=8)
            plt.title("PCB Defect Detection Results")
            plt.axis('off')
            plt.tight_layout()
            
            img_buffer = io.BytesIO()
            plt.savefig(img_buffer, format='png')
            img_buffer.seek(0)
            img_base64 = base64.b64encode(img_buffer.getvalue()).decode('utf-8')
            plt.close()

            # Get analysis from Groq API
            analysis = analyze_with_groq(unique_defect_types)
            analysis_html = markdown.markdown(analysis)

            return jsonify({
                "image": img_base64,
                "analysis": analysis_html,
                "defects": unique_defect_types
            })
        except Exception as e:
            print(f"Error processing image: {e}")
            return jsonify({"error": "Error processing the image. Please try again."}), 500

@app.route('/query', methods=['POST'])
def handle_query():
    data = request.json
    user_query = data.get('query')
    defects = data.get('defects', [])
    
    if not user_query:
        return jsonify({"error": "No query provided"}), 400
    
    try:
        response = analyze_with_groq(defects, user_query)
        response_html = markdown.markdown(response)
        return jsonify({"response": response_html})
    except Exception as e:
        print(f"Error processing query: {e}")
        return jsonify({"error": "Error processing your request. Please try again."}), 500

if __name__ == '__main__':
    app.run(debug=True)