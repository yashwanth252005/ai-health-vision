"""
Lung Cancer Classification Web Interface
Gradio-based web app with AI medical assistant (Gemini)
"""

import gradio as gr
import numpy as np
import tensorflow as tf
from tensorflow import keras
from PIL import Image
import os

# ========================================================================
# Configuration
# ========================================================================
MODEL_PATH = "training/lung_cancer_final.h5"
CLASS_NAMES = ['Benign', 'Malignant', 'Normal']
IMAGE_SIZE = (224, 224)

# Model accuracy from Kaggle training
MODEL_ACCURACY = 91.82
CLASS_ACCURACIES = {
    'Benign': 54.2,
    'Malignant': 98.2,
    'Normal': 94.0
}

# ========================================================================
# Load Model
# ========================================================================
print("Loading model...")
try:
    model = keras.models.load_model(MODEL_PATH)
    print("✅ Model loaded successfully!")
except Exception as e:
    print(f"❌ Error loading model: {e}")
    model = None

# ========================================================================
# Prediction Function
# ========================================================================
def predict_lung_cancer(image):
    """
    Predict lung cancer class from CT scan image
    
    Args:
        image: PIL Image or numpy array
        
    Returns:
        tuple: (prediction_text, confidence_dict, color_html)
    """
    if model is None:
        return "❌ Model not loaded!", {}, "<p>Error: Model file not found</p>"
    
    try:
        # Preprocess image
        if isinstance(image, np.ndarray):
            img = Image.fromarray(image.astype('uint8'))
        else:
            img = image
        
        # Resize and normalize
        img = img.resize(IMAGE_SIZE)
        img_array = np.array(img) / 255.0
        img_array = np.expand_dims(img_array, axis=0)
        
        # Predict
        predictions = model.predict(img_array, verbose=0)[0]
        
        # Get predicted class
        predicted_class_idx = np.argmax(predictions)
        predicted_class = CLASS_NAMES[predicted_class_idx]
        confidence = predictions[predicted_class_idx] * 100
        
        # Create confidence dictionary for all classes
        confidence_dict = {
            CLASS_NAMES[i]: float(predictions[i] * 100)
            for i in range(len(CLASS_NAMES))
        }
        
        # Color coding
        colors = {
            'Normal': '#28a745',    # Green
            'Benign': '#ffc107',    # Yellow
            'Malignant': '#dc3545'  # Red
        }
        
        color = colors[predicted_class]
        
        # Create result HTML
        result_html = f"""
        <div style='padding: 20px; border-radius: 10px; background: {color}20; border: 3px solid {color};'>
            <h2 style='color: {color}; margin: 0;'>
                🔬 Prediction: {predicted_class}
            </h2>
            <p style='font-size: 24px; margin: 10px 0; font-weight: bold;'>
                Confidence: {confidence:.2f}%
            </p>
            <hr style='border-color: {color};'>
            <h3>All Class Probabilities:</h3>
            <ul style='font-size: 18px;'>
                <li><strong>Benign (Non-cancerous):</strong> {predictions[0]*100:.2f}%</li>
                <li><strong>Malignant (Cancer):</strong> {predictions[1]*100:.2f}%</li>
                <li><strong>Normal (Healthy):</strong> {predictions[2]*100:.2f}%</li>
            </ul>
            <hr style='border-color: {color};'>
            <p style='font-size: 14px; color: #666;'>
                <strong>Model Accuracy:</strong> {MODEL_ACCURACY}% overall<br>
                <strong>Class-specific accuracy:</strong> 
                Benign: {CLASS_ACCURACIES['Benign']}%, 
                Malignant: {CLASS_ACCURACIES['Malignant']}%, 
                Normal: {CLASS_ACCURACIES['Normal']}%
            </p>
        </div>
        """
        
        # Medical interpretation
        interpretations = {
            'Normal': """
            <div style='background: #d4edda; padding: 15px; border-radius: 10px; margin-top: 10px; border: 2px solid #28a745;'>
                <h3 style='color: #0d4d1f;'>✅ Good News!</h3>
                <p style='color: #0d4d1f; font-size: 16px; font-weight: 500;'>The scan appears <strong>NORMAL</strong> with no signs of abnormalities detected.</p>
                <p style='color: #0d4d1f; font-size: 14px;'><em>Note: This is AI prediction. Always consult a healthcare professional for diagnosis.</em></p>
            </div>
            """,
            'Benign': """
            <div style='background: #fff3cd; padding: 15px; border-radius: 10px; margin-top: 10px; border: 2px solid #ff9800;'>
                <h3 style='color: #664d03;'>⚠️ Benign Condition Detected</h3>
                <p style='color: #664d03; font-size: 16px; font-weight: 500;'>The scan shows a <strong style='color: #664d03;'>BENIGN (non-cancerous)</strong> condition.</p>
                <p style='color: #664d03; font-size: 16px; font-weight: 500;'>While not cancerous, medical consultation is recommended for proper evaluation and monitoring.</p>
                <p style='color: #664d03; font-size: 14px;'><em>Note: This is AI prediction. Always consult a healthcare professional for diagnosis.</em></p>
            </div>
            """,
            'Malignant': """
            <div style='background: #f8d7da; padding: 15px; border-radius: 10px; margin-top: 10px; border: 2px solid #dc3545;'>
                <h3 style='color: #5a0a13;'>🚨 URGENT: Malignant (Cancer) Detected</h3>
                <p style='color: #5a0a13; font-size: 16px; font-weight: 500;'>The scan indicates a potential <strong style='color: #5a0a13;'>MALIGNANT (cancerous)</strong> condition.</p>
                <p style='color: #5a0a13; font-size: 16px; font-weight: 500;'><strong style='color: #5a0a13;'>IMPORTANT:</strong> Seek immediate medical attention from an oncologist.</p>
                <p style='color: #5a0a13; font-size: 16px; font-weight: 500;'>Early detection greatly improves treatment outcomes.</p>
                <p style='color: #5a0a13; font-size: 14px;'><em>Note: This is AI prediction. Always consult a healthcare professional for diagnosis.</em></p>
            </div>
            """
        }
        
        result_html += interpretations[predicted_class]
        
        return result_html, confidence_dict
        
    except Exception as e:
        error_msg = f"❌ Error during prediction: {str(e)}"
        return error_msg, {}

# ========================================================================
# Gemini AI Integration
# ========================================================================
def format_gemini_response(text):
    """
    Convert Gemini's markdown-style response to formatted HTML
    """
    import re
    
    # Replace **bold** with <strong>
    text = re.sub(r'\*\*(.+?)\*\*', r'<strong>\1</strong>', text)
    
    # Replace *italic* with <em>
    text = re.sub(r'\*(.+?)\*', r'<em>\1</em>', text)
    
    # Convert bullet points (• or *) to HTML list items
    lines = text.split('\n')
    formatted_lines = []
    in_list = False
    
    for line in lines:
        line = line.strip()
        
        # Check if line is a bullet point
        if line.startswith('•') or line.startswith('*'):
            if not in_list:
                formatted_lines.append('<ul style="color: #000000; font-size: 15px; line-height: 1.8; margin: 10px 0; padding-left: 25px;">')
                in_list = True
            # Remove bullet and wrap in <li>
            clean_line = line.lstrip('•*').strip()
            formatted_lines.append(f'<li style="margin: 8px 0; color: #000000;">{clean_line}</li>')
        else:
            if in_list:
                formatted_lines.append('</ul>')
                in_list = False
            
            # Check if line is a heading (starts with number or capitalized)
            if line and (line[0].isdigit() or line.isupper() or line.startswith('#')):
                # Remove # if present
                clean_line = line.lstrip('#').strip()
                formatted_lines.append(f'<h4 style="color: #000000; margin-top: 20px; margin-bottom: 10px; font-size: 18px; font-weight: 700; border-bottom: 2px solid #1976d2; padding-bottom: 5px;">{clean_line}</h4>')
            elif line:
                formatted_lines.append(f'<p style="color: #000000; font-size: 15px; line-height: 1.8; margin: 12px 0;">{line}</p>')
    
    if in_list:
        formatted_lines.append('</ul>')
    
    return '\n'.join(formatted_lines)

def get_ai_recommendations(prediction_class, confidence, image):
    """
    Get AI-powered medical recommendations using Google Gemini
    
    NOTE: This is a placeholder. To enable:
    1. Install: pip install google-generativeai
    2. Get API key from: https://makersuite.google.com/app/apikey
    3. Set environment variable: GEMINI_API_KEY
    """
    
    # Check if Gemini is available
    try:
        import google.generativeai as genai
        api_key = os.getenv('GEMINI_API_KEY')
        
        if not api_key:
            return """
            <div style='background: #f0f0f0; padding: 15px; border-radius: 10px;'>
                <h3 style='color: #333;'>🤖 AI Medical Assistant (Not Configured)</h3>
                <p style='color: #333; font-size: 15px;'>To enable AI recommendations:</p>
                <ol style='color: #333; font-size: 14px;'>
                    <li>Get Gemini API key from <a href='https://makersuite.google.com/app/apikey' target='_blank' style='color: #0066cc;'>Google AI Studio</a></li>
                    <li>Set environment variable: <code style='background: #e0e0e0; padding: 2px 5px; border-radius: 3px;'>GEMINI_API_KEY=your_key_here</code></li>
                    <li>Restart the application</li>
                </ol>
            </div>
            """
        
        # Configure Gemini
        genai.configure(api_key=api_key)
        gemini_model = genai.GenerativeModel('gemini-2.0-flash')  # Updated to latest available model
        
        # Create prompt
        prompt = f"""
        You are a medical AI assistant. Based on this lung CT scan analysis:
        
        - Prediction: {prediction_class}
        - Confidence: {confidence:.2f}%
        
        Provide:
        1. Brief explanation of what this means
        2. Next steps the patient should take
        3. General lifestyle recommendations
        4. When to seek emergency care
        
        Keep it concise, compassionate, and emphasize the importance of consulting real doctors.
        Format your response in clear sections with bullet points.
        
        IMPORTANT: Always emphasize this is AI prediction and professional medical consultation is required.
        """
        
        # Get response
        response = gemini_model.generate_content(prompt)
        
        # Format the response for better readability
        formatted_response = format_gemini_response(response.text)
        
        return f"""
        <div style='background: #e3f2fd; padding: 20px; border-radius: 10px; margin-top: 15px; border: 3px solid #1976d2;'>
            <h3 style='color: #001f3f; margin-top: 0; font-size: 20px; font-weight: 700;'>🤖 AI Medical Assistant Recommendations</h3>
            
            <!-- Scrollable Content Container -->
            <div style='max-height: 400px; overflow-y: auto; overflow-x: hidden; 
                        background: #006fffb8; padding: 15px; border-radius: 8px; 
                        border: 2px solid #1976d2; margin: 10px 0;
                        box-shadow: inset 0 2px 4px rgba(0,0,0,0.1);'>
                <div style='color: #001f3f;'>
                    {formatted_response}
                </div>
            </div>
            
            <!-- Scroll Indicator -->
            <p style='font-size: 12px; color: #1976d2; margin: 5px 0; text-align: center; font-style: italic;'>
                ↕️ Scroll to read full recommendations
            </p>
            
            <hr style='border-color: #1976d2; margin: 15px 0;'>
            <p style='font-size: 13px; color: #001f3f; margin-bottom: 0; font-weight: 600;'>
                <strong>⚠️ Disclaimer:</strong> This AI-generated advice is for informational purposes only. 
                Always consult qualified healthcare professionals for medical decisions.
            </p>
        </div>
        """
        
    except ImportError:
        return """
        <div style='background: #f0f0f0; padding: 15px; border-radius: 10px;'>
            <h3 style='color: #333;'>🤖 AI Medical Assistant (Not Installed)</h3>
            <p style='color: #333; font-size: 15px;'>To enable AI recommendations, install:</p>
            <code style='background: #e0e0e0; padding: 5px; border-radius: 3px; color: #333;'>pip install google-generativeai</code>
        </div>
        """
    except Exception as e:
        return f"""
        <div style='background: #fff3cd; padding: 15px; border-radius: 10px; border: 2px solid #856404;'>
            <h3 style='color: #856404;'>⚠️ AI Assistant Error</h3>
            <p style='color: #856404; font-size: 15px;'><strong>Error:</strong> {str(e)}</p>
            <p style='color: #856404; font-size: 14px;'>The AI assistant encountered an issue. Your prediction results are still valid.</p>
        </div>
        """

# ========================================================================
# Combined Prediction + AI Recommendations
# ========================================================================
def predict_with_ai_assistant(image):
    """Complete prediction with AI recommendations"""
    
    # Get prediction
    result_html, confidence_dict = predict_lung_cancer(image)
    
    if confidence_dict:
        # Get predicted class
        predicted_class = max(confidence_dict, key=confidence_dict.get)
        confidence = confidence_dict[predicted_class]
        
        # Get AI recommendations
        ai_recommendations = get_ai_recommendations(predicted_class, confidence, image)
        
        # Combine results
        full_result = result_html + ai_recommendations
    else:
        full_result = result_html
    
    return full_result

# ========================================================================
# Gradio Interface
# ========================================================================
def create_interface():
    """Create Gradio web interface"""
    
    with gr.Blocks(title="Lung Cancer Classification AI", theme=gr.themes.Soft()) as demo:
        gr.Markdown(
            """
            # 🫁 Lung Cancer Classification System
            ## AI-Powered CT Scan Analysis with Medical Assistant
            
            Upload a lung CT scan image to get instant AI-powered classification and personalized medical guidance.
            
            **Model Accuracy:** 91.82% overall
            - Malignant Detection: 98.2% ✅
            - Normal Detection: 94.0% ✅
            - Benign Detection: 54.2% ⚠️
            
            ---
            """
        )
        
        with gr.Row():
            with gr.Column(scale=1):
                input_image = gr.Image(
                    type="pil",
                    label="Upload Lung CT Scan",
                    sources=["upload", "clipboard"]
                )
                
                predict_btn = gr.Button(
                    "🔬 Analyze CT Scan",
                    variant="primary",
                    size="lg"
                )
                
                gr.Markdown(
                    """
                    ### 📋 Instructions:
                    1. Upload a lung CT scan image (JPG, PNG)
                    2. Click "Analyze CT Scan"
                    3. View prediction and AI recommendations
                    
                    ### ⚕️ Medical Disclaimer:
                    This AI tool is for educational/research purposes.
                    **Always consult qualified healthcare professionals** for medical decisions.
                    """
                )
            
            with gr.Column(scale=2):
                output_result = gr.HTML(label="Analysis Results")
        
        # Examples section
        gr.Markdown("### 📸 Sample Images (for testing)")
        gr.Markdown("*Upload your own CT scan images or test with sample images from your dataset*")
        
        # Connect prediction
        predict_btn.click(
            fn=predict_with_ai_assistant,
            inputs=input_image,
            outputs=output_result
        )
        
        # Footer
        gr.Markdown(
            """
            ---
            ### 🔬 About This Model
            - **Architecture:** CNN (Convolutional Neural Network)
            - **Training:** Kaggle GPU P100, 50 epochs
            - **Dataset:** 1,097 lung CT scan images (balanced via oversampling)
            - **Classes:** Benign, Malignant, Normal
            - **Challenge Solved:** Severe class imbalance (4.67x ratio)
            
            ### 🤖 AI Assistant
            Powered by Google Gemini AI for personalized medical guidance.
            
            **⚠️ Important:** AI predictions are not a substitute for professional medical diagnosis.
            """
        )
    
    return demo

# ========================================================================
# Main
# ========================================================================
if __name__ == "__main__":
    print("\n" + "=" * 70)
    print("🫁 LUNG CANCER CLASSIFICATION WEB APP")
    print("=" * 70)
    print(f"Model Path: {MODEL_PATH}")
    print(f"Model Status: {'✅ Loaded' if model is not None else '❌ Not Found'}")
    print(f"Model Accuracy: {MODEL_ACCURACY}%")
    print("=" * 70)
    
    if model is None:
        print("\n❌ ERROR: Model file not found!")
        print(f"Expected location: {MODEL_PATH}")
        print("Please ensure lung_cancer_final.h5 is in the training/ folder")
        print("=" * 70)
    
    # Create and launch interface
    demo = create_interface()
    
    print("\n🚀 Launching web interface...")
    print("Access at: http://localhost:7860")
    print("To share publicly, the app will generate a public URL")
    print("=" * 70)
    
    demo.launch(
        server_name="0.0.0.0",  # Allow external access
        server_port=7860,
        share=False,  # Set to True to create public link
        show_error=True
    )
