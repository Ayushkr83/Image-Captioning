from flask import Flask, request, render_template, send_from_directory
from PIL import Image
from transformers import AutoProcessor, BlipForConditionalGeneration
import os

app = Flask(__name__)

# Load the pretrained processor and model
processor = AutoProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")

# Function to generate caption
def generate_caption(image_path):
    # Open and convert the image to RGB format
    image = Image.open(image_path).convert('RGB')
    
    # Generate caption
    text = "the image of"
    inputs = processor(images=image, text=text, return_tensors="pt")
    outputs = model.generate(**inputs, max_length=50)
    
    # Decode the generated tokens to text
    caption = processor.decode(outputs[0], skip_special_tokens=True)
    return caption

@app.route('/', methods=['GET', 'POST'])
def index():
    caption = None
    image_filename = None
    if request.method == 'POST':
        if 'file' not in request.files:
            return "No file uploaded"
        file = request.files['file']
        if file.filename == '':
            return "No file selected"
        if file:
            # Save the uploaded file
            upload_folder = 'uploads'
            os.makedirs(upload_folder, exist_ok=True)
            image_filename = file.filename
            image_path = os.path.join(upload_folder, image_filename)
            file.save(image_path)
            
            # Generate caption
            caption = generate_caption(image_path)
    
    return render_template('index.html', caption=caption, image_filename=image_filename)

# Route to serve uploaded images
@app.route('/uploads/<filename>')
def uploaded_file(filename):
    return send_from_directory('uploads', filename)

if __name__ == '__main__':
    app.run(debug=True)