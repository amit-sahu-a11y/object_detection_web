from flask import Flask, request, jsonify, render_template
import cv2
import torch
import numpy as np
import os
from werkzeug.utils import secure_filename

# Initialize Flask app
app = Flask(__name__)

# Load YOLOv5 model
model = torch.hub.load('ultralytics/yolov5', 'yolov5s')

# Define upload folder
from flask import send_from_directory

@app.route('/uploads/<filename>')
def uploaded_file(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)

UPLOAD_FOLDER = 'uploads'
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Function to detect persons in an image
def detect_persons(image_path):
    # Load image
    img = cv2.imread(image_path)
    
    # Perform detection
    results = model(img)
    
    # Parse results
    results = results.pandas().xyxy[0]  # Convert results to pandas DataFrame
    
    # Filter results to only include persons
    persons = results[results['name'] == 'person']
    
    # Draw bounding boxes on the image
    for _, person in persons.iterrows():
        x1, y1, x2, y2 = int(person['xmin']), int(person['ymin']), int(person['xmax']), int(person['ymax'])
        cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
    
    # Save the output image
    output_path = os.path.join(app.config['UPLOAD_FOLDER'], 'output.jpg')
    cv2.imwrite(output_path, img)
    
    return output_path

# Home route
@app.route('/')
def home():
    return render_template('index.html')

# API route for image upload and detection
@app.route('/detect', methods=['POST'])
def detect():
    if 'file' not in request.files:
        return jsonify({"error": "No file uploaded"}), 400
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({"error": "No file selected"}), 400
    
    # Save the uploaded file
    filename = secure_filename(file.filename)
    file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    file.save(file_path)
    
    # Perform person detection
    output_path = detect_persons(file_path)
    
    # Return the output image path
    return jsonify({"output_image": output_path})

# Run the Flask app
if __name__ == '__main__':
    app.run(debug=True)