

import os
import subprocess
import logging
from flask import Flask, render_template, request, jsonify
import random
from flask_cors import CORS
from werkzeug.utils import secure_filename
import requests
import threading
import queue

app = Flask(__name__)
CORS(app)  # Enable CORS for all routes

# Configure logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

# Configure upload folder
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
UPLOAD_FOLDER = os.path.join(BASE_DIR, 'uploads')
ALLOWED_EXTENSIONS = {'mp4', 'avi', 'mov'}

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 300 * 1024 * 1024  # 300MB max file size

# Ensure upload directory exists
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Thread-safe data storage
dirty_data_queue = queue.Queue()

def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/')
def dashboard():
    return render_template('dashboard.html')

@app.route('/upload', methods=['POST'])
def upload_video():
    try:
        logger.debug("Upload request received")
        
        # Clear the queue for new data
        while not dirty_data_queue.empty():
            dirty_data_queue.get()
        
        if 'video' not in request.files:
            logger.error("No file part in the request")
            return jsonify({"error": "No file part"}), 400
        
        file = request.files['video']
        
        if file.filename == '':
            logger.error("No selected file")
            return jsonify({"error": "No selected file"}), 400
        
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            
            file.save(filepath)
            logger.info(f"File saved successfully: {filepath}")
            
            # Update ROI script with new video path
            update_roi_script(filepath)
            
            # Run ROI script in a separate thread
            threading.Thread(target=run_roi_script, daemon=True).start()
            
            return jsonify({
                "message": "Video uploaded and processing started", 
                "filename": filename
            }), 200
        
        logger.error("File type not allowed")
        return jsonify({"error": "File type not allowed"}), 400

    except Exception as e:
        logger.error(f"Unexpected error in upload: {str(e)}")
        return jsonify({"error": f"Unexpected error: {str(e)}"}), 500

@app.route('/receive_dirty_data', methods=['POST'])
def receive_dirty_data():
    try:
        data = request.json
        if "dirty_segments_data" in data:
            for segment in data["dirty_segments_data"]:
                dirty_data_queue.put(segment)
            return jsonify({"message": "Data received successfully"}), 200
        else:
            return jsonify({"error": "Invalid data format"}), 400
    except Exception as e:
        logger.error(f"Error receiving data: {str(e)}")
        return jsonify({"error": f"Error receiving data: {str(e)}"}), 500

@app.route('/dirty_boxes', methods=['GET'])
def get_dirty_boxes():
    dirty_data = []
    while not dirty_data_queue.empty():
        dirty_data.append(dirty_data_queue.get())
    
    labels = [entry["frame"] for entry in dirty_data]
    values = [entry["dirty_segments"] for entry in dirty_data]
    
    return jsonify({"labels": labels, "values": values})

def update_roi_script(video_path):
    try:
        # config_dir = os.path.dirname('video_path.txt')
        # os.makedirs(config_dir, exist_ok=True)    
        with open('video_path.txt', 'w') as f:
            f.write(video_path)
        logger.info(f"Video path written to config: {video_path}")
    except Exception as e:
        logger.error(f"Error updating ROI script: {str(e)}")
        raise

def run_roi_script():
    try:
        result = subprocess.run(
            ['python3', 'ROI.py'], 
            capture_output=True, 
            text=True
        )
        
        logger.info("ROI Script STDOUT: " + result.stdout)
        logger.error("ROI Script STDERR: " + result.stderr)
        
        if result.returncode != 0:
            logger.error(f"ROI script failed with return code {result.returncode}")
    
    except Exception as e:
        logger.error(f"Error running ROI script: {str(e)}")

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
