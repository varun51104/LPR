from difflib import get_close_matches
import re
from ultralytics import YOLO
yolo_model = YOLO("license_plate_detector.pt")  

import os
import cv2
from datetime import datetime, timedelta, timezone
from fuzzywuzzy import fuzz
import pytesseract
import requests
from PIL import Image
from flask import Flask, jsonify, redirect, render_template, Response, request, session, url_for
import threading
from pymongo import MongoClient

app = Flask(__name__)

# Initialize database connection
def initialize_database():
    """Initialize database with consistent configuration"""
    try:
        client = MongoClient('mongodb://localhost:27017/', serverSelectionTimeoutMS=5000)
        client.server_info()  # Force connection test
        print("✅ Connected to MongoDB")
        
        db = client.vehicle
        vehicles_collection = db['vehicle']
        history_collection = db['history']
        
        # Create indexes for better performance
        try:
            history_collection.create_index([("plate_number", 1), ("timestamp", -1)])
            print("✅ Database indexes created")
        except Exception as idx_error:
            print(f"⚠️ Index creation warning: {idx_error}")
        
        return db, vehicles_collection, history_collection
        
    except Exception as e:
        print(f"❌ Failed to connect to MongoDB: {e}")
        return None, None, None

# Initialize database collections
print("Initializing database...")
db, vehicles_collection, history_collection = initialize_database()

# Exit if database connection failed
if  db is  None:
    print("Exiting due to database connection failure")
    exit(1)

def log_unregistered_plate(plate_text):
    """Logs an unregistered plate as 'Unknown' with current timestamp"""
    try:
        # Check if plate is already in registered vehicles
        if not vehicles_collection.find_one({"plate_number": plate_text}):
            # Check for recent detection (within 5 minutes)
            recent_detection = history_collection.find_one(
                {"plate_number": plate_text}, 
                sort=[("timestamp", -1)]
            )
            
            should_log = True
            if recent_detection:
                time_diff = (datetime.now() - recent_detection['timestamp']).total_seconds()
                if time_diff < 300:  # 5 minutes
                    # Only prevent logging if the last detection was also unregistered
                    if recent_detection.get('status') == 'unregistered':
                        should_log = False
                        print(f"[DEBUG] Skipping duplicate unknown plate: {plate_text}")
            
            if should_log:
                result = history_collection.insert_one({
                    "plate_number": plate_text,
                    "owner_name": "Unknown",
                    "make": "Unknown",
                    "model": "Unknown",
                    "color": "Unknown",
                    "timestamp": datetime.now(),
                    "status": "unregistered"
                })
                print(f"✅ Logged unregistered plate: {plate_text} with ID: {result.inserted_id}")
    except Exception as e:
        print(f"❌ Failed to log unregistered plate: {e}")

def load_vehicle_database():
    """Load vehicle database from MongoDB"""
    try:
        database = {}
        for vehicle in vehicles_collection.find():
            plate_number = vehicle['plate_number']
            database[plate_number] = {
                "owner_name": vehicle['owner_name'],
                "make": vehicle['make'],
                "model": vehicle['model'],
                "color": vehicle['color']
            }
        
        print(f"✅ Vehicle database loaded successfully. {len(database)} vehicles found.")
        return database
    except Exception as e:
        print(f"❌ Failed to load vehicle database: {e}")
        return {}

# Load vehicle database
vehicle_database = load_vehicle_database()

# List to store recognized license plate information
recognized_plates = []

# Path to Tesseract OCR executable
pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

def perform_ocr(img):
    custom_config = r'--oem 3 --psm 6'  
    plate_number = pytesseract.image_to_string(img, config=custom_config)
    plate_number = plate_number.upper().replace(" ", "").replace("\n", "")
    plate_number = plate_number.replace("O", "0").replace("|", "").replace(":", "")
    return plate_number.strip()

def sanitize_plate_text(text):
    """Clean and normalize the plate text with better validation."""
    if not text:
        return ""
    
    text = text.strip().upper()
    text = text.replace(" ", "").replace("|", "").replace("\n", "").replace("\r", "")
    text = text.replace("O", "0")  # Common OCR mistake
    text = text.replace(":", "").replace(";", "").replace(",", "")
    
    # Keep only alphanumeric characters
    text = ''.join(filter(str.isalnum, text))
    
    return text

def fuzzy_match_plate(plate_text, vehicle_db, threshold=85):
    """Fuzzy match OCR result with keys in the vehicle database."""
    best_match = None
    best_score = 0
    for plate in vehicle_db.keys():
        score = fuzz.ratio(plate_text, plate)
        if score > best_score and score >= threshold:
            best_score = score
            best_match = plate
    return best_match

# Function to process the video stream
def process_video():
    cap = cv2.VideoCapture(0)
    print("🎥 Starting video capture...")

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        results = yolo_model(frame)[0]

        for box in results.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            conf = float(box.conf[0])
            if conf < 0.6:
                continue

            plate_roi = frame[y1:y2, x1:x2]
            gray_plate = cv2.cvtColor(plate_roi, cv2.COLOR_BGR2GRAY)
            gray_plate = cv2.resize(gray_plate, None, fx=2, fy=2, interpolation=cv2.INTER_LINEAR)
            _, thresh_plate = cv2.threshold(gray_plate, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            pil_image = Image.fromarray(thresh_plate)

            raw_text = pytesseract.image_to_string(pil_image, config=r'--oem 3 --psm 6')
            plate_number = sanitize_plate_text(raw_text)

            # Debug OCR output
            if plate_number:
                print(f"🔍 OCR detected: '{plate_number}'")

            # Skip if plate number is too short
            if len(plate_number) < 3:
                continue

            # Check for exact match first
            vehicle_info = vehicle_database.get(plate_number)
            matched_plate = plate_number
            
            # If no exact match, try fuzzy matching
            if not vehicle_info:
                matched = fuzzy_match_plate(plate_number, vehicle_database)
                if matched:
                    matched_plate = matched
                    vehicle_info = vehicle_database[matched]
                    print(f"🔗 Fuzzy matched '{plate_number}' to '{matched}'")

            # Time-based duplicate prevention (5 minutes)
           # Time-based duplicate prevention (5 minutes) - but only for same status
            recent_detection = None
            try:
                recent_detection = history_collection.find_one(
                    {"plate_number": matched_plate}, 
                    sort=[("timestamp", -1)]
                )
            except Exception as e:
                print(f"❌ Database query error: {e}")

            should_insert = True
            if recent_detection:
                time_diff = (datetime.now() - recent_detection['timestamp']).total_seconds()
                if time_diff < 300:  # 5 minutes
                    # Only prevent insertion if the status matches what we're about to insert
                    current_status = "registered" if vehicle_info else "unregistered"
                    if recent_detection.get('status') == current_status:
                        should_insert = False
                        print(f"[DEBUG] Skipping duplicate {current_status} plate: {matched_plate}")

            # Insert into database if conditions are met
            if should_insert:
                try:
                    if vehicle_info:
                        # Registered vehicle
                        insert_data = {
                            "plate_number": matched_plate,
                            "owner_name": vehicle_info['owner_name'],
                            "make": vehicle_info['make'],
                            "model": vehicle_info['model'],
                            "color": vehicle_info['color'],
                            "timestamp": datetime.now(),
                            "status": "registered"
                        }
                        
                        result = history_collection.insert_one(insert_data)
                        print(f"✅ INSERTED registered plate: {matched_plate} (ID: {result.inserted_id})")
                        
                        # Add to recognized_plates list
                        recognized_plates.append(insert_data)
                        
                        # Display info on frame
                        info_text = f"Owner: {vehicle_info['owner_name']}"
                        cv2.putText(frame, info_text, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                        
                    else:
                        # Unregistered vehicle - call the function
                        log_unregistered_plate(plate_number)
                        cv2.putText(frame, "UNKNOWN VEHICLE", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

                except Exception as e:
                    print(f"❌ Database insertion failed: {e}")

            # Draw bounding box
            color = (0, 255, 0) if vehicle_info else (0, 0, 255)
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            cv2.putText(frame, matched_plate if vehicle_info else plate_number, 
                       (x1, y2 + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

        # Display the processed frame
        ret, jpeg = cv2.imencode('.jpg', frame)
        frame_bytes = jpeg.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n\r\n')

    cap.release()

# REST OF YOUR FLASK ROUTES GO HERE...
# (Keep all your existing @app.route functions exactly as they are)

    pass

# Function to run the video processing in a separate thread
def run_video_processing():
    with app.app_context():
        app.video_thread = threading.Thread(target=process_video)
        app.video_thread.start()
    # Your existing code for running video processing
    pass

def get_utc_now():
    return datetime.now(timezone.utc)


@app.route('/video_feed')
def video_feed():
    return Response(process_video(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')


#login function
app.secret_key = 'admin'  # Add a secret key for session management

# Define admin credentials
ADMIN_USERNAME = 'admin'
ADMIN_PASSWORD = 'admin'

# Add a login route
@app.route('/', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']
        if username == ADMIN_USERNAME and password == ADMIN_PASSWORD:
            session['logged_in'] = True  # Set session variable indicating user is logged in
            return redirect(url_for('index'))  # Redirect to index.html
        else:
            return render_template('login.html', error=True)
    else:
        return render_template('login.html', error=False)

@app.route('/logout')
def logout():
    session.pop('logged_in', None)  # Clear session variable
    return redirect(url_for('index'))


@app.route('/add_vehicle', methods=['POST'])
def add_vehicle():

    if 'logged_in' not in session:
        return redirect(url_for('login'))  # Redirect to login page if not logged in

    plate_number = request.form['plate_number']
    owner_name = request.form['owner_name']
    make = request.form['make']
    model = request.form['model']
    color = request.form['color']

    # Insert the new vehicle information into the MongoDB database
    vehicles_collection.insert_one({
        "plate_number": plate_number,
        "owner_name": owner_name,
        "make": make,
        "model": model,
        "color": color
    })

    global vehicle_database
    vehicle_database = load_vehicle_database()
    return redirect(url_for('index'))
    # return redirect(url_for('add_vehicle'))

@app.route('/recognized_plates')
def recognized_plates_page():
    return render_template('recognized_plates.html', recognized_plates=recognized_plates)


@app.route('/index')
def index():
    try:
        # Calculate the timestamp for 24 hours ago   
        past_24_hours = datetime.now() - timedelta(hours=24)
        
        # Query the history collection for documents within the past 24 hours
        data = history_collection.find({"timestamp": {"$gte": past_24_hours}})
        
        return render_template('index.html', data=data)
        # return render_template('index.html', data=database)
    
    except Exception as e:
        app.logger.error(f"An error occurred while fetching data from the database: {e}")
        return "An error occurred while fetching data from the database. Please check the logs for more information."
    

@app.route('/records')
def records():

    try:
        database = {}
        records_collection = db['vehicle']
        database = records_collection.find()

        # Logging successful fetch
        app.logger.info("Records fetched successfully")

        # Returning the database for rendering in the template
        return render_template('records.html', rec=database)

    except Exception as e:
        app.logger.error(f"An error occurred while fetching records from the database: {e}")
        return "An error occurred while fetching records from the database. Please check the logs for more information."


if __name__ == '__main__':
    run_video_processing()
    app.run(debug=True)


