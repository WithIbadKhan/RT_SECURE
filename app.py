
import os
import uuid
from flask import Flask, request, jsonify, session
import io
from transformers import AutoTokenizer, AutoModelForTokenClassification, pipeline
from presidio_analyzer import AnalyzerEngine, RecognizerResult, EntityRecognizer
from flask_pymongo import PyMongo
from flask_cors import CORS
import pytesseract
import jwt
from datetime import timedelta
import secrets
from functools import wraps
from werkzeug.security import generate_password_hash, check_password_hash
from jwt import ExpiredSignatureError, InvalidTokenError
from dotenv import load_dotenv
import random
import string
from flask_mail import Mail, Message
import datetime
from collections import deque
from spacy import load
import time
import json
import re
import logging
from presidio_analyzer import Pattern, PatternRecognizer, AnalyzerEngine
from presidio_anonymizer import AnonymizerEngine, OperatorConfig
import os
import traceback
import dotenv
import pandas as pd
import textract
import fitz
import uuid
from PIL import Image
from flask import Flask, request, jsonify
from flask_pymongo import PyMongo
from datetime import datetime
from datetime import datetime, timedelta
from presidio_analyzer.nlp_engine.spacy_nlp_engine import SpacyNlpEngine
import spacy
from transformers import pipeline
from transformers import AutoTokenizer, AutoModelForTokenClassification
from transformers import AutoTokenizer, AutoModelForTokenClassification
from transformers import pipeline

dotenv.load_dotenv()
logger = logging.getLogger("presidio-script")

app = Flask(__name__)
CORS(app)

MONGO_URI = os.getenv("MONGO_URI")
MAIL_SERVER = os.getenv("MAIL_SERVER")
MAIL_PORT = os.getenv("MAIL_PORT")
MAIL_USE_SSL = os.getenv("MAIL_USE_SSL")
MAIL_USERNAME = os.getenv("MAIL_USERNAME")
MAIL_PASSWORD = os.getenv("MAIL_PASSWORD")
MAIL_DEFAULT_SENDER = os.getenv("MAIL_DEFAULT_SENDER")


app.config['SECRET_KEY'] = secrets.token_hex(16)
app.config['JWT_SECRET_KEY'] = secrets.token_hex(16)
app.config['JWT_EXPIRATION_DELTA'] = timedelta(minutes=60)  # Set your desired expiration time
app.config['PERMANENT_SESSION_LIFETIME'] = timedelta(days=2)

app.config['MONGO_URI'] = MONGO_URI
app.config.update(
    MAIL_SERVER=MAIL_SERVER,
    MAIL_PORT=MAIL_PORT,
    MAIL_USE_SSL=bool(MAIL_USE_SSL),
    MAIL_USERNAME=MAIL_USERNAME,
    MAIL_PASSWORD=MAIL_PASSWORD,
    MAIL_DEFAULT_SENDER=(MAIL_DEFAULT_SENDER.split(':')[0], MAIL_DEFAULT_SENDER.split(':')[1])
)

mongo = PyMongo(app)
mail = Mail(app)

# MongoDB Configuration
app.config['MONGO_URI'] = os.getenv("MONGO_URI")
mongo = PyMongo(app)

analyzer = AnalyzerEngine()
anonymizer = AnonymizerEngine()


# Pre-initialize SpaCy and RoBERTa models
spacy_model = load('en_core_web_lg')
# tokenizer = AutoTokenizer.from_pretrained("Jean-Baptiste/roberta-large-ner-english")
# model = AutoModelForTokenClassification.from_pretrained("Jean-Baptiste/roberta-large-ner-english")
# roberta_pipeline = pipeline('ner', model=model, tokenizer=tokenizer, aggregation_strategy="simple")

# Initialize logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

predefined_entities = [
    "CREDIT_CARD", "MEDICAL_LICENSE", "ORGANIZATION", "MONEY", "AGE", "URL", "NRP", "AU ACN",
    "SG_NRIC_FIN", "AU TFN", "US_ITIN", "UK_NHS", "AU_MEDICARE", "US_SSN", "US_DRIVER_LIC",
    "IN_VEHICLE_REGISTRATION", "DATE_TIME", "IN_PAN", "AU_ABN", "US_PASSPORT", "EMAIL", "PERSON",
    "IN_AADHAAR", "PHONE_NUMBER", "IBAN_CODE", "IP_ADDRESS", "US_BANK_NUMBER", "CRYPTO", "LOCATION",
    "GENERIC_PII",
]


def create_default_admin():
    admin_email = "bjohnson197924@gmail.com"
    admin_password = "Admin@1234"
    hashed_password = generate_password_hash(admin_password, method='pbkdf2:sha256')

    if not mongo.db.admin.find_one({'email': admin_email}):
        admin = {
            'email': admin_email,
            'password_hashed': hashed_password,
            'plain_password': admin_password
        }
        mongo.db.admin.insert_one(admin)
        print(f"Default admin created with email: {admin_email} and password: {admin_password}")
    else:
        print("Default admin already exists.")

create_default_admin()

def token_required(f):
    @wraps(f)
    def decorated(*args, **kwargs):
        token = request.headers.get('x-access-token')

        if not token:
            return jsonify({'message': 'Token is missing!'}), 401

        try:
            data = jwt.decode(token, app.config['SECRET_KEY'], algorithms=['HS256'])
        except ExpiredSignatureError:
            return jsonify({'message': 'Token has expired!'}), 401
        except InvalidTokenError as e:
            return jsonify({'message': f'Token is invalid: {str(e)}'}), 401

        return f(*args, **kwargs)

    return decorated

# Signup route
@app.route('/signup', methods=['POST'])
def signup():
    data = request.get_json()

    firstname = data.get('firstname')
    lastname = data.get('lastname')
    email = data.get('email')
    password = data.get('password')
    contact_number = data.get('contact_number')

    if not email or not password:
        return jsonify({'message': 'All fields are required!'}), 400
    
    if email == "admin1234@gmail.com":
        return jsonify({'message': 'Cannot use admin email for signup!'}), 400

    if mongo.db.users.find_one({'email': email}):
        return jsonify({'message': 'Email already exists!'}), 400

    hashed_password = generate_password_hash(password, method='pbkdf2:sha256')

    user = {
        'firstname': firstname,
        'lastname': lastname,
        'email': email,
        'password_hashed': hashed_password,
        'password': password,
        'contact_number': contact_number,
        'status': 'pending'
    }
    mongo.db.users.insert_one(user)

    return jsonify({'message': 'User created successfully!'}), 201

@app.route('/login', methods=['POST'])
def login():
    data = request.get_json()

    if not data or not data.get('email') or not data.get('password'):
        return jsonify({'message': 'Invalid credentials'}), 401

    email = data['email']
    password = data['password']

    user = mongo.db.users.find_one({'email': email})
    admin = mongo.db.admin.find_one({'email': email})

    if not user and not admin:
        return jsonify({'message': 'Invalid credentials'}), 401

    if user:
        if not check_password_hash(user['password_hashed'], password):
            return jsonify({'message': 'Invalid credentials'}), 401

        if user.get('status') != 'approved':
            return jsonify({'message': 'Your account is not approved yet!'}), 403

        token_data = {'identifier': email, 'exp': datetime.utcnow() + timedelta(days=1)}
        token = jwt.encode(token_data, app.config['SECRET_KEY'], algorithm='HS256')
        token_timestamp = datetime.utcnow() + timedelta(hours=24)

        mongo.db.users.update_one({'_id': user['_id']},
                                  {'$set': {'token': token, 'token_timestamp': token_timestamp}})

        return jsonify({'message': 'User login successful!', 'token': token, 'token_timestamp': token_timestamp, 'role': 'user'}), 200

    if admin:
        if not check_password_hash(admin['password_hashed'], password):
            return jsonify({'message': 'Invalid credentials'}), 401

        token_data = {'identifier': email, 'exp': datetime.utcnow() + timedelta(days=1)}
        token = jwt.encode(token_data, app.config['SECRET_KEY'], algorithm='HS256')
        token_timestamp = datetime.utcnow() + timedelta(hours=24)

        mongo.db.admin.update_one({'_id': admin['_id']},
                                  {'$set': {'token': token, 'token_timestamp': token_timestamp}})

        return jsonify({'message': 'Admin login successful!', 'token': token, 'token_timestamp': token_timestamp, 'role': 'admin'}), 200

@app.route('/forgot_password', methods=['POST'])
def forgot_password():
    data = request.json
    email = data.get('email')

    admin_user = mongo.db.admin.find_one({'email': email})
    regular_user = mongo.db.users.find_one({'email': email})

    if admin_user or regular_user:
        role = 'admin' if admin_user else 'user'
        code = ''.join(random.choices(string.digits, k=4))

        recovery_data = {
            'user_id': admin_user['_id'] if admin_user else regular_user['_id'],
            'recovery_code': code,
            'timestamp': datetime.utcnow()
        }

        mongo.db.recovery.insert_one(recovery_data)
        print(f"Recovery code for {email}: {code}")

        send_recovery_email(email, code)

        return jsonify({'message': 'Recovery code sent to your email. Please check your inbox.'}), 200
    else:
        return jsonify({'error': 'Email not found. Please enter a valid email address.'}), 404

@app.route('/verify_recovery_code', methods=['POST'])
def verify_recovery_code():
    data = request.json
    code = data.get('code')

    recovery_data = mongo.db.recovery.find_one({
        'recovery_code': code,
        'timestamp': {'$gte': datetime.utcnow() - timedelta(hours=1)}
    })

    if recovery_data:
        return jsonify({'message': 'Recovery code verified successfully.'}), 200
    else:
        return jsonify({'error': 'Invalid or expired recovery code. Please try again.'}), 400

@app.route('/reset_password', methods=['POST'])
def reset_password():
    data = request.json
    email = data.get('email')
    code = data.get('code')
    new_password = data.get('new_password')
    confirm_password = data.get('confirm_password')

    if new_password != confirm_password:
        return jsonify({'error': 'New password and confirm password do not match. Please try again.'}), 400

    recovery_data = mongo.db.recovery.find_one({'recovery_code': code})

    if recovery_data:
        expiration_time = recovery_data['timestamp'] + timedelta(minutes=5)

        if datetime.utcnow() <= expiration_time:
            user_id = recovery_data['user_id']

            admin_user = mongo.db.admin.find_one({'_id': user_id})
            regular_user = mongo.db.users.find_one({'_id': user_id})

            user_collection = None
            if admin_user:
                user_collection = mongo.db.admin
            elif regular_user:
                user_collection = mongo.db.users
            else:
                return jsonify({'error': 'User not found.'}), 404

            user = user_collection.find_one({'_id': user_id})

            if user:
                hashed_password = generate_password_hash(new_password, method='pbkdf2:sha256')

                user_collection.update_one(
                    {'_id': user_id},
                    {'$set': {'password_hashed': hashed_password, 'plain_password': new_password}}
                )

                mongo.db.recovery.delete_one({'_id': recovery_data['_id']})

                return jsonify({'message': 'Password successfully reset. You can now log in with your new password.'}), 200
            else:
                return jsonify({'error': 'User not found.'}), 404
        else:
            return jsonify({'error': 'Invalid or expired recovery code. Please try again.'}), 400
    else:
        return jsonify({'error': 'Invalid or expired recovery code. Please try again.'}), 400

def send_recovery_email(email, code):
    subject = 'Password Recovery Code'
    body = f'Your recovery code is: {code}'

    try:
        msg = Message(subject, recipients=[email], body=body)
        mail.send(msg)
        print(f"Email sent to {email} with recovery code.")
    except Exception as e:
        print(f"Error sending email: {e}")

@app.route('/edit_profile', methods=['POST'])
@token_required
def edit_profile():
    data = request.get_json()

    email = data.get('email')
    role = data.get('role')
    first_name = data.get('firstname')
    last_name = data.get('lastname')
    address = data.get('address')
    contact_number = data.get('contact_number')

    if not email or not role:
        return jsonify({'message': 'Email and role are required!'}), 400

    user_collection = None
    if role == 'admin':
        user_collection = mongo.db.admin
    elif role == 'user':
        user_collection = mongo.db.users
    else:
        return jsonify({'message': 'Invalid role!'}), 400

    existing_user = user_collection.find_one({'email': email})

    if existing_user:
        # Update existing user information
        update_data = {}
        if first_name:
            update_data['firstname'] = first_name
        if last_name:
            update_data['lastname'] = last_name
        if address:
            update_data['address'] = address
        if contact_number:
            update_data['contact_number'] = contact_number

        user_collection.update_one(
            {'_id': existing_user['_id']},
            {'$set': update_data}
        )
        return jsonify({'message': 'Profile updated successfully!'}), 200
    else:
        # Insert new user information
        new_user = {
            'email': email,
            'firstname': first_name,
            'lastname': last_name,
            'address': address,
            'contact_number': contact_number
        }
        user_collection.insert_one(new_user)
        return jsonify({'message': 'Profile created successfully!'}), 201

@app.route('/get_profile', methods=['POST'])
@token_required
def get_profile():
    data = request.get_json()

    email = data.get('email')
    role = data.get('role')

    if not email or not role:
        return jsonify({'message': 'Email and role are required!'}), 400

    user_collection = None
    if role == 'admin':
        user_collection = mongo.db.admin
    elif role == 'user':
        user_collection = mongo.db.users
    else:
        return jsonify({'message': 'Invalid role!'}), 400

    user = user_collection.find_one({'email': email}, {'_id': 0, 'firstname': 1, 'lastname': 1, 'address': 1, 'contact_number': 1, 'email': 1})

    if not user:
        return jsonify({'message': 'User not found!'}), 404

    return jsonify({'profile': user}), 200

@app.route('/change_password', methods=['POST'])
def change_password():
    data = request.get_json()

    email = data.get('email')
    role = data.get('role')
    current_password = data.get('current_password')
    new_password = data.get('new_password')
    confirm_password = data.get('confirm_password')

    if not email or not current_password or not new_password or not confirm_password or not role:
        return jsonify({'message': 'All fields are required!'}), 400

    if new_password != confirm_password:
        return jsonify({'message': 'New password and confirm password do not match!'}), 400

    user_collection = None
    if role == 'admin':
        user_collection = mongo.db.admin
    elif role == 'user':
        user_collection = mongo.db.users
    else:
        return jsonify({'message': 'Invalid role!'}), 400

    user = user_collection.find_one({'email': email})

    if not user or not check_password_hash(user['password_hashed'], current_password):
        return jsonify({'message': 'Invalid email or current password!'}), 401

    hashed_new_password = generate_password_hash(new_password, method='pbkdf2:sha256')

    user_collection.update_one(
        {'_id': user['_id']},
        {'$set': {'password_hashed': hashed_new_password, 'plain_password': new_password}}
    )

    return jsonify({'message': 'Password changed successfully!'}), 200

@app.route('/approve_user', methods=['POST'])
@token_required
def approve_user():
    data = request.get_json()

    email = data.get('email')
    action = data.get('action')  # 'approve' or 'pending'

    admin_token = request.headers.get('x-access-token')
    admin_data = jwt.decode(admin_token, app.config['SECRET_KEY'], algorithms=['HS256'])
    admin_email = admin_data['identifier']
    admin =  mongo.db.admin.find_one({'email': admin_email})

    if not admin:
        return jsonify({'message': 'Admin privileges required!'}), 403

    user = mongo.db.users.find_one({'email': email})

    if not user:
        return jsonify({'message': 'User not found!'}), 404

    if action not in ['approve', 'pending']:
        return jsonify({'message': 'Invalid action!'}), 400

    new_status = 'approved' if action == 'approve' else 'pending'
    mongo.db.users.update_one({'_id': user['_id']}, {'$set': {'status': new_status}})
    
    return jsonify({'message': f'User status updated to {new_status} successfully!'}), 200


@app.route('/pending_users', methods=['GET'])
@token_required
def pending_users():
    admin_token = request.headers.get('x-access-token')
    admin_data = jwt.decode(admin_token, app.config['SECRET_KEY'], algorithms=['HS256'])
    admin_email = admin_data['identifier']
    admin = mongo.db.admin.find_one({'email': admin_email})

    if not admin:
        return jsonify({'message': 'Admin privileges required!'}), 403

    pending_users = mongo.db.users.find({'status': 'pending'}, {'_id': 0, 'firstname': 1, 'lastname': 1, 'email': 1, 'contact_number': 1, 'status': 1})

    user_list = list(pending_users)

    return jsonify({'pending_users': user_list}), 200



def extract_text_from_file(file_path):
    file_extension = os.path.splitext(file_path)[1].lower()
    if file_extension == '.pdf':
        return extract_text_from_pdf(file_path)
    elif file_extension in ('.jpg', '.jpeg', '.png'):
        return extract_text_from_image(file_path)
    elif file_extension in ('.txt', '.doc', '.docx'):
        return extract_text_from_doc(file_path)
    elif file_extension == '.xlsx':
        return extract_text_from_xlsx(file_path)
    else:
        return ""  # Unsupported file format

def extract_text_from_pdf(pdf_path):
    text = ""
    with fitz.open(pdf_path) as doc:
        for page in doc:
            text += page.get_text()
            image_list = page.get_images(full=True)
            for img_index, img in enumerate(image_list):
                xref = img[0]
                base_image = doc.extract_image(xref)
                image_bytes = base_image["image"]
                image = Image.open(io.BytesIO(image_bytes))
                text += pytesseract.image_to_string(image)
    return text

def extract_text_from_image(image_path):
    try:
        image = Image.open(image_path)
        extracted_text = pytesseract.image_to_string(image)
        if not extracted_text.strip():
            return "No text found in the uploaded image"
        return extracted_text
    except Exception as e:
        return f"Error processing image: {str(e)}"

def extract_text_from_doc(doc_path):
    file_extension = os.path.splitext(doc_path)[1].lower()
    if file_extension == '.txt':
        with open(doc_path, 'r', encoding='utf-8') as file:
            return file.read()
    else:
        return textract.process(doc_path).decode('utf-8')

def extract_text_from_xlsx(xlsx_path):
    dfs = pd.read_excel(xlsx_path, sheet_name=None)
    text = ""
    for sheet_name, df in dfs.items():
        text += df.to_string(index=False)
    return text

@app.route('/upload', methods=['POST'])
def upload_file():
    try:
        email = request.form.get('email')
        if not email:
            return jsonify({"error": "Email is required"}), 400

        user = mongo.db.users.find_one({"email": email})
        admin = mongo.db.admin.find_one({"email": email})

        if not user and not admin:
            return jsonify({"error": "User not found"}), 404

        if 'files' not in request.files:
            return jsonify({"error": "No file part in the request"}), 400

        files = request.files.getlist('files')
        if not files:
            return jsonify({"error": "No files selected for uploading"}), 400

        responses = []

        for file in files:
            file_id = str(uuid.uuid4())
            file_name = file.filename
            file_path = f"{file_id}{os.path.splitext(file.filename)[1]}"
            file.save(file_path)

            text = extract_text_from_file(file_path)
            os.remove(file_path)

            if not text:
                return jsonify({"error": "Unsupported file format or empty file"}), 400

            text = text.strip()
            text = re.sub(r'\s+', ' ', text)  # Replace multiple whitespace/newlines with a single space

            # Presidio Analysis
            results = analyzer.analyze(text=text, entities=predefined_entities, language='en', score_threshold=0.35)

            # Add custom SSN recognizer
            ssn_pattern = Pattern(name="ssn_pattern", regex=r'\b\d\s*(\d\s*){2}-?\s*(\d\s*){2}-?\s*(\d\s*){4}\b', score=1)
            ssn_recognizer = PatternRecognizer(supported_entity="SSN", patterns=[ssn_pattern])
            ssn_results = ssn_recognizer.analyze(text=text, entities=["SSN"])
            results.extend(ssn_results)

            # Add custom credit card recognizer
            credit_card_pattern = Pattern(name="credit_card_pattern", regex=r'\b(?:\d[\s\n-]*){13,19}\b', score=1)
            credit_card_recognizer = PatternRecognizer(supported_entity="CREDIT_CARD", patterns=[credit_card_pattern])
            credit_card_results = credit_card_recognizer.analyze(text=text, entities=["CREDIT_CARD"])
            results.extend(credit_card_results)

            # Add custom phone number recognizer
            phone_number_pattern = Pattern(name="phone_number_pattern", regex=r'(\+1[-.\s\n]*)?(\b\d{3}[-.\s\n]*\d{3}[-.\s\n]*\d{4}\b|\(\d{3}\)[-\s\n]*\d{3}[-.\s\n]*\d{4})', score=0.9)
            phone_number_recognizer = PatternRecognizer(supported_entity="Phone_Number", patterns=[phone_number_pattern])
            phone_number_results = phone_number_recognizer.analyze(text=text, entities=["Phone_Number"])
            results.extend(phone_number_results)

            # Add custom street recognizer 
            street_pattern = Pattern(name="street_pattern", regex=r"\b\d{1,3}\s+[A-Za-z]+\s+(?:Street|St\.|Avenue|Ave\.|Boulevard|Blvd\.|Road|Rd\.|Drive|Dr\.|Place|Pl\.|Lane|Ln\.|Court|Ct\.|Circle|Cir\.|Way|Parkway|Pkwy\.|Terrace|Ter\.|Trail|Trl\.|Commons|Loop|Crescent|Cres\.|Square|Sq\.|Grove|Grv\.|View|Vw\.|Walk|Wk\.|Path|Pth\.|Row|Rw\.|Valley|Vly\.|Heights|Hts\.|Plaza|Plz\.|Point|Pt\.|Meadow|Mdw\.|Field|Fld\.|Hill|Hl\.|Landing|Lndg\.|Quarters|Qtrs\.|Branch|Br\.|Drive|Drv\.?)\b", score=0.9)
            street_recognizer = PatternRecognizer(supported_entity="Address", patterns=[street_pattern])
            street_results = street_recognizer.analyze(text=text, entities=["Address"])
            results.extend(street_results)

            spacy_doc = spacy_model(text)

            # Convert SpaCy results to Presidio format
            spacy_entities = []
            for ent in spacy_doc.ents:
                score = float(ent.kb_id_) if ent.kb_id_ else 1.0
                if score < 0.6:
                    continue
                if ent.label_ in ["PRODUCT", "CARDINAL", "SSN", "CREDIT_CARD", "PHONE_NUMBER","QUANTITY","FAC","WORK_OF_ART","DATE_TIME","WORK_OF_ART","EVENT"]:
                    continue
                if ent.label_ == "GPE" and ent.text == "Mobile":
                    continue
                if ent.label_ in "PERSON" and ent.text == "Page" or ent.text == "VRML" or ent.text == "VRML Mission Statement":
                    continue
                if ent.label_ == "ORG" and (ent.text.isdigit() or ent.text == "Mobile" or ent.text == "House" or ent.text == "SSN" or re.fullmatch(r'\W+', ent.text) or re.fullmatch(r'[\d\W]+', ent.text)):
                    continue
                if ent.label_ == "LOC" and ent.text.strip().lower() == "street":
                    continue
                if ent.label_ == "DATE" and not any(char.isdigit() for char in ent.text):
                    continue
                spacy_entities.append(RecognizerResult(
                    entity_type=ent.label_,
                    start=ent.start_char,
                    end=ent.end_char,
                    score=score
                ))

            combined_entities = spacy_entities + credit_card_results + phone_number_results + ssn_results + street_results

            unique_entities = []
            seen_entities = set()
            for entity in combined_entities:
                entity_signature = (entity.start, entity.end, text[entity.start:entity.end])
                if entity_signature not in seen_entities:
                    seen_entities.add(entity_signature)
                    unique_entities.append(entity)

            # Create operator configs for each entity type
            operator_configs = {result.entity_type: OperatorConfig("replace", {"new_value": f"<{result.entity_type}>"}) for result in unique_entities}

            # Anonymize the text by replacing entities with their types
            anonymized_text = anonymizer.anonymize(
                text=text,
                analyzer_results=unique_entities,
                operators=operator_configs
            ).text
            
            findings = [
                {
                    "id": str(uuid.uuid4()),
                    "file_id": file_id,  # Added file_id to each finding
                    "entity_type": res.entity_type,
                    "start": res.start,
                    "end": res.end,
                    "score": res.score,
                    "text": text[res.start:res.end],
                    "highlight": True
                } 
                for res in unique_entities
            ]
    
            document_data = {
                "file_id": file_id,
                "file_name": file_name,
                "original_text": text,
                "anonymized_text": anonymized_text,
                "entities": findings,
                "created_at": datetime.utcnow()
            }

            # Update the respective document based on the user type
            if user:
                mongo.db.users.update_one(
                    {"email": email},
                    {"$push": {"documents": document_data}}
                )
            elif admin:
                mongo.db.admin.update_one(
                    {"email": email},
                    {"$push": {"documents": document_data}}
                )

            response = {
                "file_id": file_id,
                "file_name": file_name,
                "original_text": text,
                "anonymized_text": anonymized_text,
                "findings": findings
            }

            responses.append(response)

        return jsonify(responses), 200

    except Exception as e:
        logger.error(f"Error processing the file: {str(e)}")
        return jsonify({"error": str(e)}), 500

  
    
@app.route('/admin/edit_user', methods=['POST'])
@token_required
def admin_edit_user():
    data = request.get_json()

    # Extract admin token from headers
    admin_token = request.headers.get('x-access-token')
    admin_data = jwt.decode(admin_token, app.config['SECRET_KEY'], algorithms=['HS256'])
    admin_email = admin_data['identifier']
    admin = mongo.db.admin.find_one({'email': admin_email})

    if not admin:
        return jsonify({'message': 'Admin privileges required!'}), 403

    email = data.get('email')
    role = data.get('role')
    first_name = data.get('firstname')
    last_name = data.get('lastname')
    address = data.get('address')
    contact_number = data.get('contact_number')
    status = data.get('status')  # New status field

    if not email or not role:
        return jsonify({'message': 'Email and role are required!'}), 400

    user_collection = None
    if role == 'admin':
        user_collection = mongo.db.admin
    elif role == 'user':
        user_collection = mongo.db.users
    else:
        return jsonify({'message': 'Invalid role!'}), 400

    existing_user = user_collection.find_one({'email': email})

    if existing_user:
        # Update existing user information
        update_data = {}
        if first_name:
            update_data['firstname'] = first_name
        if last_name:
            update_data['lastname'] = last_name
        if address:
            update_data['address'] = address
        if contact_number:
            update_data['contact_number'] = contact_number

        user_collection.update_one(
            {'_id': existing_user['_id']},
            {'$set': update_data}
        )
        return jsonify({'message': 'User profile updated successfully!'}), 200
    else:
        
        return jsonify({'message': 'User not found!'}), 404
    
@app.route('/approved_users_list', methods=['GET'])
@token_required
def approved_users():
    
    admin_token = request.headers.get('x-access-token')
    admin_data = jwt.decode(admin_token, app.config['SECRET_KEY'], algorithms=['HS256'])
    admin_email = admin_data['identifier']
    admin = mongo.db.admin.find_one({'email': admin_email})

    if not admin:
        return jsonify({'message': 'Admin privileges required!'}), 403

    approved_users = mongo.db.users.find({'status': 'approved'}, {'_id': 0, 'firstname': 1, 'lastname': 1, 'email': 1, 'status': 1})

    user_list = list(approved_users)

    return jsonify({'approved_users': user_list}), 200


@app.route('/user_documents', methods=['POST'])
@token_required  
def user_documents():
    data = request.get_json()

    user_email = data.get('email')

    if not user_email:
        return jsonify({'message': 'Email is required!'}), 400

    # Check if the email exists in the users collection
    user = mongo.db.users.find_one({'email': user_email})
    
    if user:
        # Extract required details from the documents array
        documents = user.get('documents', [])
        document_details = [
            {
                'file_name': doc['file_name'],
                'anonymized_text': doc.get('anonymized_text', ''),
                'created_at': doc.get('created_at', '')
            }
            for doc in documents
            if 'file_name' in doc
        ]
        return jsonify({'documents': document_details}), 200
    
    # Check if the email exists in the admin collection
    admin = mongo.db.admin.find_one({'email': user_email})
    
    if admin:
        # Extract required details from the documents array
        documents = admin.get('documents', [])
        document_details = [
            {
                'file_name': doc['file_name'],
                'anonymized_text': doc.get('anonymized_text', ''),
                'created_at': doc.get('created_at', '')
            }
            for doc in documents
            if 'file_name' in doc
        ]
        return jsonify({'documents': document_details}), 200

    return jsonify({'message': 'Email not found in users or admin collections!'}), 404


if __name__ == '__main__':
    app.run(debug=True)
    
