from flask import render_template
from flask import Flask 
from flask import flash
from flask import send_file
from flask import request
from flask import session
from flask import redirect
from flask import jsonify
from flask import make_response
#from flask_jwt import JWT, jwt_required, current_identity
from werkzeug.security import safe_str_cmp
#import jwt
from flask_session import Session
from werkzeug.utils import secure_filename
from functools import wraps
from flask_cors import CORS, cross_origin

import yaml
import redis

import utils
import os
import uuid
import rq
import json
import datetime

import firebase_admin
from firebase_admin import firestore
from firebase_admin import auth
import pyrebase
import nibabel as nib
import pydicom
import io
import cv2
# import flask_socketio


import auto_script

import firebase_admin
import pyrebase
import json


app = Flask(__name__)
cors = CORS(app)



def maybe_mkdir(path):
    if not os.path.exists(path):
        os.mkdir(path)

config = yaml.load(open('config.yaml'), Loader=yaml.Loader)

all_models = config['models']
labels_colors = config['labels_colors']
labels = config['labels']
ignore = config['ignore']

niftis_folder = config['NIFTIS_FOLDER']
outputs_folder = config['OUTPUTS_FOLDER']
uploads_folder = config['UPLOADS_FOLDER']
results_folder = config['RESULTS_FOLDER']
pngs_folder = config['PNGS_FOLDER']
objs_folder = config['OBJS_FOLDER']
reports_folder = config['REPORTS_FOLDER']

maybe_mkdir(f"{uploads_folder}/nii")
maybe_mkdir(f"{uploads_folder}/dcm")
maybe_mkdir(f"{uploads_folder}/labels")

for model in all_models:
    maybe_mkdir(f"{outputs_folder}/{model}")
    maybe_mkdir(f"{objs_folder}/{model}")
    maybe_mkdir(f"{uploads_folder}/labels/{model}")

default_ctss_map = {
    '0': [0, 0.1],
    '1': [0.1, 5],
    '2': [5, 25],
    '3': [25, 50],
    '4': [50, 75],
    '5': [75, 100],
}


maybe_mkdir(f"{niftis_folder}/images")

firebase_cred = firebase_admin.credentials.Certificate('chestomx-firebase.json')
firebase_admin.initialize_app(firebase_cred)
pb = pyrebase.initialize_app(json.load(open('app_firebase.json')))
db = firestore.client()
col = db.collection('predictionRecords')

username = config['app_user']
password = config['app_pass']
print(config)

r = redis.Redis(host=config['redis_host'], port=6379)
r.set(username, password)

jobs_queue = rq.Queue(connection=r)


app.config["SESSION_PERMANENT"] = False
app.config["SESSION_TYPE"] = "redis"
app.config['SESSION_REDIS'] = r
Session(app)

## get token
@app.route('/api/token', methods=['POST'])
@cross_origin()
def get_token():
    print('hello')
    if request.method == 'POST':
        username = request.json['username']
        password = request.json['password']
        print(username, password)
        try:
            token = pb.auth().sign_in_with_email_and_password(username, password)
            #user_id = db.collection('users').where('email', '==', username).get()[0].to_dict()['user_id']
            user_id = db.collection('users').where('email', '==',
                    username).get()[0].to_dict()['uid']
            return jsonify({'token': token['idToken'], 'user_id': user_id})
        except Exception as e:
            print(e)
            return jsonify({'message': 'Invalid credentials'}), 401

# create a new user using the form
@app.route('/register', methods=['POST'])
@cross_origin()
def register():
    if request.method == 'POST':
        username = request.json['username']
        password = request.json['password']
        email = request.json['email']
        try:
            user = pb.auth().create_user_with_email_and_password(email, password)
            return jsonify({'message': 'User created successfully', 'user_id': user['localId']}), 200
        except Exception as e:
            # catch error message and from the error
            error = json.loads(e.args[1])
            return jsonify({'message': error['error']['message']}), 400


# this for activating user by saving their details
@app.route('/activate_user', methods=['POST'])
@cross_origin()
def activate_user():
    if request.method == 'POST':
        try:
            user_id = request.json['user_id']
            username = request.json['username']
            email = request.json['email']
            phone = request.json['phone']
            country = request.json['country']
            name = request.json['name']
            profession = request.json['profession']
            institution = request.json['institution']
            # check if any is none
            if not user_id or not username or not email or not phone or not country or not name:
                return jsonify({'message': 'Missing fields'}), 400
            # check if user is in db
            user_doc = db.collection('users').document(user_id).get()
            if user_doc.exists:
                return jsonify({'message': 'User already exists'}), 409
            db.collection('users').document(user_id).set({
                'username': username,
                'email': email,
                'phone': phone,
                'country': country,
                'name': name,
                'user_id': user_id,
                'profession': profession,
                'institution': institution,
                'ctss': default_ctss_map
            })
            return jsonify({'message': 'User created successfully'}), 200
        except Exception as e:
            # catch error message and from the error
            error = json.loads(e.args[1])
            return jsonify({'message': error['error']['message']}), 400


            
            
# get image
#@app.route('/api/image/<img_id>/<slice_id>', methods=['GET'])
#def get_image(img_id):
#    img_path = f"pngs/{img_id}/{slice_id}.png"
#    if os.path.exists(img_path):
#        return send_file(img_path, mimetype='image/png')
#    else:
#        return jsonify({'message': 'Image not found'}), 404






# get image
@app.route('/api/image/<img_id>/<slice_id>', methods=['GET'])
def get_image(img_id):
    img_path = f"pngs/{img_id}/{slice_id}.png"
    if os.path.exists(img_path):
        response = make_response(send_file(img_path, mimetype='image/png'))
        response.headers['Access-Control-Allow-Private-Network'] = true
        response.headers['Access-Control-Request-Private-Network'] = true
        return response

    else:
        return jsonify({'message': 'Image not found'}), 404












# this is a middleware to check if the user is logged in and activated
def api_login_required(f):
    @wraps(f)
    def wrap(*args, **kwargs):
        if not request.headers.get('authorization'):
            return {'message': 'No token provided'},400
        try:
            print('token: ', request.headers['authorization'])
            user = auth.verify_id_token(request.headers['authorization'].replace('Bearer ', ''))
            # check if user is in db
            user_id = user['uid']
            print('user_id: ', user_id)
            user_doc = db.collection('users').document(user_id).get()
            if not user_doc.exists:
                return {'message': 'User not found'}, 404
            request.user = user
        except Exception as e:
            print("error: ", str(e))
            if e.args[0].startswith('Invalid token'):
                return {'message': 'Invalid token provided.'}, 400
            elif e.args[0].startswith('Token expired'):
                return {'message': 'Token expired.'}, 400
        return f(*args, **kwargs)
    return wrap

# check if user is activated
@app.route('/api/is_activated')
@api_login_required
def is_activated():
    # try:
    #     user_id = request.user['uid']
    #     user_doc = db.collection('users').document(user_id).get()
    #     if user_doc.exists:
    #         return jsonify({'message': 'User activated'})
    #     else:
    #         return jsonify({'message': 'User not activated'}), 404
    # except Exception as e:
    #     print(e)            
    #     return jsonify({'message': 'Error: ' + e.message}), 500
    return jsonify({'message': 'User activated'})


@app.route('/api/ctss_map', methods=['GET', 'POST'])
@cross_origin()
@api_login_required
def ctss_map():
    if request.method == 'POST':
        try:
            user_id = request.user['uid']
            user_doc = db.collection('users').document(user_id).get()
            if not user_doc.exists:
                return jsonify({'message': 'User does not exist'}), 404
            ctss = request.json['ctss']
            db.collection('users').document(user_id).update({
                'ctss': ctss
            })
            return jsonify({'message': 'CTSS updated successfully'}), 200  
        except Exception as e:
            # catch error message and from the error
            error = json.loads(e.args[1])
            return jsonify({'message': error['error']['message']}), 400
    elif request.method == 'GET':
        try:
            user_id = request.user['uid']
            user_doc = db.collection('users').document(user_id).get()
            if not user_doc.exists:
                return jsonify({'message': 'User does not exist'}), 404
            return jsonify({'ctss': user_doc.to_dict()['ctss']})
        except Exception as e:
            # catch error message and from the error
            error = json.loads(e.args[1])
            return jsonify({'message': error['error']['message']}), 400


# get the models available
@app.route('/api/models')
@cross_origin()
def get_models():
    query_dict = {}
    patient_id = request.args.get('patient_id', None)
    unique_id = request.args.get('unique_id', None)
    access_key = request.args.get('access_key', None)
    if not patient_id or not unique_id or not access_key:
        print("ALL MODELS RESQUEST")
        return jsonify(all_models)

    if not r.get(unique_id):
        return jsonify({'status': 'No such patient_id or accession_number'}), 400
    
    if r.get(unique_id).decode() != access_key:
        return jsonify({'status': 'Access key is incorrect'}), 400

    record = col.where('uniqueId', '==', unique_id).get()[0].to_dict()
#    print(record, 'record')
    models = record['models'].split(',')

    return jsonify({'models': models, 'ignore': ignore})

# get obj and mtl files (used in 3d rendering)
@app.route('/get_obj_mtl')
@cross_origin()
def get_obj():
    query_dict = request.args.to_dict()
    patient_id = query_dict.get('patient_id', '')
    unique_id = query_dict.get('unique_id', '')
    access_key = query_dict.get('access_key', '')
    model = query_dict.get('model', '')
    file_type = query_dict.get('file_type', 'obj')

    if not r.get(unique_id):
        return jsonify({'status': 'No such patient_id or accession_number'}), 400
    
    if r.get(unique_id).decode() != access_key:
        return jsonify({'status': 'Access key is incorrect'}), 400
    record = col.where('uniqueId', '==', unique_id).get()[0].to_dict()
    #print(record, 'record')
    models = record['models'].split(',')
    if model not in models:
        model = models[0]
    file_type = file_type.split('/')[0]
    filepath = f"{objs_folder}/{model}/{patient_id}_{unique_id}.{file_type}"

    print(filepath, objs_folder, model, patient_id, unique_id, file_type)

    return send_file(filepath, cache_timeout=6000)

@app.route('/get', methods=['GET'])
def get():
    if 'patient_id' in request.args and 'accession_number' in request.args:
        patient_id = request.args['patient_id']
        accession_number = request.args['accession_number']

        print(patient_id, accession_number)

        key = f'{patient_id}_{accession_number}'

        if not r.get(key):
            return jsonify({'status': 'not_created'})
        else:     
            download_id = r.get(key).decode()
            link = f'/download?patient_id={patient_id}&accession_number={accession_number}&download_id={download_id}'
            return jsonify({
                'status': 'created',
                'link': link
                })    
    return jsonify({'status': 'invalid request (please pass (patient_id and accession_number)'})


def validate_datetime(s):
    try:
        _ = datetime.datetime.strptime(s, '%Y-%m-%d %H:%M')
        return True
    except Exception as e:
        print('Date not in proper format: ', str(e))
        return False

@app.route('/api/user', methods=['GET', 'POST'])
@api_login_required
def user_data():
    if request.method == 'POST':
        try:
            user_id = request.user['uid']
            user_doc = db.collection('users').document(user_id).get()
            if not user_doc.exists:
                return jsonify({'message': 'User not found'}), 404
            print('request json', request.json)
            #data = request.json['data']
            data = request.json

            del data['user_id']
            del data['username']
            # mongo_db['users'].update_one({'user_id': user_id}, {'$set': data})
            db.collection('users').document(user_id).update(data)
            
            return jsonify({'message': 'Profile updated successfully'}), 200
        except Exception as e:
            return jsonify({'message': str(e)}), 400
    else:
        try:
            user_id = request.user['uid']
            # user_doc = mongo_db['users'].find_one({'user_id': user_id}, {'_id': False, 'ctss': False})
            user_doc = db.collection('users').document(user_id).get()
            # if user_doc is None:
            if not user_doc.exists:
                return jsonify({'message': 'User not found'}), 404
            # return jsonify(user_doc)
            return jsonify(user_doc.to_dict())
        except Exception as e:
            return jsonify({'message': str(e)}), 400
         

# get all records (predictions) of the particular user
@app.route('/api/records', methods=['GET'])
@cross_origin()
@api_login_required
def get_records():
    
    # get results by receivedAt time
    results = col.where('user_id', '==', request.user['uid']).order_by('receivedAt', direction=firestore.Query.DESCENDING).limit(100).get()
    # results = col.order_by('receivedAt', direction=firestore.Query.DESCENDING).limit(100).get()

    def format_result(result):
        new_result = {}
        for key, val in result.items():
            if val is not None and isinstance(val, datetime.datetime):
                new_result[key] = val.strftime('%Y-%m-%d %H:%M')
            else:
                new_result[key] = str(val)
        if r.get(result['uniqueId']):
            new_result['accessId'] = r.get(result['uniqueId']).decode()
	#elif 'resultKey' in result:
	#elif result['accessKey']
        elif 'accessKey' in result:
            new_result['accessId'] = result['accessKey']
            r.set(result['uniqueId'], result['accessKey'])
        return new_result

    results = [ format_result(x.to_dict()) for x in results]
    return jsonify(results)

@app.route('/api/delete_record', methods=['POST'])
@cross_origin()
@api_login_required
def delete_record():
    try:
        unique_id = request.json.get('unique_id', None)
        patient_id = request.json.get('patient_id', None)
        if not unique_id  or not patient_id:
            return jsonify({'message': 'Please provide the unique_id and patient_id of the record to delete'}), 400
        # record = col.find_one({'uniqueId': unique_id})
        record = col.where('uniqueId', '==', unique_id).get()[0]
        if not record:
            return jsonify({'message': 'The record does not exist!'}), 400
        # col.delete_one({'uniqueId': unique_id})
        col.document(unique_id).delete()
        # delete output of the case
        os.system(f"rm -r '{pngs_folder}/{patient_id}_{unique_id}'")
        os.system(f"rm -r '{niftis_folder}/images/{patient_id}_{unique_id}*'")
        os.system(f"rm -r '{reports_folder}/{patient_id}_{unique_id}*'")
        os.system(f"rm -r '{uploads_folder}/dcm/{patient_id}_{unique_id}*'")
        os.system(f"rm -r '{uploads_folder}/nii/{patient_id}_{unique_id}*'")

        for model in all_models:
            os.system(f"rm -r '{outputs_folder}/{model}/{patient_id}_{unique_id}*'")
            os.system(f"rm -r '{objs_folder}/{model}/{patient_id}_{unique_id}*'")
            os.system(f"rm -r '{uploads_folder}/label/{model}/{patient_id}_{unique_id}*'")
        return jsonify({'message': 'Record deleted sucessfully!'}), 200
    except Exception as e:
        print("Exception in deleting: ", e)
        return jsonify({"message": "Error in deleting: " + str(e)}), 500

@app.route('/api/edit_record', methods=['POST'])
@cross_origin()
@api_login_required
def edit_record():
    try:
        unique_id = request.json.get('unique_id', None)
        patient_id = request.json.get('patient_id', None)
        if not unique_id  or not patient_id:
            return jsonify({'message': 'Please provide the unique_id and patient_id of the record to delete'}), 400
        # record = col.find_one({'uniqueId': unique_id})
        record = col.where('uniqueId', '==', unique_id).get()[0]
        if not record:
            return jsonify({'message': 'The record does not exist!'}), 400

        patient_name = request.json.get('patient_name', '')
        patient_birth_date = request.json.get('patient_dob', '')
        patient_sex = request.json.get('patient_sex', '')
        institution_name = request.json.get('institution_name', '')
        series_desc = request.json.get('series_desc', '')
        
        col.document(unique_id).update({
            'patientId': patient_id,
            'patientName': patient_name,
            'patientBirthDate': patient_birth_date,
            'patientSex': patient_sex,
            'institutionName': institution_name,
            'seriesDesc': series_desc,
        })
       
        return jsonify({'message': 'Record updated sucessfully!'}), 200
    except Exception as e:
        print("Exception in editing: ", e)
        return jsonify({"message": "Error in editing: " + str(e)}), 500

@app.route('/api/report/<patient_id>/<unique_id>/<access_key>', methods=['GET'])
def get_report(patient_id, unique_id, access_key):
    report_path = f"{reports_folder}/{patient_id}_{unique_id}.html"
    if not r.get(unique_id):
        return jsonify({'status': 'No such patient_id or accession_number'}), 400
    if r.get(unique_id).decode() != access_key:
        return jsonify({'status': 'Access key is incorrect'}), 400
    if not os.path.exists(report_path):
        return jsonify({'message': 'No report available'}), 404
    return send_file(report_path)

@app.route('/api/report_download/<patient_id>/<unique_id>/<access_key>', methods=['GET'])
def download_report(patient_id, unique_id, access_key):
    report_path = f"{reports_folder}/{patient_id}_{unique_id}.html"
    if not r.get(unique_id):
        return jsonify({'status': 'No such patient_id or accession_number'}), 400
    if r.get(unique_id).decode() != access_key:
        return jsonify({'status': 'Access key is incorrect'}), 400
    if not os.path.exists(report_path):
        return jsonify({'message': 'No report available'}), 404

    response = send_file(report_path, as_attachment=True, attachment_filename=f"{patient_id}_{unique_id}.html")
    response.headers['Content-Encoding'] = ''
    response.headers['Content-Type'] = 'text/xml'
    return response

@app.route('/api/report_pdf_download/<patient_id>/<unique_id>/<access_key>', methods=['GET'])
def download_report_pdf(patient_id, unique_id, access_key):
    report_path = f"{reports_folder}/{patient_id}_{unique_id}.pdf"
    if not r.get(unique_id):
        return jsonify({'status': 'No such patient_id or accession_number'}), 400
    if r.get(unique_id).decode() != access_key:
        return jsonify({'status': 'Access key is incorrect'}), 400
    if not os.path.exists(report_path):
        return jsonify({'message': 'No report available'}), 404

    return send_file(report_path, as_attachment=True, attachment_filename=f"{patient_id}_{unique_id}.pdf")
    

# this returns all the pngs required to view the particular case
import functools 
@app.route('/api/view_model', methods=['GET'])
@cross_origin()
def view_model():
 
    query_dict = request.args.to_dict()
    patient_id = query_dict.get('patient_id', '')
    unique_id = query_dict.get('unique_id', '')
    access_key = query_dict.get('access_key', '')

    print(access_key, 'access_key')

    # if not r.get(unique_id):
    #     return jsonify({'status': 'No such patient_id or accession_number'}), 400
    
    # if r.get(unique_id).decode() != access_key:
    #     return jsonify({'status': 'Access key is incorrect'}), 400
    record = col.where('uniqueId', '==', unique_id).get()[0].to_dict()
    if not record:
        print('No such record')
        return jsonify({'status': 'No such patient_id or accession_number'}), 400
    
    # check if the access key is correct from the records
    if record['accessKey'] != access_key:
        print('Access key is incorrect')
        return jsonify({'status': 'Access key is incorrect'}), 400

    r.set(unique_id, access_key)

    record = col.where('uniqueId', '==', unique_id).get()[0].to_dict()
    #print(record, 'record')
    models = record['models'].split(',')
    ctss = record['ctss'] if 'ctss' in record else None
    images = [ f"png/{patient_id}/{unique_id}/{i.replace('.png', '')}/{access_key}" for i in os.listdir(f'./pngs/{patient_id}_{unique_id}/') if i.endswith('.png') ]
    labels_images = [ image for image in images if '_'.join(image.split('/')[-2].split('_')[1:]) in models ]
    images = list(set(images).difference(set(labels_images)))
    images = sorted(images, key=lambda x: int(x.split('/')[-2]))
    # images = images[:100]
    img_size = cv2.imread(f"./pngs/{patient_id}_{unique_id}/{images[0].split('/')[-2]}.png").shape[:2]
    image_data = {
        'patient_id': patient_id,
        'images': images,
        'labels_images': [ {'name': '_'.join(image.split('/')[-2].split('_')[1:]), 'url': image} for image in labels_images ],
        'ignore': ignore,
        'num_images': len(images),
        'img_size': img_size,
        'labels': { model: labels[model] for model in models },
        'labels_colors': { model: labels_colors[model] for model in models },
        'models': models,
        'ctss': ctss
    }

    return jsonify(image_data)

# this is for downloading the nii image of a label file
@app.route('/nii/<patient_id>/<unique_id>/<access_key>/<model>', methods=['GET'])
@cross_origin()
def get_nii(patient_id, unique_id, access_key, model):
    print('getting nii image for download!')
    filepath = f"{outputs_folder}/{model}/{patient_id}_{unique_id}.nii.gz"
    
    if not r.get(unique_id):
        return jsonify({'status': 'No such patient_id or accession_number'}), 400
    
    if r.get(unique_id).decode() != access_key:
        return jsonify({'status': 'Access key is incorrect'}), 400
    # set headers for gzip
    response = send_file(filepath, as_attachment=True, attachment_filename=f"{patient_id}_{unique_id}_{model}.nii.gz")
    response.headers['Content-Encoding'] = ''
    return response

# this is for downloading the dcm image of a label file
@app.route('/dcm/<patient_id>/<unique_id>/<access_key>/<model>', methods=['GET'])
@cross_origin()
def get_dcm(patient_id, unique_id, access_key, model):
    print('getting dcm image for download!')
    filepath = f"{outputs_folder}/{model}/{patient_id}_{unique_id}.zip"
    
    if not r.get(unique_id):
        return jsonify({'status': 'No such patient_id or accession_number'}), 400
    
    if r.get(unique_id).decode() != access_key:
        return jsonify({'status': 'Access key is incorrect'}), 400

    return send_file(filepath, cache_timeout=6000)

# get a png of a particular image slice
@app.route('/png/<patient_id>/<unique_id>/<slice_id>/<access_key>', methods=['GET'])
@cross_origin()
def get_lobes_png(patient_id, unique_id, slice_id, access_key):
    image_path = f'./pngs/{patient_id}_{unique_id}/{slice_id}.png'

    if not r.get(unique_id):
        return jsonify({'status': 'No such patient_id'}), 400
    
    if r.get(unique_id).decode() != access_key:
        return jsonify({'status': 'Access key is incorrect'}), 400

    if os.path.exists(image_path):
        return send_file(image_path, mimetype='image/png')
    else:
        return jsonify({'status': 'No such image'})

# get ctss score for a particular case from the database
@app.route('/api/prediction/ctss/<unique_id>/<access_key>', methods=['GET'])
def get_prediction(unique_id, access_key):
    key = f"{patient_id}_{accession_number}"
    if not r.get(key):
        return jsonify({'status': False, 'message': 'Patient does not exist'}), 400
    elif r.get(key).decode() != access_key:
        return jsonify({'status': False, 'message': 'Access key is incorrect'}), 401
    else:
        try:
            ctss_record = col.get(documnet).to_dict()['ctss']
            ctss_record = ctss_results_map(ctss_record)
            return jsonify(ctss_record), 200
        except Exception as e:
            print(e)
            return jsonify({'status': False, 'message': 'No CTSS scores found'}), 400

# for uploading files for new prediction, note, the prediction does not happen here
@app.route('/api/prediction/upload', methods=['POST'])
@cross_origin()
@api_login_required
def upload_prediction():
    print('uploading...')
    if request.method == 'POST':
        print(request.files)
        if 'file' not in request.files:
            return jsonify({'status': False, 'message': 'No file part in the request'}), 400
        fileType = request.form.get('fileType', '')
        print('fileType', fileType)
        if fileType not in ['nii', 'dcm']:
            return jsonify({'status': False, 'message': 'Invalid file type'}), 400
        if fileType == 'nii':
            file = request.files['file']
            if file.filename == '':
                return jsonify({'status': False, 'message': 'No file selected for uploading'}), 400
            filename = secure_filename(file.filename)
            patient_id = filename.split('.nii')[0]
            unique_id = uuid.uuid4().hex
            filename = f'{patient_id}_{unique_id}.nii.gz'
            file.save(os.path.join(f'{uploads_folder}/nii/', filename))
            return jsonify({'status': True, 'message': 'File successfully uploaded', 'id':{
                'patient_id': patient_id,
                'unique_id': unique_id
            }}), 200
        elif fileType == 'dcm':
            ## multiple dcm files received...
            files = request.files.getlist('file')
            if len(files) == 0:
                return jsonify({'status': False, 'message': 'No file selected for uploading'}), 400
            ## get patient id from first dcm file
            dicom_file = pydicom.dcmread(files[0].stream)
            if hasattr(dicom_file, 'PatientID'):
                patient_id = dicom_file.PatientID
            else:
                patient_id = uuid.uuid4().hex

            patient_id = patient_id.replace('/', '').replace('\\', '')

            files[0].stream.seek(0)
            unique_id = uuid.uuid4().hex
            folder = f"{uploads_folder}/dcm/{patient_id}_{unique_id}"
            os.mkdir(folder)
            for file in files:
                filename = secure_filename(file.filename)
                file.save(os.path.join(folder, filename))
            return jsonify({'status': True, 'message': 'File successfully uploaded', 'id': {
                'patient_id': patient_id,
                'unique_id': unique_id
            }}), 200
        else:
            return jsonify({'status': False, 'message': 'Invalid file type'}), 400

# this triggers the inference for a particular case
@app.route('/api/prediction/inference', methods=['POST'])
@cross_origin()
@api_login_required
def inference():
    if request.method == 'POST':

        print('hii', request.json)
        try:
            patient_id = request.json.get('patientId', '')
            unique_id = request.json.get('uniqueId', '')
            models = request.json.get('models', '')
            file_type = request.json.get('fileType', '')
            print('gello')
            print(patient_id, unique_id, models, file_type)

            if not patient_id or not unique_id or not models or not file_type:
                print('Invalid request')
                return jsonify({'status': False, 'message': 'Invalid request'}), 400
            print('inferencing..')
            # args = {
            #     'redis_host': config['redis_host'],
            #     'patient_id': patient_id,
            #     'unique_id': unique_id,
            #     'models': models,
            #     'file_type': file_type,
            #     'models': [ model.lower() for model in models.split(',')],
            #     'device': config['device'],
            #     'lungs_fold': config['lungs_fold'],
            #     'lobes_fold': config['lobes_fold'],
            #     'infiltration_fold': config['infiltration_fold'],
            #     'ild_fold': config['ild_fold'],
            #     'lungs_task': config['lungs_task'],
            #     'lobes_task': config['lobes_task'],
            #     'infiltration_task': config['infiltration_task'],
            #     'ild_task': config['ild_task'],
            #     'RESULTS_FOLDER': config['RESULTS_FOLDER'],
            #     'niftis_folder': niftis_folder,
            #     'outputs_folder': outputs_folder,
            #     'uploads_folder': uploads_folder,
            #     'pngs_folder': pngs_folder,
            #     'objs_folder': objs_folder,
            #     'labels': labels,
            #     'user_id': request.user['uid'],
            # }
            args = (
                config,
                patient_id,
                unique_id,
                [ model.lower() for model in models.split(',')],
                file_type,
                request.user['uid'],
                # None
            )

            if file_type == 'dcm':

                folder = f"{uploads_folder}/dcm/{patient_id}_{unique_id}"
                dicom_file = pydicom.dcmread(f"{folder}/{os.listdir(folder)[0]}") 

                patient_name = str(dicom_file.PatientName) if hasattr(dicom_file, 'PatientName') else 'NA'
                patient_birth_date = utils.format_date(dicom_file.PatientBirthDate) if hasattr(dicom_file, 'PatientBirthDate') else 'NA'
                patient_sex = str(dicom_file.PatientSex) if hasattr(dicom_file, 'PatientSex') else 'NA'
                institution_name = str(dicom_file.InstitutionName) if hasattr(dicom_file, 'InstitutionName') else 'NA'
                series_desc = str(dicom_file.SeriesDescription) if hasattr(dicom_file, 'SeriesDescription') else 'NA'
            else:
                patient_name = request.json.get('patientName', 'NA')
                patient_birth_date = request.json.get('patientBirthDate', 'NA')
                patient_sex = request.json.get('patientSex', 'NA')
                institution_name = request.json.get('institutionName', 'NA')
                series_desc = request.json.get('seriesDesc', 'NA')

            col.document(unique_id).set({
                'predictionStatus': 'Inference queued',
                'patientId': patient_id,
                'uniqueId': unique_id,
                'models': models,
                'receivedAt': datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                'predictedAt': '',
                'patientName': patient_name,
                'patientBirthDate': patient_birth_date,
                'patientSex': patient_sex,
                'institutionName': institution_name,
                'seriesDesc': series_desc,
                'user_id': request.user['uid']
            })

            # inference is not called here but pushed to redis queue
            jobs_queue.enqueue(auto_script.do_inference, args=args, job_timeout=7200)
            print('kept for inference')
            return jsonify({'status': True, 'message': 'Inference started'}), 200
        except Exception as e:
            print(e)
            return jsonify({'status': False, 'message': 'Error starting inference'}), 400
        
# calculate metrics for a particular case
@app.route('/api/prediction/calc_metrics', methods=['POST'])
@cross_origin()
@api_login_required
def calc_metrics():
    if request.method == 'POST':
        try:
            print('calculting metrics!!')
            f = request.files['label_file']
            if f.filename == '':
                return jsonify({'status': False, 'message': 'No file selected for uploading'}), 400

            print(request.form)
            filename = secure_filename(f.filename)
            patient_id = request.form.get('patient_id', '')
            unique_id = request.form.get('unique_id', '')
            filename = f'{patient_id}_{unique_id}.nii.gz'

            model = request.form.get('model', 'lungs')
            if patient_id is None or unique_id is None or model is None:
                return jsonify({'status': False, 'message': 'Wrong request'}), 400
            
            label_file_path = os.path.join(f'{uploads_folder}/labels/', model, filename)
            f.save(label_file_path)

            model_labels = list(labels[model].keys())

            pred_file_path = f"{outputs_folder}/{model}/{patient_id}_{unique_id}.nii.gz"
            
            if not os.path.exists(pred_file_path):
                return jsonify({'status': False, 'message': 'Prediction for the model does not exist!'}), 400

          #  mail_to = request.user['email']
            mail_to = None
            args = (
                unique_id,
                model,
                pred_file_path,
                label_file_path,
                model_labels,
                mail_to,
                config
            )
            col.document(unique_id).update({
                'metrics.{}'.format(model): 'Calculating',
            })
            jobs_queue.enqueue(auto_script.calc_metrics, args=args, job_timeout=3600)
            return jsonify({'status': True, 'message': 'Calculating metrics!'}), 200
        except Exception as e:
            print("Error: ", e)
            return jsonify({'status': False, 'message': 'Error calculating metrics'}), 500


@app.route('/api/prediction/metrics', methods=['GET'])
@cross_origin()
def get_metrics():
    query_dict = request.args.to_dict()
    unique_id = query_dict.get('unique_id', None)
    model = query_dict.get('model', None)
    print(unique_id, model)
    if unique_id is None or model is None:
        return jsonify({'status': False, 'message': 'Wrong input, please send unique_id and model!'}), 200
    record = col.document(unique_id).get().to_dict()
    #print(record, 'record')
    if 'metrics' not in record or model not in record['metrics']:
        return jsonify({'status': False, 'message': 'Metrics for the model do not exist!'}), 200
    if record['metrics'][model] == 'Calculating':
        return jsonify({'status': False, 'message': 'Metrics are being calculated!'}), 200
    return jsonify({'metrics': record['metrics'][model], 'message': 'Metrics Calculated!'}), 200


if __name__ == "__main__":

    print('port: ', config['port'])
    app.run(port=4002, host='0.0.0.0', debug=True)
