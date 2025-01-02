from flask import Flask, render_template, request, url_for, redirect
import os
from werkzeug.utils import secure_filename
from PIL import Image, ImageDraw, ImageOps
from ultralytics import YOLO
import tensorflow as tf
import numpy as np
import cv2

app = Flask(__name__)


### folders
UPLOAD_FOLDER = 'static/upload'
PROCESSED_FOLDER = 'static/processed'
CROPPED_FOLDER = 'static/cropped'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['PROCESSED_FOLDER'] = PROCESSED_FOLDER
app.config['CROPPED_FOLDER'] = CROPPED_FOLDER

# files allowed
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'mp4', 'avi', 'mov'}
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

# face detect model
model = YOLO('./static/models/yolov8n-face.pt')

# deepfake detect model
model_predict_path = "./static/models/VGG19.keras"
model_predict = tf.keras.models.load_model(model_predict_path)

# fake or real
class_names = ['Fake', 'Real']
class_indices = {name: idx for idx, name in enumerate(class_names)}

# detect faces, return the boxes of the faces
def detect_faces(model, img_path):
    results = model(img_path)  # Perform inference
    boxes = results[0].boxes.xyxy  # Get bounding boxes
    return boxes

# if video, extract frames
def extract_frames_from_video(video_path, output_folder, num_frames):
    cap = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frame_interval = max(total_frames // num_frames, 1)  # calculate interval between frames
    extracted_frame_paths = []

    for i in range(num_frames):
        frame_position = i * frame_interval
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_position)  # move to the specific frame
        success, frame = cap.read()
        if success:
            frame_filename = f"frame_{i}.jpg"
            frame_path = os.path.join(output_folder, frame_filename)
            cv2.imwrite(frame_path, frame)
            extracted_frame_paths.append(frame_path)
        else:
            print(f"Failed to extract frame at position {frame_position}")
            break

    cap.release()
    return extracted_frame_paths

# model predict
def predict_image(image):
    file_path = os.path.normpath(os.path.join(app.config['CROPPED_FOLDER'], image))
    
    # process file
    img = Image.open(file_path)
    img = img.convert('RGB')
    img = img.resize((224, 224))
    img_array = np.array(img)
    img_array = np.expand_dims(img_array, axis=0)
    preprocess_input = tf.keras.applications.vgg19.preprocess_input
    img_array = preprocess_input(img_array)
    
    # predict
    predictions = model_predict.predict(img_array)
    
    # score
    confidence_score = predictions[0][0]
    threshold = 0.5
    predicted_class_index = 1 if confidence_score >= threshold else 0

    predicted_class_name = class_names[predicted_class_index]
    
    confidence = get_confidence_score(predictions[0][0])
    return predicted_class_name, confidence

# confidence score
def get_confidence_score(prob):
    if prob > 0.5:
        confidence = (prob - 0.5) * 200
    elif prob < 0.5:
        confidence = (0.5 - prob) * 200
    else:
        confidence = 0
    
    return round(confidence, 2)

# remove files in a directory
def clean_directory(directory):
    for filename in os.listdir(directory):
        file_path = os.path.join(directory, filename)
        try:
            if os.path.isfile(file_path):
                os.remove(file_path)
        except Exception as e:
            print(f"Error deleting file {file_path}: {e}")

# detect faces in an image/frame
def process_images(file_path, filename):
    print(filename)
    # prepare image and detect faces
    img = Image.open(file_path)
    img = ImageOps.exif_transpose(img)  # correct orientation
    boxes = detect_faces(model, file_path)

    # if no faces detected
    if not boxes.size(0):
        return False,False

    # draw bounding boxes and crop faces
    img_with_boxes = img.copy()
    draw = ImageDraw.Draw(img_with_boxes)
    cropped_images = []

    for idx, box in enumerate(boxes):
        x1, y1, x2, y2 = map(int, box[:4].tolist())  # convert to integers
        draw.rectangle([x1, y1, x2, y2], outline="#39FF14", width=10)
        
        # crop face and save
        cropped = img.crop((x1, y1, x2, y2))
        
        base_filename = f"cropped_{idx}_{filename}"
        cropped_filename = base_filename
        counter = 1
        cropped_path = os.path.join(app.config['CROPPED_FOLDER'], cropped_filename)
        
        while os.path.exists(cropped_path):
            cropped_filename = f"cropped_{idx}_{counter}_{filename}"
            cropped_path = os.path.join(app.config['CROPPED_FOLDER'], cropped_filename)
            counter += 1
        
        # save cropped image
        cropped.save(cropped_path)
        cropped_images.append(cropped_filename)
    
    return img_with_boxes, cropped_images

# predictions for all of the faces in an image/frame
def get_frame_prediction(predictions):
    fake_confidences = [pred[1] for pred in predictions if pred[0] == "Fake"]
    real_confidences = [pred[1] for pred in predictions if pred[0] == "Real"]

    # if any prediction is fake, picture/frame is fake
    if fake_confidences:
        avg_fake_confidence = sum(fake_confidences) / len(fake_confidences)
        return "Fake", round(avg_fake_confidence, 2)

    # if no fake, then real
    avg_real_confidence = sum(real_confidences) / len(real_confidences)
    return "Real", round(avg_real_confidence, 2)

# predictions for all frames in a video
def get_video_prediction(predictions):
    fake_count = sum(1 for pred in predictions if pred[0] == "Fake")
    real_count = sum(1 for pred in predictions if pred[0] == "Real")
    fake_confidence = sum(pred[1] for pred in predictions if pred[0] == "Fake")
    real_confidence = sum(pred[1] for pred in predictions if pred[0] == "Real")
    
    # majority voting
    if fake_count > real_count:
        final_prediction = "Fake"
        confidence = fake_confidence / fake_count if fake_count else 0
    else:
        final_prediction = "Real"
        confidence = real_confidence / real_count if real_count else 0

    return final_prediction, round(confidence, 2)


### routing
@app.route('/', methods=['GET','POST'])
def hello_world():
    clean_directory(app.config['UPLOAD_FOLDER'])
    clean_directory(app.config['PROCESSED_FOLDER'])
    clean_directory(app.config['CROPPED_FOLDER'])
    
    return render_template('index.html')

@app.route("/detect", methods=["POST"])
def detect():
    if request.method == 'POST':
        # handle file upload
        if 'image' not in request.files or not request.files['image']:
            return redirect(url_for('hello_world'))
        
        file = request.files['image']

        if allowed_file(file.filename):
            # save the uploaded file to uploaded folder
            filename = secure_filename(file.filename)
            file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(file_path)
            
            if filename.split('.')[-1].lower() in {'mp4', 'avi', 'mov'}:
                # Extract multiple evenly spaced frames
                frame_folder = app.config['UPLOAD_FOLDER']
                extracted_frames = extract_frames_from_video(file_path, frame_folder, num_frames=5)

                if not extracted_frames:
                    return render_template(
                        'index.html', 
                        prediction="Failed to extract frames from video",
                        uploaded_image=f'upload/{filename}'
                    )
                i = 0
                filename = os.path.splitext(filename)[0] + ".jpg"
                
                predictions = []
                num_no_faces = 0
                
                for frames in extracted_frames:
                    img_with_boxes, cropped_images = process_images(frames, filename)
                    if img_with_boxes == False:
                        num_no_faces+=1

                        if num_no_faces == 5:
                            return render_template(
                                'index.html', 
                                prediction="No faces detected",
                                uploaded_image=f'upload/frame_0.jpg'
                            )
                        
                        continue
                    
                    if i == 0:
                        # save processed image that has bounding boxes
                        processed_filename = f"processed_{filename}"
                        processed_path = os.path.join(app.config['PROCESSED_FOLDER'], processed_filename)
                        img_with_boxes.save(processed_path)
                    
                    i=1
                    # face predictions
                    prediction = predict_image(cropped_images[0])
                    face_predictions = []
                    for img in cropped_images:
                        prediction = predict_image(img)
                        face_predictions.append(prediction)
                    
                    if face_predictions:
                        prediction = get_frame_prediction(face_predictions)
                        
                    predictions.append(prediction)
                if predictions:
                    prediction = get_video_prediction(predictions)
                    
            else: 
                img_with_boxes, cropped_images = process_images(file_path, filename)

                # save processed image that has bounding boxes
                processed_filename = f"processed_{filename}"
                processed_path = os.path.join(app.config['PROCESSED_FOLDER'], processed_filename)
                img_with_boxes.save(processed_path)

                predictions=[]
                # prediction
                prediction = predict_image(cropped_images[0])
                for img in cropped_images:
                    prediction = predict_image(img)
                    predictions.append(prediction)
                    if prediction[0] == "Fake":
                        break
                    
            return render_template(
                'index.html',
                prediction=prediction,
                uploaded_image=f'upload/{filename}',
                processed_image=f'processed/{processed_filename}',
                cropped_images=cropped_images
            )

    return redirect(url_for('hello_world'))


if __name__ == "__main__":
    # create necessary folders if they don't exist
    os.makedirs(UPLOAD_FOLDER, exist_ok=True)
    os.makedirs(PROCESSED_FOLDER, exist_ok=True)
    os.makedirs(CROPPED_FOLDER, exist_ok=True)
    app.run(debug=True)