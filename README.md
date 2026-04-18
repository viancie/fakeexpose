# FakeExpose  
Deepfake Detection System for Images and Videos

## About  
FakeExpose is an AI-powered application designed to detect deepfake-generated faces in both images and videos. With the increasing presence of synthetic media, the system aims to provide a reliable solution for identifying manipulated visual content and promoting digital authenticity.

The application combines computer vision and deep learning techniques to analyze facial features and determine whether a given input is real or artificially generated.

---

## Project Overview  

FakeExpose follows a two-stage detection pipeline:

1. **Face Detection**  
   Faces are first detected and extracted from images or video frames using YOLO, ensuring that the analysis focuses only on relevant facial regions.

2. **Deepfake Classification**  
   An optimized VGG19 model is then used to classify each detected face as either real or deepfake.

---

## Model Architecture  

- **Base Model:** VGG19 (pretrained)  
- **Optimization Strategy:**  
  - Selected layers are unfrozen for fine-tuning  
  - Enhanced feature learning for detecting subtle deepfake artifacts  
- **Output:** Binary classification (Real vs Fake)

---

## Results  

The optimized VGG19 model achieved improved performance compared to a frozen baseline:

- Better generalization on unseen data  
- Enhanced detection of subtle manipulations  
- Improved overall classification accuracy  

---

## Features  

- Supports detection for:
  - Images  
  - Videos (frame-by-frame analysis)  
- Face-focused detection using YOLO  
- Improved accuracy through model fine-tuning  
- Simple and efficient application workflow  

---

## Tech Stack  

- Python  
- TensorFlow  
- YOLO (Face Detection)  
- VGG19 (Transfer Learning)  
- Pandas  

---

## How It Works  

1. Upload an image or video  
2. Faces are detected using YOLO  
3. Detected faces are extracted and preprocessed  
4. Each face is passed through the trained VGG19 model  
5. The system outputs a prediction: **Real** or **Deepfake**  

---

## Future Improvements  

- Integrate Vision Transformer (ViT) models for comparison  
- Improve real-time video processing performance  
- Expand dataset diversity for better generalization  
- Deploy as a web-based application  

---

## Authors  

- Mary Chris Viancie Oceña  
- Shane Tophy Linganay  

---

This project is for academic and research purposes.  
