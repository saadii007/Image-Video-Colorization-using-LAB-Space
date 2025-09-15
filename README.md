# Image and Video Colorization using Lab Color Space and Deep Learning

## 📌 Overview
This project aims to provide a comprehensive solution for image and video colorization using deep learning techniques. Using the help of CIELab color space, convolutional neural networks (CNNs) and modern web technologies, the project enables users to easily add color to grayscale images and videos.

## 📌 CIELAB Color Space

The CIELAB (or LAB) color space is a color representation designed to approximate human vision. It separates an image into three channels:

- L → Lightness (intensity/brightness)

- a → Green to Red color component

- b → Blue to Yellow color component

In this project, the L channel (grayscale input) is fed to the neural network, which predicts the missing a and b channels. Combining them reconstructs the full-color image. LAB is chosen because it is device-independent and better aligned with human perception of color compared to RGB.


## 📌 Features
- Image colorization: Convert grayscale images to colorized versions.
- Video colorization: Extend image colorization to video content.
- User-friendly interface: Web-based interface for easy interaction and colorization.
- Real-time processing: Instant colorization of uploaded images and videos.
- Streamlit integration: Utilizes Streamlit for web application development and deployment.

## 📌 Requirements
- Python 3.110
- OpenCV
- Streamlit

## 📌 Usage
1. Clone the repository: git clone [https://github.com/saadii007/Image-Video-Colorization-using-LAB-Space.git](https://github.com/saadii007/Image-Video-Colorization-using-LAB-Space)
2. Install dependencies: pip install -r requirements.txt
3. Run the application: streamlit run app.py
4. Access the application:
Open your web browser and navigate to http://localhost:8501.

## 📌 Acknowledgements
- The colorization model is based on research by Richard Zhang, Phillip Isola, and Alexei A. Efros. In ECCV, 2016.
<div align="center">
    <img src="https://github.com/saadii007/Image-Video-Colorization-using-LAB-Space/assets/126228618/8161cdc7-3467-46f9-9c9d-b8ae20dd9a56" alt="Image" style="width: 400px;">
</div>
- Link - https://richzhang.github.io/colorization/ 

## 📌 Screenshots

1. Video Colorization

<img src="https://github.com/saadii007/Image-Video-Colorization-using-LAB-Space/assets/126228618/6e7a92f4-47c8-461f-bfa4-aac91dd1f8ad" alt="Image" style="width: 400px;">

2. Image Colorization

<div>
    <img src="https://github.com/saadii007/Image-Video-Colorization-using-LAB-Space/assets/126228618/f8c2cce8-9692-4824-903f-d5ce55fd6277" alt="Image" style="width: 400px;">
    <img src="https://github.com/saadii007/Image-Video-Colorization-using-LAB-Space/assets/126228618/6b2c6019-2289-469d-988a-df4b2299dd87" alt="Image" style="width: 400px;">
</div>









