# Real-Time Emotion Detection Desktop App
This desktop application utilizes computer vision techniques and deep learning models to perform real-time emotion detection using your computer's camera. It detects faces in the camera feed and predicts the emotions associated with each face.

## Features
* Detect face emotions fom image and video
* Real-time emotion detection using your computer's camera
* Detection of multiple faces simultaneously
* Displays the predicted emotion label on each detected face
* Easy-to-use interface

## Requirements
* Python 3.x
* OpenCV
* NumPy
* Keras (for loading and using the pre-trained emotion detection model)

## Installation
1. Clone this repository to your local machine:

   ```
   git clone https://github.com/youssefelgoumri/Real-Time_Emotion_Detection.git

2. Install the required Python packages:

   ```
   pip install opencv-python numpy keras


# Usage
1. Navigate to the project directory:

   ```
   cd real-time-emotion-detection

2. Run the application:

   ```
   python emotion_detection.py

3. The application will open a window showing the camera feed with real-time emotion detection. Detected faces will be outlined, and the predicted emotion label will be displayed on each face.

4. Press the 'q' key to quit the application.

## Customization
You can customize the application by modifying the following:

* Model: You can replace the pre-trained emotion detection model (model_no_disgust_emotion.h5) with your own trained model if desired. Ensure that the model architecture matches and the input dimensions are appropriate.
* Face Detection: If you find that the face detection is not accurate enough, you can experiment with different Haar cascade classifiers or use other face detection techniques.
* User Interface: If you want to enhance the user interface or add additional features, you can modify the emotion_detection.py script accordingly.

## License
This project is licensed under the [MIT License](LICENSE).
