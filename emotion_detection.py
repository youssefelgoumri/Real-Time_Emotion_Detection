from PyQt5.QtWidgets import (
    QApplication,
    QMainWindow,
    QPushButton,
    QVBoxLayout,
    QHBoxLayout,
    QWidget,
    QLabel,
    QFileDialog,
)
from PyQt5.QtGui import QPixmap, QColor, QImage
from PyQt5.QtCore import Qt, QSize, QThread, pyqtSignal, pyqtSlot

import qdarktheme
import numpy as np
from keras.utils import img_to_array
from keras.models import  load_model

model = load_model("model_no_disgust_emotion.h5")
import sys, cv2

WIDTH, HEIGHT = 1000, 700
#EMOTIONS = ('Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral')
EMOTIONS = ['Angry', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']
face_haar_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

class LiveVideoThread(QThread):
    change_pixmap_signal = pyqtSignal(np.ndarray)
    emotion_signal = pyqtSignal(str)

    def __init__(self):
        super().__init__()
        self._run_flag = True

    def run(self):
        # capture from web cam
        cap = cv2.VideoCapture(0)
        while self._run_flag:
            ret, cv_img = cap.read()
            if ret:
                gray_image = cv2.cvtColor(cv_img, cv2.COLOR_BGR2GRAY)
                faces_detected = face_haar_cascade.detectMultiScale(gray_image, 1.32, 5)

                for (x, y, w, h) in faces_detected:
                    cv2.rectangle(cv_img, (x, y), (x + w, y + h), (255, 0, 0), thickness=7)
                    roi_gray = gray_image[y:y + w, x:x + h]  # cropping region of interest i.e. face area from  image
                    roi_gray = cv2.resize(roi_gray, (48, 48), interpolation=cv2.INTER_AREA)
                    
                    if np.sum([roi_gray])!=0:
                        #img_pixels = img_to_array(roi_gray)
                        img_pixels = roi_gray.astype('float32')
                        img_pixels = np.expand_dims(img_pixels, axis=0)
                        img_pixels /= 255

                        predictions = model.predict(img_pixels)

                        # find max indexed array
                        max_index = np.argmax(predictions[0])

                        predicted_emotion = EMOTIONS[max_index]

                        pred = predictions[0]*100
                        str = f""
                        for i, em in enumerate(EMOTIONS):
                            str += f'{em}: {pred[i]:.2f}%   '

                        self.emotion_signal.emit(str)

                        cv2.putText(cv_img, predicted_emotion, (int(x), int(y)), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                    else:
                        cv2.putText(cv_img, "No Faces", (30, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

                self.change_pixmap_signal.emit(cv_img)
        # shut down capture system
        cap.release()

    def stop(self):
        """Sets run flag to False and waits for thread to finish"""
        self._run_flag = False
        #self.wait()

class VideoThread(QThread):
    change_pixmap_signal = pyqtSignal(np.ndarray)
    emotion_signal = pyqtSignal(str)

    def __init__(self, filename):
        super().__init__()
        self._run_flag = True
        self.filename = filename

    def run(self):
        # capture from web cam
        cap = cv2.VideoCapture(self.filename)
        if (cap.isOpened() == False):
            print("Error opening video file")
        
        # Read until video is completed
        while self._run_flag and cap.isOpened():
            ret, cv_img = cap.read()
            if ret:
                gray_image = cv2.cvtColor(cv_img, cv2.COLOR_BGR2GRAY)
                faces_detected = face_haar_cascade.detectMultiScale(gray_image, 1.32, 5)

                for (x, y, w, h) in faces_detected:
                    cv2.rectangle(cv_img, (x, y), (x + w, y + h), (255, 0, 0), thickness=7)
                    roi_gray = gray_image[y:y + w, x:x + h]  # cropping region of interest i.e. face area from  image
                    roi_gray = cv2.resize(roi_gray, (48, 48), interpolation=cv2.INTER_AREA)
                    
                    if np.sum([roi_gray])!=0:
                        #img_pixels = img_to_array(roi_gray)
                        img_pixels = roi_gray.astype('float32')
                        img_pixels = np.expand_dims(img_pixels, axis=0)
                        img_pixels /= 255

                        predictions = model.predict(img_pixels)

                        # find max indexed array
                        max_index = np.argmax(predictions[0])

                        predicted_emotion = EMOTIONS[max_index]

                        pred = predictions[0]*100
                        str = f""
                        for i, em in enumerate(EMOTIONS):
                            str += f'{em}: {pred[i]:.2f}%   '

                        self.emotion_signal.emit(str)

                        cv2.putText(cv_img, predicted_emotion, (int(x), int(y)), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                    else:
                        cv2.putText(cv_img, "No Faces", (30, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)



                self.change_pixmap_signal.emit(cv_img)
        
        # shut down capture system
        cap.release()

    def stop(self):
        """Sets run flag to False and waits for thread to finish"""
        self._run_flag = False
        #self.wait()

class Window(QMainWindow):
    def __init__(self):
        super().__init__()

        # set the title of main window
        self.setWindowTitle('Emotion Detection')

        # set the size of window
        self.Width = 1366
        self.Height = int(0.618 * self.Width)
        self.resize(self.Width, self.Height)

        # add all widgets
        self.upload_img_btn = QPushButton('Upload Image', self)
        self.live_video_btn = QPushButton('Start Live Video', self)
        self.upload_video_btn = QPushButton('Upload Video', self)

        self.upload_img_btn.clicked.connect(self.upload_img)
        self.live_video_btn.clicked.connect(self.start_live_video)
        self.upload_video_btn.clicked.connect(self.upload_video)

        # add tabs
        self.image_label = QLabel()
        self.result_label = QLabel("Result should be here")
        self.result_label.setStyleSheet("font: bold 24px;")

        self.initUI()

    def initUI(self):
        right_layout = QVBoxLayout()
        right_layout.addWidget(self.upload_img_btn)
        right_layout.addWidget(self.live_video_btn)
        right_layout.addWidget(self.upload_video_btn)
        right_layout.addStretch(5)
        right_layout.setSpacing(20)
        right_widget = QWidget()
        right_widget.setLayout(right_layout)

        left_layout = QHBoxLayout()
        left_widget = QWidget()
        left_layout.addWidget(self.image_label, alignment=Qt.AlignmentFlag.AlignCenter)
        grey = QPixmap(WIDTH, HEIGHT)
        grey.fill(QColor('darkGray'))
        # set the image image to the grey pixmap
        self.image_label.setPixmap(grey)
        left_widget.setLayout(left_layout)

        main_layout = QHBoxLayout()
        main_layout.addWidget(left_widget)
        main_layout.setStretch(0, 20)
        main_layout.addWidget(right_widget, alignment=Qt.AlignmentFlag.AlignVCenter)
        main_layout.setStretch(1, 7)

        last_layout = QVBoxLayout()
        last_layout.addLayout(main_layout, stretch=50)
        last_layout.addWidget(self.result_label, alignment=Qt.AlignmentFlag.AlignTop, stretch=15)
        #last_layout.setStretch(1, 5)

        main_widget = QWidget()
        main_widget.setLayout(last_layout)
        self.setCentralWidget(main_widget)

    # ----------------- 
    # buttons

    def upload_img(self):
        print('upload button pressed')
        image = QFileDialog.getOpenFileName(None, 'OpenFile', '', "Image file(*.png *.jpg *.bmp)")
        
        if not image[0]:
            return

        cv_img = cv2.imread(image[0])
        gray_image = cv2.cvtColor(cv_img, cv2.COLOR_BGR2GRAY)
        faces_detected = face_haar_cascade.detectMultiScale(gray_image, 1.32, 5)

        for (x, y, w, h) in faces_detected:
            cv2.rectangle(cv_img, (x, y), (x + w, y + h), (255, 0, 0), thickness=7)
            roi_gray = gray_image[y:y + w, x:x + h]  # cropping region of interest i.e. face area from  image
            roi_gray = cv2.resize(roi_gray, (48, 48), interpolation=cv2.INTER_AREA)
               
            if np.sum([roi_gray])!=0:
                #img_pixels = img_to_array(roi_gray)
                img_pixels = roi_gray.astype('float32')
                img_pixels = np.expand_dims(img_pixels, axis=0)
                img_pixels /= 255

                predictions = model.predict(img_pixels)

                # find max indexed array
                max_index = np.argmax(predictions[0])

                pred = predictions[0]*100
                str = f""
                for i, em in enumerate(EMOTIONS):
                    str += f'{em}: {pred[i]:.2f}%   '

                predicted_emotion = EMOTIONS[max_index]
                
                self.result_label.setText("Result is:\n"+ str)
                cv2.putText(cv_img, predicted_emotion, (int(x), int(y)), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            else:
                cv2.putText(cv_img, "No Faces", (30, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        
        self.update_image(cv_img)
        print("image loaded")
        #self.image_label.setScaledContents(False)

    @pyqtSlot(QImage)
    def setImage(self, image):
        self.image_label.setPixmap(QPixmap.fromImage(image).scaled(QSize(WIDTH, HEIGHT), Qt.AspectRatioMode.KeepAspectRatio))
    
    @pyqtSlot(np.ndarray)
    def update_image(self, cv_img):
        """Updates the image_label with a new opencv image"""
        qt_img = self.convert_cv_qt(cv_img)
        self.image_label.setPixmap(qt_img.scaled(QSize(WIDTH, HEIGHT), Qt.AspectRatioMode.KeepAspectRatio))
    
    @pyqtSlot(str)
    def set_result_lbl(self, emotion: str):
        self.result_label.setText("Result is:\n" + emotion)

    def start_live_video(self):
        print('live video pressed')
        # Change label color to light blue
        self.live_video_btn.clicked.disconnect(self.start_live_video)
        # Change button to stop
        self.live_video_btn.setText('Stop live video')
        self.th = LiveVideoThread()
        self.th.change_pixmap_signal.connect(self.update_image)
        self.th.emotion_signal.connect(self.set_result_lbl)

        # start the thread
        self.th.start()
        self.live_video_btn.clicked.connect(self.th.stop)  # Stop the video if button clicked
        self.live_video_btn.clicked.connect(self.stop_live_video)
        self.upload_img_btn.setEnabled(False)
        self.upload_video_btn.setEnabled(False)

    
    def stop_live_video(self):
        print("stop live video")
        self.th.change_pixmap_signal.disconnect()
        self.live_video_btn.setText('Start live video')

        self.live_video_btn.clicked.disconnect(self.stop_live_video)
        self.live_video_btn.clicked.disconnect(self.th.stop)
        self.live_video_btn.clicked.connect(self.start_live_video)

        self.upload_img_btn.setEnabled(True)
        self.upload_video_btn.setEnabled(True)

    def upload_video(self):
        filename = QFileDialog.getOpenFileName(None, 'OpenFile', '', "Video file(*.mp4 *.mkv)")
        if not filename[0]:
            return
        print('upload video pressed')

        # Change label color to light blue
        self.upload_video_btn.clicked.disconnect(self.upload_video)
        # Change button to stop
        self.upload_video_btn.setText('Stop video')
        self.th1 = VideoThread(filename[0])
        self.th1.change_pixmap_signal.connect(self.update_image)
        self.th1.emotion_signal.connect(self.set_result_lbl)

        # start the thread
        self.th1.start()
        self.upload_video_btn.clicked.connect(self.th1.stop)  # Stop the video if button clicked
        self.upload_video_btn.clicked.connect(self.stop_upload_video)
        self.upload_img_btn.setEnabled(False)
        self.live_video_btn.setEnabled(False)

    def stop_upload_video(self):
        print('stop video played pressed')
        self.th1.change_pixmap_signal.disconnect()
        self.upload_video_btn.setText('Upload video')

        self.upload_video_btn.clicked.disconnect(self.stop_upload_video)
        self.upload_video_btn.clicked.disconnect(self.th1.stop)
        self.upload_video_btn.clicked.connect(self.upload_video)

        self.upload_img_btn.setEnabled(True)
        self.live_video_btn.setEnabled(True)

    def detect_emotion(self):
        print('detect emotion pressed')

    
    
    def convert_cv_qt(self, cv_img):
        """Convert from an opencv image to QPixmap"""
        rgb_image = cv2.cvtColor(cv_img, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb_image.shape
        bytes_per_line = ch * w
        convert_to_Qt_format = QImage(rgb_image.data, w, h, bytes_per_line, QImage.Format_RGB888)
        p = convert_to_Qt_format.scaled(WIDTH, HEIGHT, Qt.AspectRatioMode.KeepAspectRatio)
        return QPixmap.fromImage(p)


if __name__ == '__main__':
    app = QApplication(sys.argv)
    qdarktheme.setup_theme("dark")
    ex = Window()
    ex.show()
    sys.exit(app.exec_())
