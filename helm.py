import cv2

class HelmetDetector:
    def __init__(self):
        # Load Haar cascades
        self.face_cascade = cv2.CascadeClassifier("xml/myhaar.xml")
        self.motor_cascade = cv2.CascadeClassifier("xml/motor-v4.xml")

    def detect_helmet(self, roi):
        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        faces = self.face_cascade.detectMultiScale(gray, 1.1, 4)
        motors = self.motor_cascade.detectMultiScale(gray, 1.1, 4)

        if len(motors) > 0:
            if len(faces) > 0:
                return "No Helmet"
            else:
                return "Helmet"
        else:
            return "No Bike"
