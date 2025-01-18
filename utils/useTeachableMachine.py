import os

from tensorflow.keras.models import load_model
import cv2  # Install opencv-python
import numpy as np




class CircleRecognition:
    def __init__(self):
        self.camera = cv2.VideoCapture(0)
        base_path = os.path.dirname(os.path.abspath(__file__))
        self.model = load_model(os.path.join(base_path, "../Models/teachable/keras_model.h5"), compile=False)
        self.class_names = open(os.path.join(base_path, "../Models/teachable/labels.txt"), "r").readlines()

    def predict(self, image):
        # Resize the raw image into (224-height,224-width) pixels
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = cv2.resize(image, (224, 224), interpolation=cv2.INTER_AREA)
        # display image for debugging
        #cv2.imshow("Image", image)

        # Make the image a numpy array and reshape it to the models input shape.
        image = np.asarray(image, dtype=np.float32).reshape(1, 224, 224, 3)

        # Normalize the image array
        image = (image / 127.5) - 1

        # Predicts the model
        prediction = self.model.predict(image)
        index = np.argmax(prediction)
        class_name = self.class_names[index]
        confidence_score = prediction[0][index]

        # Print prediction and confidence score
        print("Class:", class_name[2:], end="")
        print("Confidence Score:", str(np.round(confidence_score * 100))[:-2], "%")

        return class_name, confidence_score

