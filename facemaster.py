import cv2
import numpy as np
import training_data

# Load the trained model
faces, labels = training_data.prepare_training_data("training-data")
face_recognizer = cv2.face.LBPHFaceRecognizer_create()
face_recognizer.train(faces, np.array(labels))

def main():
    cap = cv2.VideoCapture(0)
    faceCascade = cv2.CascadeClassifier("opencv-files/haarcascade_frontalface_alt.xml")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = faceCascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

        for (x, y, w, h) in faces:
            face_img = gray[y:y+h, x:x+w]
            label, confidence = face_recognizer.predict(face_img)

            if confidence < 50:  # Adjust this threshold based on your model's performance
                label_text = f"Matched: ID {label}"
                color = (0, 255, 0)  # Green color for matched faces
            else:
                label_text = "Unknown"
                color = (0, 0, 255)  # Red color for unknown faces

            cv2.putText(frame, label_text, (x, y-5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
            cv2.rectangle(frame, (x, y), (x+w, y+h), color, 2)

        cv2.imshow('Face Recognition', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
