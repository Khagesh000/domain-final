import cv2
import face_recognition
import pickle
import datetime
import os
import csv
from plyer import notification



# Load the trained KNN classifier
with open('trained_model.clf', 'rb') as f:
    knn_clf = pickle.load(f)



# Dictionary to store verification images and their encodings
verification_images = {
    'ABHI': r'C:\Users\vijay\Documents\chando 55 face_recognition_project-main\FINAL\DATASET\ABHI\0.jpg',
    'VIJAY': r'C:\Users\vijay\Documents\chando 55 face_recognition_project-main\FINAL\DATASET\VIJAY\1.jpg',
    'GOWTHAM': r'C:\Users\vijay\Documents\chando 55 face_recognition_project-main\FINAL\DATASET\GOWTHAM\40.jpg',
    'ABHIRAM': r'C:\Users\vijay\Documents\chando 55 face_recognition_project-main\FINAL\DATASET\ABHIRAM\0.jpg',
    'ABHISHIEK': r'C:\Users\vijay\Documents\chando 55 face_recognition_project-main\FINAL\DATASET\ABHISHIEK\0.jpg',
    'ARUN': r'C:\Users\vijay\Documents\chando 55 face_recognition_project-main\FINAL\DATASET\ARUN\0.jpg',
    'BHEEMARAJU': r'C:\Users\vijay\Documents\chando 55 face_recognition_project-main\FINAL\DATASET\BHEEMARAJU\0.jpg',
    'GIRISH': r'C:\Users\vijay\Documents\chando 55 face_recognition_project-main\FINAL\DATASET\GIRISH\0.jpg',
    'GOUSE': r'C:\Users\vijay\Documents\chando 55 face_recognition_project-main\FINAL\DATASET\GOUSE\0.jpg',
    'GOWTHAM': r'C:\Users\vijay\Documents\chando 55 face_recognition_project-main\FINAL\DATASET\GOWTHAM\0.jpg',
    'KARTHIK': r'C:\Users\vijay\Documents\chando 55 face_recognition_project-main\FINAL\DATASET\KARTHIK\0.jpg',
    'KISHORE': r'C:\Users\vijay\Documents\chando 55 face_recognition_project-main\FINAL\DATASET\KISHORE\0.jpg',
    'LAVAKUMAR': r'C:\Users\vijay\Documents\chando 55 face_recognition_project-main\FINAL\DATASET\LAVAKUMAR\0.jpg',
    'LOKESH': r'C:\Users\vijay\Documents\chando 55 face_recognition_project-main\FINAL\DATASET\LOKESH\15.jpg',
    'MAHESH': r'C:\Users\vijay\Documents\chando 55 face_recognition_project-main\FINAL\DATASET\MAHESH\14.jpg',
    'MANIKANTA': r'C:\Users\vijay\Documents\chando 55 face_recognition_project-main\FINAL\DATASET\MANIKANTA\39.jpg',
    'MOULI': r'C:\Users\vijay\Documents\chando 55 face_recognition_project-main\FINAL\DATASET\MOULI\14.jpg',
    'SURESH': r'C:\Users\vijay\Documents\chando 55 face_recognition_project-main\FINAL\DATASET\SURESH\14.jpg',
    'VARDHAN': r'C:\Users\vijay\Documents\chando 55 face_recognition_project-main\FINAL\DATASET\VARDHAN\26.jpg',
    'VIVEK': r'C:\Users\vijay\Documents\chando 55 face_recognition_project-main\FINAL\DATASET\VIVEK\14.jpg',
    # Add more persons as needed
}



# Load verification images and their encodings
verification_encodings = {}
for person, image_path in verification_images.items():
    verification_image = face_recognition.load_image_file(image_path)
    # Ensure that at least one face is detected
    face_encodings = face_recognition.face_encodings(verification_image)
    if face_encodings:
        verification_encodings[person] = face_encodings[0]
    else:
        print(f"No face detected in {person}'s image: {image_path}")



# Open the video capture
video = cv2.VideoCapture(1)



# CSV file to record recognition and verification events
csv_file_path = 'recognition_log.csv'

# Folder to save all recognized faces
output_folder = 'recognized_faces'
os.makedirs(output_folder, exist_ok=True)

# File to save recognized names
recognized_names_file_path = 'recognized_names.txt'

# Metrics variables
start_time = None
total_recognitions = 0
average_recognition_time = 0.0
confidence_threshold = 0.5  # Set the confidence threshold as needed

# CSV columns
csv_columns = ['Timestamp', 'Year', 'Month', 'Date', 'Hour', 'Minute', 'Second', 'Name']

with open(csv_file_path, 'w', newline='') as csv_file:
    writer = csv.DictWriter(csv_file, fieldnames=csv_columns)
    writer.writeheader()

while True:
    # Read a frame from the video
    ret, frame = video.read()

    # Detect faces using face_recognition library
    face_locations = face_recognition.face_locations(frame)

    # Ensure that there are face encodings
    if len(face_locations) > 0:
        face_encodings = face_recognition.face_encodings(frame, known_face_locations=face_locations)

        # Perform face verification
        for face_encoding, face_location in zip(face_encodings, face_locations):
            name = knn_clf.predict([face_encoding])[0]
            (top, right, bottom, left) = face_location

            # Perform face recognition with confidence
            face_distances = face_recognition.face_distance(list(verification_encodings.values()), face_encoding)
            min_distance = min(face_distances)
            if min_distance < confidence_threshold:
                recognized_name = list(verification_encodings.keys())[list(face_distances).index(min_distance)]
            else:
                recognized_name = "Unknown"

            # Log recognition event
            with open(csv_file_path, 'a', newline='') as csv_file:
                timestamp = datetime.datetime.now()
                writer = csv.DictWriter(csv_file, fieldnames=csv_columns)
                writer.writerow({
                    'Timestamp': timestamp.strftime('%Y-%m-%d %H:%M:%S'),
                    'Year': timestamp.year,
                    'Month': timestamp.month,
                    'Date': timestamp.day,
                    'Hour': timestamp.hour,
                    'Minute': timestamp.minute,
                    'Second': timestamp.second,
                    'Name': recognized_name
                })

            # Display the name of the recognized person
            font = cv2.FONT_HERSHEY_DUPLEX
            cv2.putText(frame, f"{recognized_name}", (left + 6, bottom - 6), font, 0.5, (255, 255, 0), 1)  # Yellow text

            # Draw a rectangle around the face
            cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)

            if recognized_name != "Unknown":
                # Save the recognized face to the output folder
                face_image = frame[top:bottom, left:right]
                face_filename = os.path.join(output_folder,
                                             f"{recognized_name}_{timestamp.strftime('%Y%m%d%H%M%S')}.jpg")
                cv2.imwrite(face_filename, face_image)

                # Save the recognized name to the text file
                with open(recognized_names_file_path, 'a') as names_file:
                    names_file.write(f"{recognized_name}\n")

                # Display notification
                notification_title = "Face Recognition"
                notification_text = f"Recognized: {recognized_name}"
                notification.notify(title=notification_title, message=notification_text, app_icon=None, timeout=10)

                # Update metrics
                total_recognitions += 1
                if start_time is not None:
                    elapsed_time = (datetime.datetime.now() - start_time).total_seconds()
                    average_recognition_time = (
                            average_recognition_time * (total_recognitions - 1) + elapsed_time) / total_recognitions

            # Record start time for the next face
            start_time = datetime.datetime.now()

    # Display metrics on the frame
    cv2.putText(frame, f"Total Recognitions: {total_recognitions}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                (0, 255, 255), 2)  # Yellow text
    cv2.putText(frame, f"Average Recognition Time: {average_recognition_time:.2f} seconds", (10, 60),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)  # Yellow text

    # Display the frame
    cv2.imshow("Face Recognition and Verification", frame)

    # Break the loop if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the video capture and close all windows
video.release()
cv2.destroyAllWindows()
