import cv2
import face_recognition
import os
import csv

known_faces_directory = "known_faces"
csv_filename = "user_data.csv"

# Create the "known_faces" directory if it doesn't exist
if not os.path.exists(known_faces_directory):
    os.mkdir(known_faces_directory)

# Initialize the webcam with reduced frame size and frame rate
video_capture = cv2.VideoCapture(0)
video_capture.set(3, 640)  # Set width to 640
video_capture.set(4, 480)  # Set height to 480
video_capture.set(5, 10)   # Set frame rate to 10 fps

# Load known faces and their names
known_face_encodings = []
known_face_names = []
for file_name in os.listdir(known_faces_directory):
    if file_name.endswith(".jpg"):
        name = os.path.splitext(file_name)[0]
        image_path = os.path.join(known_faces_directory, file_name)
        known_image = face_recognition.load_image_file(image_path)
        face_encodings = face_recognition.face_encodings(known_image)
        if face_encodings:
            known_face_encodings.append(face_encodings[0])
            known_face_names.append(name)

# Create or open the CSV file for saving user data
with open(csv_filename, mode='a', newline='') as csv_file:
    csv_writer = csv.writer(csv_file)
    # Check if the CSV file is empty and add headers if necessary
    if os.path.getsize(csv_filename) == 0:
        csv_writer.writerow(['Name', 'Password', 'ImageName'])

while True:
    # Capture a frame from the webcam
    ret, frame = video_capture.read()

    # Find face locations and encodings in the captured frame
    face_locations = face_recognition.face_locations(frame)
    face_encodings = face_recognition.face_encodings(frame, face_locations)

    # Display the captured image
    cv2.imshow("Capture an Image", frame)

    # Wait for the user to press a key
    if cv2.waitKey(1) & 0xFF == ord('c'):
        if face_encodings:
            for face_encoding in face_encodings:
                # Check if the captured face matches any known face
                matches = face_recognition.compare_faces(known_face_encodings, face_encoding)

                if True in matches:
                    name = known_face_names[matches.index(True)]
                    print(f"User '{name}' already exists.")
                else:
                    # Prompt the user to enter their name and password
                    name = input("Enter your name: ")
                    password = input("Enter your password: ")

                    # Prompt the user to enter a name for the captured image
                    image_name = name

                    # Save the captured image
                    image_path = os.path.join(known_faces_directory, image_name + ".jpg")
                    cv2.imwrite(image_path, frame)

                    # Save the user data to the CSV file
                    with open(csv_filename, mode='a', newline='') as csv_file:
                        csv_writer = csv.writer(csv_file)
                        csv_writer.writerow([name, password, image_name])

                    print(f"User '{name}' has been added.")
        else:
            print("No face detected in the captured image. Please try again.")

        break

# Release the webcam and close OpenCV windows
video_capture.release()
cv2.destroyAllWindows()
