import cv2
import math
from sklearn import neighbors
import os
import os.path
import pickle
from PIL import Image, ImageDraw
import face_recognition
from face_recognition.face_recognition_cli import image_files_in_folder
import numpy as np
from datetime import datetime
from datetime import date
import csv
from csv import writer


ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'JPG'}

def predict(X_frame, knn_clf=None, model_path=None, distance_threshold=0.5):
    if knn_clf is None and model_path is None:
        raise Exception("Must supply knn classifier either thourgh knn_clf or model_path")

    # Load a trained KNN model (if one was passed in)
    if knn_clf is None:
        with open(model_path, 'rb') as f:
            knn_clf = pickle.load(f)

    X_face_locations = face_recognition.face_locations(X_frame)

    # If no faces are found in the image, return an empty result.
    if len(X_face_locations) == 0:
        return []

    # Find encodings for faces in the test image
    faces_encodings = face_recognition.face_encodings(X_frame, known_face_locations=X_face_locations)

    # Use the KNN model to find the best matches for the test face
    closest_distances = knn_clf.kneighbors(faces_encodings, n_neighbors=1)
    are_matches = [closest_distances[0][i][0] <= distance_threshold for i in range(len(X_face_locations))]

    # Predict classes and remove classifications that aren't within the threshold
    return [(pred, loc) if rec else ("unknown", loc) for pred, loc, rec in zip(knn_clf.predict(faces_encodings), X_face_locations, are_matches)]





# Find Row in DB 
def filterFile(inputFileName,filterCriteria,columnToFilter):
		
	RVal = False
	#input file reader
	infile = open(inputFileName, "r")
	read = csv.reader(infile)
	headers = next(read) # header

	#for each row
	for row in read:
			
		if (row[columnToFilter] == filterCriteria and row[2] == date.today().strftime('%Y-%m-%d')):
			RVal = True
			break



	return RVal
	

def append_list_as_row(file_Name, list_of_elemnts):
    with open(file_Name, 'a+', newline='') as write_obj:
        csv_writer=writer(write_obj)
        csv_writer.writerow(list_of_elemnts)






def rescale_frame(frame):
    width = int(frame.shape[1] * GetScale())
    height = int(frame.shape[0] * GetScale())
    dim = (width, height)
    return cv2.resize(frame, dim, fx=GetScale(), fy=GetScale(), interpolation =cv2.INTER_AREA)



def Attendance(fname):
	
	row_content = [fname, '20',date.today(),datetime.now()]
	if filterFile('emp.csv',fname,0) == False:
		append_list_as_row('emp.csv', row_content)

	
		
def show_prediction_labels_on_image(frame, predictions, scale):
    
    pil_image = Image.fromarray(frame)
    draw = ImageDraw.Draw(pil_image)

    for name, (top, right, bottom, left) in predictions:
        # enlarge the predictions for the full sized image.
        top *= scale
        right *= scale
        bottom *= scale
        left *= scale
        # Draw a box around the face using the Pillow module
        draw.rectangle(((left, top), (right, bottom)), outline="green")

        
		# Insert Attendance Row in DB
        Attendance(name)

		# There's a bug in Pillow where it blows up with non-UTF-8 text
        # when using the default bitmap font
        name = name.encode("UTF-8")
		
		# Draw a label with a name below the face
        text_width, text_height = draw.textsize(name)
        draw.rectangle(((left, bottom - text_height - 10), (right, bottom)), fill="darkgreen", outline="green")
        draw.text((left + 6, bottom - text_height - 5), name, fill="white")
		
    # Remove the drawing library from memory as per the Pillow docs.
    del draw
    # Save image in open-cv format to be able to show it.

    opencvimage = np.array(pil_image)
    return opencvimage




def GetScale():
	return 0.40
	

		
		
if __name__ == "__main__":
    # process one frame in every 30 frames for speed
    process_this_frame = 29
    print('Setting cameras up...')
    # multiple cameras can be used with the format url = 'http://username:password@camera_ip:port'
    url = 'http://admin:admin@192.168.0.106:8081/'
    cap = cv2.VideoCapture(0) # 'rtsp://admin:asAS1212@100.0.0.121/1'
	
		
	
	
    while 1 > 0:
        ret, frame = cap.read()
        if ret:
            # Different resizing options can be chosen based on desired program runtime.
            img = rescale_frame(frame)
			# Image resizing for more stable streaming
            #img = cv2.resize(frame, (0, 0), fx=GetScale(), fy=GetScale())
            process_this_frame = process_this_frame + 1
            if process_this_frame % 30 == 0:
                predictions = predict(img, model_path="BEK_model.clf")
            frame = show_prediction_labels_on_image(frame, predictions, 1/GetScale())
            cv2.imshow('camera', frame)
            if ord('q') == cv2.waitKey(10):
                cap.release()
                cv2.destroyAllWindows()
                exit(0)
