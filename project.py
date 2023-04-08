import face_recognition
import cv2
import numpy as np
import csv
import os
from datetime import datetime

video_capture = cv2.VideoCapture(0)
kausar_image = face_recognition.load_image_file("photos/kausar.jpg")
kausar_encoding = face_recognition.face_encodings(kausar_image)[0]

manzur_image = face_recognition.load_image_file("photos/manzur.jpg")
manzur_encoding = face_recognition.face_encodings(manzur_image)[0]

mohsin_image = face_recognition.load_image_file("photos/mohsin.jpg")
mohsin_encoding = face_recognition.face_encodings(mohsin_image)[0]

nafees_image = face_recognition.load_image_file("photos/nafees.jpg")
nafees_encoding = face_recognition.face_encodings(nafees_image)[0]

ishaaq_image = face_recognition.load_image_file("photos/ishaaq.jpg")
ishaaq_encoding = face_recognition.face_encodings(ishaaq_image)[0]

anabia_image = face_recognition.load_image_file("photos/anabia.jpg")
anabia_encoding = face_recognition.face_encodings(anabia_image)[0]

ikraam_image = face_recognition.load_image_file("photos/ikraam.jpg")
ikraam_encoding = face_recognition.face_encodings(ikraam_image)[0]

elon_image = face_recognition.load_image_file("photos/elon.jpg")
elon_encoding = face_recognition.face_encodings(elon_image)[0]

known_face_encoding = [
kausar_encoding,
manzur_encoding,
mohsin_encoding,
nafees_encoding,
ishaaq_encoding,
anabia_encoding,
ikraam_encoding,
elon_encoding,
]

known_faces_names = [
"kausar",
"manzur",
"mohsin",
"nafees",
"ishaaq",
"anabia",
"ikraam",
"elon",
]

students = known_faces_names.copy()

face_location = []
face_encodings = []
face_names = []
s=True

now = datetime.now()
current_date = now.strftime("%Y-%m-%d")
f=open(current_date+'.csv','w+',newline = '')
lnwriter = csv.writer(f)

while True:
	_,frame = video_capture.read()
	small_frame = cv2.resize(frame,(0,0),fx=0.25,fy=0.25)
	rgb_small_frame = small_frame[:,:,::-1]
	if s:
		face_locations = face_recognition.face_locations(rgb_small_frame)
		face_encodings = face_recognition.face_encodings(rgb_small_frame,face_locations)
		face_names =  []
		for face_encoding in face_encodings:
			matches = face_recognition.compare_faces(known_face_encoding,face_encoding)
			name=""
			face_distance = face_recognition.face_distance(known_face_encoding,face_encoding)
			best_match_index = np.argmin(face_distance)
			if matches[best_match_index]:
				name = known_faces_names[best_match_index]

			face_names.append(name)
			if name in known_faces_names:
				if name in students:
					students.remove(name)
					print(students)
					current_time = now.strftime("%H-%M-%S")
					lnwriter.writerow([name,current_time])
	cv2.imshow("attendence system",frame)
	if cv2.waitKey(1) & 0xFF == ord('q'):
		break

video_capture.release()
cv2.destroyAllWindows()
f.close()					