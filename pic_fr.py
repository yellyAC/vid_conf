import os
import os.path
import glob

import face_recognition
from PIL import Image, ImageDraw
import numpy as np

from datetime import datetime

def retrain_model():
    edcel_image = face_recognition.load_image_file("face_images/edcel.jpg")
    edcel_face_encoding = face_recognition.face_encodings(edcel_image)[0]

    ian_image = face_recognition.load_image_file("face_images/ian.jpg")
    ian_face_encoding = face_recognition.face_encodings(ian_image)[0]

    james_image = face_recognition.load_image_file("face_images/james.jpg")
    james_face_encoding = face_recognition.face_encodings(james_image)[0]

    parco_image = face_recognition.load_image_file("face_images/parco.jpg")
    parco_face_encoding = face_recognition.face_encodings(parco_image)[0]

    

    known_face_encodings = [
        edcel_face_encoding,
        ian_face_encoding,
        james_face_encoding,
        parco_face_encoding
    ]

    known_face_names = [
        "Edcel",
        "Ian",
        "James",
        "Parco"
    ]

    # attendance checking that will be stored in excel file
    def markAttendance(name):
        with open('Attendance.csv','r+') as f:
            myDataList = f.readlines()
            nameList = []
            for line in myDataList:
                entry = line.split(',')
                nameList.append(entry[0])
            
            if name not in nameList:
                now = datetime.now()
                dtString = now.strftime('%H:%M:%S')
                f.writelines(f'\n{name},{dtString}')


    # automatically import the latest screenshot file - JPG format only
    folder_path = f"{os.getenv('USERPROFILE')}\\Downloads"
    file_type = f'\*jpg'
    files = glob.glob(folder_path + file_type)
    max_file = max(files, key=os.path.getctime)
    print(max_file)

    unknown_image = face_recognition.load_image_file(max_file)

    face_locations = face_recognition.face_locations(unknown_image)
    face_encodings = face_recognition.face_encodings(unknown_image, face_locations)

    pil_image = Image.fromarray(unknown_image)
    draw = ImageDraw.Draw(pil_image)


    for (top, right, bottom, left), face_encoding in zip(face_locations, face_encodings):
        matches = face_recognition.compare_faces(known_face_encodings, face_encoding)
        
        name = "Unknown"
        
        face_distances = face_recognition.face_distance(known_face_encodings, face_encoding)
        best_match_index = np.argmin(face_distances)
        if matches[best_match_index]:
            name = known_face_names[best_match_index]
            print(name)
            markAttendance(name)
            
        draw.rectangle(((left, top), (right, bottom)), outline=(0, 0, 255))
        
        text_width, text_height = draw.textsize(name)
        draw.rectangle(((left, bottom - text_height - 10), (right, bottom)), fill=(0, 0, 255), outline=(0, 0, 255))
        draw.text((left + 6, bottom - text_height - 5), name, fill=(255, 255, 255, 255))
        
    del draw
    pil_image.show()

if __name__=="__main__":
    retrain_model()