import numpy as np
import cv2
import face_recognition
import imutils
import pickle
import time
import os

palette = (2 ** 11 - 1, 2 ** 15 - 1, 2 ** 20 - 1)


def compute_color_for_labels(label):
    """
    Simple function that adds fixed color depending on the class
    """
    color = [int((p * (label ** 2 - label + 1)) % 255) for p in palette]
    return tuple(color)

def face_recog(frame):
    print("[INFO] loading encodings...")
    pickle_file="../utils/encodings.pickle"
    os.path.isfile(pickle_file)
    data = pickle.loads(open("encodings.pickle", "rb").read())
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    r = frame.shape[1] / float(rgb.shape[1])
    boxes = face_recognition.face_locations(rgb,
		model="cnn")
    encodings = face_recognition.face_encodings(rgb, boxes)
    names = []
    found_face = 0
    for encoding in encodings:
		# attempt to match each face in the input image to our known
		# encodings
        matches = face_recognition.compare_faces(data["encodings"],	encoding)
        name = "Unknown"

		# check to see if we have found a match
        if True in matches:
			# find the indexes of all matched faces then initialize a
			# dictionary to count the total number of times each face
			# was matched
            matchedIdxs = [i for (i, b) in enumerate(matches) if b]
            counts = {}

			# loop over the matched indexes and maintain a count for
			# each recognized face face
            for i in matchedIdxs:
                name = data["names"][i]
                counts[name] = counts.get(name, 0) + 1

			# determine the recognized face with the largest number
			# of votes (note: in the event of an unlikely tie Python
			# will select first entry in the dictionary)
            name = max(counts, key=counts.get)
		
		# update the list of names
        names.append(name)

	# loop over the recognized faces
    for ((top, right, bottom, left), name) in zip(boxes, names):
		# rescale the face coordinates
        top = int(top * r)
        right = int(right * r)
        bottom = int(bottom * r)
        left = int(left * r)

        if(name != "Unknown" ):
            # draw the predicted face name on the image
            cv2.rectangle(frame, (left, top), (right, bottom),
    			(0, 255, 0), 2)
            y = top - 15 if top - 15 > 15 else top + 15
            cv2.putText(frame, name, (left, y), cv2.FONT_HERSHEY_SIMPLEX,
    			0.75, (0, 255, 0), 2)
            found_face = found_face + 1
    if(found_face > 0):
        return True
    else:
        return False

def draw_boxes(img, bbox, identities=None, offset=(0,0)):
    person = False
    bounding_rectangle = None
    for i,box in enumerate(bbox):
        x1,y1,x2,y2 = [int(i) for i in box]
        x1 += offset[0]
        x2 += offset[0]
        y1 += offset[1]
        y2 += offset[1]
        # box text and bar
        id = int(identities[i]) if identities is not None else 0    
        color = compute_color_for_labels(id)
        label = '{}{:d}'.format("", id)
        roi = img[y1:y2, x1:x2] 
        found_person = face_recog(roi)
        #t_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_PLAIN, 2 , 2)[0]
        if(found_person == True):
            cv2.rectangle(img,(x1, y1),(x2, y2),color,3)
            cv2.imwrite('demo/Test_gray.jpg', roi)
            height, width = roi.shape[:2]
            bounding_rectangle = (x1, y1, width, height)
            person = True
            return img, bounding_rectangle, person
        #cv2.rectangle(img,(x1, y1),(x1+t_size[0]+3,y1+t_size[1]+4), color,-1)
        #cv2.putText(img,label,(x1,y1+t_size[1]+4), cv2.FONT_HERSHEY_PLAIN, 2, [255,255,255], 2)
    return img, bounding_rectangle, person

if __name__ == '__main__':
    for i in range(82):
        print(compute_color_for_labels(i))
