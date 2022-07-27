import cv2
import dlib
import numpy as np

import face_recognition
import matplotlib.pyplot as plt
from skimage.feature import hog
from skimage import data, exposure

from sklearn import svm
from skimage.transform import resize

ppc = 16
hog_images = []
hog_features = []
hog_features_test = []
categories = []
categories = np.array(categories).reshape(len(categories), 1)

# age = 7 - 11 (anak2), 12 - 25(remaja), 26 - 45(dewasa), >45 (tua)
hog_images_age = []
hog_features_age = []
hog_features_test_age = []
ages = []
ages = np.array(ages).reshape(len(ages), 1)

cascPath = "/usr/local/lib/python3.7/site-packages/cv2/data/haarcascade_frontalface_default.xml"
eyePath = "/usr/local/lib/python3.7/site-packages/cv2/data/haarcascade_eye.xml"
smilePath = "/usr/local/lib/python3.7/site-packages/cv2/data/haarcascade_smile.xml"

faceCascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
eyeCascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')
smileCascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_smile.xml')

detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor('shape_predictor_68_face_landmarks.dat')

cap = cv2.VideoCapture(0)
font = cv2.FONT_HERSHEY_SIMPLEX

while True:
    # Capture the image from the webcam
    ret, image = cap.read()
    # Convert the image color to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # Detect the face
    faces = detector(gray, 1)

    # Detect landmarks for each face
    for rect in faces:
        x1 = rect.left()
        y1 = rect.top()
        x2 = rect.right()
        y2 = rect.bottom()

        # draw rectangle around face
        # cv2.rectangle(image, (x1,y1), (x2,y2), (0,255,0), 4)

        # Get the landmark points
        shape = predictor(gray, rect)
        # Convert it to the NumPy Array
        shape_np = np.zeros((68, 2), dtype="int")

        for i in range(0, 68):
            shape_np[i] = (shape.part(i).x, shape.part(i).y)
        shape = shape_np

        # Display the landmarks
        for i, (x, y) in enumerate(shape):
            # Draw the circle to mark the keypoint
            cv2.circle(image, (x, y), 3, (0, 0, 255), -1)

    # Display the image
    cv2.imshow('Face Landmark Detection', image)

    # capturing and save
    key = cv2.waitKey(1)
    if key == ord('s'):  # for saving image press S
        print('captured')
        cv2.imwrite(filename='saved_img.jpg', img=image)
        cap.release()
        img_new = cv2.imread('saved_img.jpg', cv2.IMREAD_GRAYSCALE)
        img_new = cv2.imshow("Captured Image", img_new)
        cv2.waitKey(1650)
        cv2.destroyAllWindows()
        print("Processing image...")
        img_ = cv2.imread('saved_img.jpg', cv2.IMREAD_ANYCOLOR)
        print("Image saved!")
        break
    elif key == ord('q'):
        print("Turning off camera.")
        cap.release()
        print("Camera off.")
        print("Program ended.")
        cv2.destroyAllWindows()
        break

    # ==Histogram of Oriented Gradient==
    # pre trained
    faces = faceCascade.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(200, 200),
        flags=cv2.CASCADE_SCALE_IMAGE
    )

    # Draw a rectangle around the faces
    for (x, y, w, h) in faces:
        cv2.rectangle(image, (x, y), (x + w, y + h), (255, 0, 0), 3)
        # Creating two regions of interest
        roi_gray = gray[y:y + h, x:x + w]
        roi_color = image[y:y + h, x:x + w]
        cv2.putText(image, 'Face', (x, y), font, 2, (255, 0, 0), 5)

        # for mouth detected
        smile = smileCascade.detectMultiScale(
            roi_gray,
            scaleFactor=1.1,
            minNeighbors=35,
            minSize=(25, 25),
            flags=cv2.CASCADE_SCALE_IMAGE
        )

        # draw rectangle
        for (sx, sy, sw, sh) in smile:
            cv2.rectangle(roi_color, (sh, sy), (sx + sw, sy + sh), (255, 0, 0), 2)
            cv2.putText(image, 'Mouth', (x + sx, y + sy), 1, 1, (0, 255, 0), 1)

        # for each eye detected
        eyes = eyeCascade.detectMultiScale(roi_gray)
        index = 0
        # Creating for loop in order to divide one eye from another
        for (ex, ey, ew, eh) in eyes:
            if index == 0:
                eye_1 = (ex, ey, ew, eh)
            elif index == 1:
                eye_2 = (ex, ey, ew, eh)
            cv2.rectangle(roi_color, (ex, ey), (ex + ew, ey + eh), (0, 255, 0), 2)
            cv2.putText(image, 'Eye', (x + ex, y + ey), 1, 1, (0, 255, 0), 1)
            index = index + 1

        # differentiating between left_eye and right_eye, smaller eye will be our left_eye
        if eye_1[0] < eye_2[0]:
            left_eye = eye_1
            right_eye = eye_2
        else:
            left_eye = eye_2
            right_eye = eye_1

        # Calculating coordinates of a central points of the rectangles
        left_eye_center = (int(left_eye[0] + (left_eye[2] / 2)), int(left_eye[1] + (left_eye[3] / 2)))
        print("left eye center : ", left_eye_center)
        left_eye_x = left_eye_center[0]
        print("left eye x : ", left_eye_x)
        left_eye_y = left_eye_center[1]
        print("left eye y : ", left_eye_y)

        right_eye_center = (int(right_eye[0] + (right_eye[2] / 2)), int(right_eye[1] + (right_eye[3] / 2)))
        print("right eye center  : ", right_eye_center)
        right_eye_x = right_eye_center[0]
        print("right eye x : ", right_eye_x)
        right_eye_y = right_eye_center[1]
        print("right eye y : ", right_eye_y)

        cv2.circle(roi_color, left_eye_center, 5, (255, 0, 0), -1)
        cv2.circle(roi_color, right_eye_center, 5, (255, 0, 0), -1)
        cv2.line(roi_color, right_eye_center, left_eye_center, (0, 200, 200), 3)

        # draw a horizontal line and calculate the angle between that line
        if left_eye_y > right_eye_y:
            A = (right_eye_x, left_eye_y)
            # Integer -1 indicates that the image will rotate in the clockwise direction
            direction = -1
        else:
            A = (left_eye_x, right_eye_y)
            # Integer 1 indicates that image will rotate in the counter clockwise direction
            direction = 1

        cv2.circle(roi_color, A, 5, (255, 0, 0), -1)

        cv2.line(roi_color, right_eye_center, left_eye_center, (0, 200, 200), 3)
        cv2.line(roi_color, left_eye_center, A, (0, 200, 200), 3)
        cv2.line(roi_color, right_eye_center, A, (0, 200, 200), 3)

        # find the length of two legs of a right triangle to calculate the angle
        delta_x = right_eye_x - left_eye_x
        print("delta x : ", delta_x)
        delta_y = right_eye_y - left_eye_y
        print("delta y : ", delta_y)
        angle = np.arctan(delta_y / delta_x)
        angle = (angle * 180) / np.pi
        print("angle : ", angle)

        # Width and height of the image
        h, w = image.shape[:2]
        # Calculating a center point of the image
        # Integer division "//"" ensures that we receive whole numbers
        center = (w // 2, h // 2)
        # Defining a matrix M and calling
        M = cv2.getRotationMatrix2D(center, (angle), 1.0)
        # Applying the rotation to image
        rotated = cv2.warpAffine(image, M, (w, h))

        # count the total number of face detected
        cv2.putText(image, 'Number of Faces : ' + str(len(faces)), (40, 40), font, 1, (255, 0, 0), 2)

    # Display the resulting frame
    #cv2.imshow('HOG', image)

    # Press the escape button to terminate the code
    if cv2.waitKey(10) == 27:
        break

image = face_recognition.load_image_file("saved_img.jpg")
#cv2.imshow('HOG', image)
print(image.shape)

# creating hog feature
gray_img = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
fd, hog_image = hog(gray_img, orientations=8, pixels_per_cell=(ppc, ppc), cells_per_block=(4, 4), visualize=True)
print(hog_image.shape)
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8, 4), sharex=True, sharey=True)

ax1.axis('off')
ax1.imshow(image, cmap=plt.cm.gray)
ax1.set_title('Input image')

# Rescale histogram for better display
hog_image_rescaled = exposure.rescale_intensity(hog_image, in_range=(0, 10))

ax2.axis('off')
ax2.imshow(hog_image_rescaled, cmap=plt.cm.gray)
ax2.set_title('Histogram of Oriented Gradients')
plt.show()

# ======== SVM Classifier ========

#gender training
for i in range(1, 401):
    img = cv2.imread(f"train/im{i}.jpg")
    if i <= 201:
        categories = np.append(categories, 1)
    else:
        categories = np.append(categories, 2)

    img_resized = resize(img, (150, 150))
    # print(img_resized)
    img_parsed = np.float32(img_resized)
    greyscale_img = cv2.cvtColor(img_parsed, cv2.COLOR_BGR2GRAY)
    fd, hog_image = hog(greyscale_img, orientations=8, pixels_per_cell=(ppc, ppc),
                        cells_per_block=(4, 4), block_norm='L2', visualize=True)
    hog_images.append(hog_image)
    hog_features.append(fd)
#print("training gender from image")
clf = svm.SVC()
hog_features = np.array(hog_features)
gender_model = clf.fit(hog_features, categories)

#gender predict
img_test = cv2.imread("saved_img.jpg")
img_resized = resize(img_test, (150, 150))
img_parsed = np.float32(img_resized)
greyscale_img = cv2.cvtColor(img_parsed, cv2.COLOR_BGR2GRAY)
fd, hog_image = hog(greyscale_img, orientations=8, pixels_per_cell=(ppc, ppc),
                    cells_per_block=(4, 4), block_norm='L2', visualize=True)
hog_images.append(hog_image)
hog_features_test.append(fd)

hog_features_test = np.array(hog_features_test)
predicted = clf.predict(hog_features_test)
print(f"Gender Predicted: {predicted}")

#age training
for k in range(1, 401):
    img = cv2.imread(f"train/im{k}.jpg")
    if k <= 50:
        ages = np.append(ages, 1) #man, anak2
    elif 50 < k <= 100:
        ages = np.append(ages, 2) #man, remaja
    elif 100 < k <= 150:
        ages = np.append(ages, 3) #man, dewasa
    elif 150 < k <= 200:
        ages = np.append(ages, 4) #man, tua
    elif 200 < k <= 250:
        ages = np.append(ages, 5) #woman, anak
    elif 250 < k <= 300:
        ages = np.append(ages, 6) #woman, remaja
    elif 300 < k <= 350:
        ages = np.append(ages, 7) #woman, dewasa
    else:
        ages = np.append(ages, 8) #woman, tua

    img_resized = resize(img, (150, 150))
    #print(img_resized)
    img_parsed = np.float32(img_resized)
    greyscale_img = cv2.cvtColor(img_parsed, cv2.COLOR_BGR2GRAY)
    fd, hog_image = hog(greyscale_img, orientations=8, pixels_per_cell=(ppc, ppc),
                        cells_per_block=(4, 4), block_norm='L2', visualize=True)
    hog_images_age.append(hog_image)
    hog_features_age.append(fd)
#print("training age from image")
clf = svm.SVC()
hog_features_age = np.array(hog_features_age)
age_model = clf.fit(hog_features_age, ages)

#age predict
img_test = cv2.imread("saved_img.jpg")
img_resized = resize(img_test, (150, 150))
img_parsed = np.float32(img_resized)
greyscale_img = cv2.cvtColor(img_parsed, cv2.COLOR_BGR2GRAY)
fd, hog_image = hog(greyscale_img, orientations=8, pixels_per_cell=(ppc, ppc),
                        cells_per_block=(4, 4), block_norm='L2', visualize=True)
hog_images_age.append(hog_image)
hog_features_test_age.append(fd)

hog_features_test_age = np.array(hog_features_test_age)
predicted = clf.predict(hog_features_test_age)
print(f"Ages Predicted: {predicted}")

cap.release()
