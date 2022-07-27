import PySimpleGUI as sg
import cv2
import dlib
import numpy as np

import face_recognition
import matplotlib.pyplot as plt
from skimage.feature import hog
from skimage import data, exposure

from sklearn import svm
from skimage.transform import resize

font = cv2.FONT_HERSHEY_SIMPLEX

ppc = 16

"""
Demo program that displays a webcam using OpenCV and applies some very basic image functions
- functions from top to bottom -
none:       no processing
threshold:  simple b/w-threshold on the luma channel, slider sets the threshold value
canny:      edge finding with canny, sliders set the two threshold values for the function => edge sensitivity
blur:       simple Gaussian blur, slider sets the sigma, i.e. the amount of blur smear
hue:        moves the image hue values by the amount selected on the slider
enhance:    applies local contrast enhancement on the luma channel to make the image fancier - slider controls fanciness.
"""

detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor('shape_predictor_68_face_landmarks.dat')

cascPath = "/usr/local/lib/python3.7/site-packages/cv2/data/haarcascade_frontalface_default.xml"
eyePath = "/usr/local/lib/python3.7/site-packages/cv2/data/haarcascade_eye.xml"
smilePath = "/usr/local/lib/python3.7/site-packages/cv2/data/haarcascade_smile.xml"

faceCascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
eyeCascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')
smileCascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_smile.xml')

# ======== SVM Classifier ========
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

# gender training
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

clf = svm.SVC()
hog_features = np.array(hog_features)
gender_model = clf.fit(hog_features, categories)

# age training
for k in range(1, 401):
    img = cv2.imread(f"train/im{k}.jpg")
    if k <= 50:
        ages = np.append(ages, 1)  # man, anak2
    elif 50 < k <= 100:
        ages = np.append(ages, 2)  # man, remaja
    elif 100 < k <= 150:
        ages = np.append(ages, 3)  # man, dewasa
    elif 150 < k <= 200:
        ages = np.append(ages, 4)  # man, tua
    elif 200 < k <= 250:
        ages = np.append(ages, 5)  # woman, anak
    elif 250 < k <= 300:
        ages = np.append(ages, 6)  # woman, remaja
    elif 3000 < k <= 350:
        ages = np.append(ages, 7)  # woman, dewasa
    else:
        ages = np.append(ages, 8)  # woman, tua

    img_resized = resize(img, (150, 150))
    # print(img_resized)
    img_parsed = np.float32(img_resized)
    greyscale_img = cv2.cvtColor(img_parsed, cv2.COLOR_BGR2GRAY)
    fd, hog_image = hog(greyscale_img, orientations=8, pixels_per_cell=(ppc, ppc),
                    cells_per_block=(4, 4), block_norm='L2', visualize=True)
    hog_images_age.append(hog_image)
    hog_features_age.append(fd)

clf = svm.SVC()
hog_features_age = np.array(hog_features_age)
age_model = clf.fit(hog_features_age, ages)

sg.theme('LightGreen')

# define the window layout
layout = [
    [sg.Text('OpenCV Demo', size=(60, 1), justification='center')],
      [sg.Image(filename='', key='-IMAGE-')],
        [sg.Button('Capture', size=(10, 1))],
        [sg.Text('Gender', size=(40, 1), key="-GENDER-")],
        [sg.Text('Age', size=(40, 1), key="-AGE-")],
      [sg.Button('Exit', size=(10, 1))]
]

# create the window and show it without the plot
window = sg.Window('OpenCV Integration', layout, location=(800, 400))

cap = cv2.VideoCapture(0)

while True:
    event, values = window.read(timeout=20)
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

    if event == 'Capture':
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

        # creating hog feature
        gray_img = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        fd, hog_image = hog(gray_img, orientations=8, pixels_per_cell=(ppc, ppc), cells_per_block=(4, 4),
                                visualize=True)
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

        # gender predict
        img_test = cv2.imread("saved_img.jpg")
        img_resized = resize(img_test, (150, 150))
        img_parsed = np.float32(img_resized)
        greyscale_img = cv2.cvtColor(img_parsed, cv2.COLOR_BGR2GRAY)
        fd, hog_image = hog(greyscale_img, orientations=8, pixels_per_cell=(ppc, ppc),
                                cells_per_block=(4, 4), block_norm='L2', visualize=True)

        hog_features_test = np.array(hog_image)
        predicted = clf.predict([fd])
        #print(f"Gender Predicted: {predicted[0]}")
        if predicted[0] <= 4.0:
            window["-GENDER-"].update("Gender : Male")
            print("Gender Predicted: Male")
        else :
            window["-GENDER-"].update("Gender : Female")
            print("Gender Predicted: Female")

        # age predict
        img_test = cv2.imread("saved_img.jpg")
        img_resized = resize(img_test, (150, 150))
        img_parsed = np.float32(img_resized)
        greyscale_img = cv2.cvtColor(img_parsed, cv2.COLOR_BGR2GRAY)
        fd, hog_image = hog(greyscale_img, orientations=8, pixels_per_cell=(ppc, ppc),
                                cells_per_block=(4, 4), block_norm='L2', visualize=True)

        hog_features_test_age = np.array(hog_image)
        predicted = clf.predict([fd])
        print(f"Ages Predicted: {predicted}")
        if predicted[0] == 1.0:
            window["-AGE-"].update("Age : Male Child")
            print("Ages Predicted: Male Child")
        elif predicted[0] == 2.0:
            window["-AGE-"].update("Age : Male Adult")
            print("Ages Predicted: Male Adult")
        elif predicted[0] == 3.0:
            window["-AGE-"].update("Age : Male Mature")
            print("Ages Predicted: Male Mature")
        elif predicted[0] == 4.0:
            window["-AGE-"].update("Age : Male Old")
            print("Ages Predicted: Male Old")
        elif predicted[0] == 5.0:
            window["-AGE-"].update("Age : Female Child")
            print("Ages Predicted: Female Child")
        elif predicted[0] == 6.0:
            window["-AGE-"].update("Age : Female Adult")
            print("Ages Predicted: Female Adult")
        elif predicted[0] == 7.0:
            window["-AGE-"].update("Age : Female Mature")
            print("Ages Predicted: Female Mature")
        else:
            window["-AGE-"].update("Age : Female Old")
            print("Ages Predicted: Female Old")
        break

    if event == 'Exit' or event == sg.WIN_CLOSED:
        break

    ret, frame = cap.read()
    frame = image

    imgbytes = cv2.imencode('.png', frame)[1].tobytes()
    window['-IMAGE-'].update(data=imgbytes)

window.close()