import cv2
import PySimpleGUI as sg
import os.path

from skimage.feature import hog
import numpy as np
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

#train gender
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

#TRAIN GENDER
for i in range(1, 401):
    img = cv2.imread(f"train/im{i}.jpg")
    if i <= 201:
        categories = np.append(categories, 1)
    else:
        categories = np.append(categories, 2)

    img_resized = resize(img, (150, 150))
    #print(img_resized)
    img_parsed = np.float32(img_resized)
    greyscale_img = cv2.cvtColor(img_parsed, cv2.COLOR_BGR2GRAY)
    fd, hog_image = hog(greyscale_img, orientations=8, pixels_per_cell=(ppc, ppc),
                        cells_per_block=(4, 4), block_norm='L2', visualize=True)
    hog_images.append(hog_image)
    hog_features.append(fd)

clf = svm.SVC()
#filename = 'model_gender.sav'
hog_features = np.array(hog_features)
gender_model = clf.fit(hog_features, categories)

#TRAIN AGE
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
    img_parsed = np.float32(img_resized)
    greyscale_img = cv2.cvtColor(img_parsed, cv2.COLOR_BGR2GRAY)
    fd, hog_image = hog(greyscale_img, orientations=8, pixels_per_cell=(ppc, ppc),
                        cells_per_block=(4, 4), block_norm='L2', visualize=True)
    hog_images_age.append(hog_image)
    hog_features_age.append(fd)

clf = svm.SVC()
#filename = 'model_age.sav'
hog_features_age = np.array(hog_features_age)
age_model = clf.fit(hog_features_age, ages)

# First the window layout in 2 columns

def buttonhandler():
    print("hello")

file_list_column = [
    [
        sg.Text("Image Folder"),
        sg.In(size=(25, 1), enable_events=True, key="-FOLDER-"),
        sg.FolderBrowse(),
        sg.Button('Exit', size=(7, 1)),
    ],
    [
        sg.Listbox(
            values=[], enable_events=True, size=(40, 20), key="-FILE LIST-"
        ),
        sg.Button('Gender', size=(10, 1)),
        sg.Button('Age', size=(10, 1)),
    ],
]

# For now will only show the name of the file that was chosen
image_viewer_column = [
    [sg.Text("Choose an image from list on left:")],
    [sg.Text(size=(40, 1), key="-TOUT-")],
    [sg.Image(key="-IMAGE-")],
    [sg.Text(size=(40, 1), key="-GENDER-")],
    [sg.Text(size=(40, 1), key="-AGE-")],
]

# ----- Full layout -----
layout = [
    [
        sg.Column(file_list_column),
        sg.VSeperator(),
        sg.Column(image_viewer_column),
    ]
]

window = sg.Window("Image Viewer", layout)

# Run the Event Loop
while True:
    event, values = window.read()
    fileChossen=None
    if event == "Exit" or event == sg.WIN_CLOSED:
        break
    # Folder name was filled in, make a list of files in the folder
    if event == "-FOLDER-":
        folder = values["-FOLDER-"]
        try:
            # Get list of files in folder
            file_list = os.listdir(folder)
        except:
            file_list = []

        fnames = [
            f
            for f in file_list
            if os.path.isfile(os.path.join(folder, f))
            and f.lower().endswith((".png", ".jpg"))
        ]
        window["-FILE LIST-"].update(fnames)

    elif event == "-FILE LIST-":  # A file was chosen from the listbox
        try:
            filename = os.path.join(
                values["-FOLDER-"], values["-FILE LIST-"][0]
            )
            window["-TOUT-"].update(filename)
            fileChossen = filename
            window["-IMAGE-"].update(filename=filename)
        except:
            pass
    elif event == "Gender" :
        print("hello gender")
        img_test = cv2.imread(filename)
        print(filename)
        img_resized = resize(img_test, (150, 150))
        img_parsed = np.float32(img_resized)
        greyscale_img = cv2.cvtColor(img_parsed, cv2.COLOR_BGR2GRAY)
        fd, hog_image = hog(greyscale_img, orientations=8, pixels_per_cell=(ppc, ppc),
                            cells_per_block=(4, 4), block_norm='L2', visualize=True)

        hog_features_test = np.array(hog_image)
        predicted = clf.predict([fd])
        print(f"Gender Predicted: {predicted[0]}")
        if predicted[0] <= 4.0:
            window["-GENDER-"].update("Gender : Male")
        else :
            window["-GENDER-"].update("Gender : Female")

    elif event == "Age" :
        print("hello age")
        img_test = cv2.imread(filename)
        print(filename)
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
        elif predicted[0] == 2.0:
            window["-AGE-"].update("Age : Male Adult")
        elif predicted[0] == 3.0:
            window["-AGE-"].update("Age : Male Mature")
        elif predicted[0] == 4.0:
            window["-AGE-"].update("Age : Male Old")
        elif predicted[0] == 5.0:
            window["-AGE-"].update("Age : Female Child")
        elif predicted[0] == 6.0:
            window["-AGE-"].update("Age : Female Adult")
        elif predicted[0] == 7.0:
            window["-AGE-"].update("Age : Female Mature")
        else:
            window["-AGE-"].update("Age : Female Old")

window.close()