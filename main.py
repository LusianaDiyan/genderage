import cv2
from skimage.feature import hog
import numpy as np
from sklearn import svm
from skimage.transform import resize
import pickle

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

for i in range(1, 401):
    img = cv2.imread(f"train/im{i}.jpg")
    if i <= 201:
        categories = np.append(categories, 1)
    else:
        categories = np.append(categories, 2)

    img_resized = resize(img, (150, 150))
    #print(img_resized)
    #print("training image")
    img_parsed = np.float32(img_resized)
    greyscale_img = cv2.cvtColor(img_parsed, cv2.COLOR_BGR2GRAY)
    fd, hog_image = hog(greyscale_img, orientations=8, pixels_per_cell=(ppc, ppc),
                        cells_per_block=(4, 4), block_norm='L2', visualize=True)
    hog_images.append(hog_image)
    hog_features.append(fd)

clf = svm.SVC()
#filename = 'model_gender.sav'
hog_features = np.array(hog_features)

# print(hog_features.shape)
# print(categories)

gender_model = clf.fit(hog_features, categories)
#pickle.dump(clf, open(filename, 'wb'))

for j in range(1, 11):
    img_test = cv2.imread(f"test/im{j}.jpg")
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
    #print("training image")
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
#pickle.dump(clf, open(filename, 'wb'))

for l in range(1, 11):
    img_test = cv2.imread(f"test/im{l}.jpg")
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
