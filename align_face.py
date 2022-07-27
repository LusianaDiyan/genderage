import cv2
import numpy as np

# Creating face_cascade and eye_cascade objects
faceCascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
eyeCascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')
#smileCascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_smile.xml')

while True:
    # Read the image file
    image = cv2.imread('saved_img.jpg');
    # Convert the image color to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # Creating variable faces
    faces = faceCascade.detectMultiScale(gray, 1.1, 4)
    # Defining and drawing the rectangle around the face
    for (x, y, w, h) in faces:
        cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 3)

    # Creating two regions of interest
    roi_gray = gray[y:(y + h), x:(x + w)]
    roi_color = image[y:(y + h), x:(x + w)]

    # created a for loop to segment one eye from another.
    # stored coordinates of the first and second eye
    # in eye_1 variable and eye_2  variables, respectively.

    # Creating variable eyes
    eyes = eyeCascade.detectMultiScale(roi_gray, 1.1, 4)
    index = 0
    # Creating for loop in order to divide one eye from another
    for (ex, ey, ew, eh) in eyes:
        if index == 0:
            eye_1 = (ex, ey, ew, eh)
        elif index == 1:
            eye_2 = (ex, ey, ew, eh)
        # Drawing rectangles around the eyes
        cv2.rectangle(roi_color, (ex, ey), (ex + ew, ey + eh), (0, 0, 255), 3)
        index = index + 1

    # differentiating between left_eye and right_eye
    # by looking at the figure that smaller eye will be our left_eye.
    if eye_1[0] < eye_2[0]:
        left_eye = eye_1
        right_eye = eye_2
    else:
        left_eye = eye_2
        right_eye = eye_1

    # Note that index 0 refers to x coordinate, index 1 refers to y coordinate,
    # index 2 refers to rectangle width, and finally index 3 refers to rectangle height.

    # Calculating coordinates of a central points of the rectangles
    left_eye_center = (int(left_eye[0] + (left_eye[2] / 2)), int(left_eye[1] + (left_eye[3] / 2)))
    left_eye_x = left_eye_center[0]
    left_eye_y = left_eye_center[1]

    right_eye_center = (int(right_eye[0] + (right_eye[2] / 2)), int(right_eye[1] + (right_eye[3] / 2)))
    right_eye_x = right_eye_center[0]
    right_eye_y = right_eye_center[1]

    cv2.circle(roi_color, left_eye_center, 5, (255, 0, 0), -1)
    cv2.circle(roi_color, right_eye_center, 5, (255, 0, 0), -1)
    cv2.line(roi_color, right_eye_center, left_eye_center, (0, 200, 200), 3)

    # draw a horizontal line and calculate the angle between that line
    # and the line that connects two central points of the eyes.
    # the goal is to rotate the image based on this angle

    if left_eye_y > right_eye_y:
        A = (right_eye_x, left_eye_y)
        # Integer -1 indicates that the image will rotate in the clockwise direction
        direction = -1
    else:
        A = (left_eye_x, right_eye_y)
        # Integer 1 indicates that image will rotate in the counter clockwise
        # direction
        direction = 1

    cv2.circle(roi_color, A, 5, (255, 0, 0), -1)

    cv2.line(roi_color, right_eye_center, left_eye_center, (0, 200, 200), 3)
    cv2.line(roi_color, left_eye_center, A, (0, 200, 200), 3)
    cv2.line(roi_color, right_eye_center, A, (0, 200, 200), 3)

    # To calculate the angle, need to find the length of two legs of a right triangle.
    # Then find the required angle
    delta_x = right_eye_x - left_eye_x
    delta_y = right_eye_y - left_eye_y
    angle = np.arctan(delta_y / delta_x)
    angle = (angle * 180) / np.pi

    # to convert the result in degree, we need to multiply our angle θ with 180 and then divide it by π.
    # Width and height of the image
    h, w = image.shape[:2]
    # Calculating a center point of the image
    # Integer division "//"" ensures that we receive whole numbers
    center = (w // 2, h // 2)
    # Defining a matrix M and calling
    # cv2.getRotationMatrix2D method
    M = cv2.getRotationMatrix2D(center, (angle), 1.0)
    # Applying the rotation to our image using the
    # cv2.warpAffine method
    rotated = cv2.warpAffine(image, M, (w, h))

    # Display the image
    cv2.imshow('Face Detection', rotated)

    # Press the escape button to terminate the code
    if cv2.waitKey(10) == 27:
        break

cv2.destroyAllWindows()