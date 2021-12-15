import cv2 as cv
"""
// This is to detect faces in a image and getting its coordinates

# Load the cascade
face_cascade = cv.CascadeClassifier('haarcascade_frontalface_default.xml')
# Read the input image
img = cv.imread('resources/groupPhoto.jpeg')
# Convert into grayscale
gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
# Detect faces
faces = face_cascade.detectMultiScale(gray, 1.1, 4)
print(faces)


# Draw rectangle around the faces
for (x, y, w, h) in faces:
    cv.rectangle(img, (x, y), (x+w, y+h), (0, 0, 0), 2)
    # croppedImg = img[y:y+h, x:x+w]
    # img[y:y + h, x:x + w] = cv.blur(img[y:y + h, x:x + w], (23, 23))
# Display the output

cv.imshow('img', img)
filename = "saved.jpeg"
cv.imwrite(filename, img)
cv.waitKey(0)
"""
# The below code does the same thing as above but with every frame captured.
# As a video is nothing but a bunch of several images

cap = cv.VideoCapture(0)
# setting width and height of window
cap.set(cv.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv.CAP_PROP_FRAME_HEIGHT, 480)

# loading cascade
face_cascade = cv.CascadeClassifier('haarcascade_frontalface_default.xml')

while True:
    flag, frame = cap.read()  # capture frame by frame
    #  if frame is read correctly flag is true
    if not flag:
        print("Can't receive frame !")
        break

    gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.1, 4)
    for (x, y, w, h) in faces:
        frame[y:y + h, x:x + w] = cv.blur(frame[y:y + h, x:x + w], (23, 23)) # GaussianBlur can also be used
        # blurframe = cv.GaussianBlur(frame, (13, 13), 0)
        # edgeFrame = cv.Canny(frame, 100, 200)
        cv.imshow("webCam", frame)
        # cv.imshow("webCam-Blur", blurframe) # displays the frame
        # cv.imshow("webCam-canny", edgeFrame)
    if cv.waitKey(1) == ord('q'):  # press 'q' to close the window
        break
# When all's done, release the capture
cap.release()
cv.destroyAllWindows()
