import cv2

# load a pretrained data on face frontald from opencv (haarcascade algo)
trained_face_data = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

#choose an image to detect faces in
#img= cv2.imread('facetest.jpg') #call the image read function you are reading the image in a 2d array
webcam = cv2.VideoCapture(0) # replace the image with the webcam

#iterate over all the frames forever
while True:
    #read succesful frames - this is a boolean true or false and frame will read the webcam
    successful_frame_read, frame=webcam.read()

#convert the image to a black and white version/grayscale - you can change the whole image based on rgb etc
    blackwhite_img = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)

#detect faces 
# detect multiscale is a function that looks at the overal composition of an object not only faces 
# small or big/the relationship to the eyes nose etc
    face_coordinates = trained_face_data.detectMultiScale(blackwhite_img)

#these are the coordinates of the square 
    print(face_coordinates)

#draw rectangles around the faces for every face detected
    for (x,y,w,h) in face_coordinates: #for each item in the array it will draw a rectangle
        cv2.rectangle(frame,(x,y),(x+w, y+h), (0,255,0),2) # we are establishing a rectangle on the image var the parameter in the middle are coordinates we get from the analyser and the last parameter is the color green the last parameter is the tickness of the line 

    cv2.imshow('facedetection testing', frame) #this function bascially dislplays the requeted image 2 parameters 1 is the name the other is the var you want to display

# wait key is basically needed to wait for us to run view the code otherwise it will open and close the image insantly
#press any key to continue the code and close the image

    key = cv2.waitKey(1) ##wait key allows the program to wait to later then continue the program we put 1 in because it will wait one milli second
     #then we capture the key in a var and we will compare the key and say if the key is a particular key like q then we will break the loop

    if key==81 or key==113:
         break


    

print("Code completed") 