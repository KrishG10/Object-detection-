#!/usr/bin/env python
# coding: utf-8

# In[1]:


import cv2

# Load the pre-trained MobileNet SSD model
model = cv2.dnn.readNetFromCaffe('C:\\Users\\Harshal\\Downloads\\MobileNetSSD_deploy.prototxt.txt', 'C:\\Users\\Harshal\\Downloads\\MobileNetSSD_deploy.caffemodel')
# Define the classes of objects we want to detect (in this case, people)
classes = ["background", "aeroplane", "bicycle", "bird", "boat",
           "bottle", "bus", "car", "cat", "chair", "cow", "diningtable",
           "dog", "horse", "motorbike", "person", "pottedplant", "sheep",
           "sofa", "train", "tvmonitor"]

# Set the minimum confidence level for a detection to be shown
confidence_threshold = 0.5

# Initialize the video capture device (0 is the default built-in camera)
cap = cv2.VideoCapture('C:\\Users\\Harshal\\OneDrive\\Desktop\\model\\mixkit-times-square-during-a-rainy-night-4332-medium.mp4')

# Loop through the frames of the video feed
while True:
    # Read a frame from the video feed
    ret, frame = cap.read()
    if not ret:
        break  # exit the loop if there are no more frames to read

    # Resize the frame to a width of 300 pixels (the input size of the model)
    frame_resized = cv2.resize(frame, (300, 300))

    # Create a blob (binary large object) from the resized frame for input to the model
    blob = cv2.dnn.blobFromImage(frame_resized, 0.007843, (300, 300), 127.5)

    # Set the input for the model to the blob and perform a forward pass
    model.setInput(blob)
    detections = model.forward()

    # Loop through the detections and draw a rectangle around any objects with high confidence
    for i in range(detections.shape[2]):
        confidence = detections[0, 0, i, 2]
        if confidence > confidence_threshold:
            class_id = int(detections[0, 0, i, 1])
            class_label = classes[class_id]
            x_left_bottom = int(detections[0, 0, i, 3] * frame.shape[1])
            y_left_bottom = int(detections[0, 0, i, 4] * frame.shape[0])
            x_right_top = int(detections[0, 0, i, 5] * frame.shape[1])
            y_right_top = int(detections[0, 0, i, 6] * frame.shape[0])
            cv2.rectangle(frame, (x_left_bottom, y_left_bottom), (x_right_top, y_right_top), (0, 255, 0), 2)
            cv2.putText(frame, class_label + " {:.2f}%".format(confidence * 100), (x_left_bottom, y_left_bottom-5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    # Show the frame with detections
    cv2.imshow('Object Detection', frame)

    # Check if the user has pressed the 'q' key to quit the program
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

#


# Release the video capture device and close the window
cap.release()
cv2.destroyAllWindows()


# In[ ]:




