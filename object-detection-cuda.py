import cv2
import numpy as np

import os

import time


def get_output_layers(net):
    layer_names = net.getLayerNames()
    try:
        output_layers = [layer_names[i - 1] for i in net.getUnconnectedOutLayers()]
    except:
        output_layers = [layer_names[i[0] - 1] for i in net.getUnconnectedOutLayers()]
    return output_layers


def draw_prediction(img, classes, class_id, confidence, x, y, x_plus_w, y_plus_h):
    label = f"{str(classes[class_id])} + {str(int(confidence*100))}%"
    color = COLORS[class_id]
    cv2.rectangle(img, (x, y), (x_plus_w, y_plus_h), color, 2)
    cv2.putText(img, label, (x - 10, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)


def show_video(cap, net, classes):
    conf_threshold = 0.5
    nms_threshold = 0.4

    scale = 0.00392  # Increasing will decrease how accurate it is

    prev_frame_time = 0
    new_frame_time = 0

    while cap.isOpened():
        ret, frame = cap.read()
        if ret:
            new_frame_time = time.time()
            fps = 1 / (new_frame_time - prev_frame_time)
            prev_frame_time = new_frame_time

            fps = str(int(fps))

            statistics_font = cv2.FONT_HERSHEY_SIMPLEX

            width = frame.shape[1]
            height = frame.shape[0]

            blob = cv2.dnn.blobFromImage(
                frame, scale, (416, 416), (0, 0, 0), True, crop=False
            )

            net.setInput(blob)

            outs = net.forward(get_output_layers(net))

            class_ids = []
            confidences = []
            boxes = []

            detected_in_frame = {}

            for detection_class in classes:
                detected_in_frame[detection_class] = []

            for out in outs:
                for detection in out:
                    scores = detection[5:]
                    class_id = np.argmax(scores)
                    confidence = scores[class_id]

                    if confidence > 0.5:
                        center_x = int(detection[0] * width)
                        center_y = int(detection[1] * height)
                        w = int(detection[2] * width)
                        h = int(detection[3] * height)
                        x = center_x - w / 2
                        y = center_y - h / 2
                        class_ids.append(class_id)
                        confidences.append(float(confidence))
                        boxes.append([x, y, w, h])

            indices = cv2.dnn.NMSBoxes(
                boxes, confidences, conf_threshold, nms_threshold
            )

            for i in indices:
                if type(i) == list:
                    i = i[0]
                box = boxes[i]

                x, y, w, h = box

                detected_in_frame[classes[class_ids[i]]].append(
                    [
                        [round(x), round(y), round(x + w), round(y + h)],
                        confidences[i],
                    ]  # Key value format: [[coordinates], accuracy]
                )

                draw_prediction(
                    frame,
                    classes,
                    class_ids[i],
                    confidences[i],
                    round(x),
                    round(y),
                    round(x + w),
                    round(y + h),
                )

            statistics = f"FPS: {fps}; People: {str(len(detected_in_frame['person']))}; Vehicles: {str(len(detected_in_frame['car']))}"

            cv2.putText(
                frame,
                statistics,
                (5, 15),
                statistics_font,
                0.5,
                (20, 35, 200),
                2,
                cv2.LINE_AA,
            )

            cv2.imshow("object detection", frame)

            if cv2.waitKey(1) & 0xFF == ord("q"):
                return
        else:
            return


cv2.cuda.setDevice(0)
cv2.ocl.setUseOpenCL(False)

weights = "./settings/yolov3.weights"
config = "./settings/config.cfg"

classes = None
with open("./settings/classes.txt", "r") as f:
    classes = [line.strip() for line in f.readlines()]

COLORS = [
    [20, 35, 200] for i in range(80)
]  # Changing color codes will result different colors. Color format: BGR

net = cv2.dnn.readNetFromDarknet(config, weights)
net.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
net.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA)

dir_path = "./videos"

for file_path in os.listdir(dir_path):
    print(os.path.join(dir_path, file_path))
    cap = cv2.VideoCapture(os.path.join(dir_path, file_path))

    if not cap.isOpened():
        print("Error opening video")
        continue

    show_video(cap, net, classes)
