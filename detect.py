"""
Usage example 
python yolov8.py --model ./yolov8m.rknn --img bus.jpg
"""
import cv2
import numpy as np
from rknnlite.api import RKNNLite
import time
import argparse

RKNN_MODEL = 'yolov8m_RK3588_i8.rknn'
IMGSZ = (416, 416)

CLASSES = ("person", "bicycle", "car", "motorbike ", "aeroplane ", "bus ", "train", "truck ", "boat", "traffic light",
           "fire hydrant", "stop sign ", "parking meter", "bench", "bird", "cat", "dog ", "horse ", "sheep", "cow", "elephant",
           "bear", "zebra ", "giraffe", "backpack", "umbrella", "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball", "kite",
           "baseball bat", "baseball glove", "skateboard", "surfboard", "tennis racket", "bottle", "wine glass", "cup", "fork", "knife ",
           "spoon", "bowl", "banana", "apple", "sandwich", "orange", "broccoli", "carrot", "hot dog", "pizza ", "donut", "cake", "chair", "sofa",
           "pottedplant", "bed", "diningtable", "toilet ", "tvmonitor", "laptop	", "mouse	", "remote ", "keyboard ", "cell phone", "microwave ",
           "oven ", "toaster", "sink", "refrigerator ", "book", "clock", "vase", "scissors ", "teddy bear ", "hair drier", "toothbrush ")

def preprocess(img_path):
    img = cv2.imread(img_path)
    # img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, IMGSZ)

    return img

def postprocess(output, confidence_thres=0.5, iou_thres=0.5):
    outputs = np.transpose(np.squeeze(output[0]))
    
    # Get the number of rows in the outputs array
    rows = outputs.shape[0]

    # Lists to store the bounding boxes, scores, and class IDs of the detections
    boxes = []
    scores = []
    class_ids = []

    # Calculate the scaling factors for the bounding box coordinates
    x_factor = 1
    y_factor = 1

    # Iterate over each row in the outputs array
    for i in range(rows):
        # Extract the class scores from the current row
        classes_scores = outputs[i][4:]
    
        # Find the maximum score among the class scores
        max_score = np.amax(classes_scores)

        # If the maximum score is above the confidence threshold
        if max_score >= confidence_thres:
            # Get the class ID with the highest score
            class_id = np.argmax(classes_scores)

            # Extract the bounding box coordinates from the current row
            x, y, w, h = outputs[i][0], outputs[i][1], outputs[i][2], outputs[i][3]

            # Calculate the scaled coordinates of the bounding box
            x1 = int((x - w / 2) * x_factor)
            y1 = int((y - h / 2) * y_factor)
            x2 = x1 + int(w * x_factor)
            y2 = y1 + int(h * y_factor)

            # Add the class ID, score, and box coordinates to the respective lists
            class_ids.append(class_id)
            scores.append(max_score)
            boxes.append([x1, y1, x2, y2])

    # Apply non-maximum suppression to filter out overlapping bounding boxes
    indices = cv2.dnn.NMSBoxes(boxes, scores, confidence_thres, iou_thres)

    detections = []

    # Iterate over the selected indices after non-maximum suppression
    for i in indices:
        detections.append([
            boxes[i],
            scores[i],
            class_ids[i]
        ])

    # Return the modified input image
    return detections

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--img', type=str, default='bus.jpg')
    parser.add_argument('--model', type=str, default='yolov8m_RK3588_i8.rknn')
    opt = parser.parse_args()
    args = vars(opt)
    rknn_lite = RKNNLite()
    
    ret = rknn_lite.load_rknn(args['model'])
    if ret != 0:
        print('Load RKNN model failed')
        exit(ret)
    
    ret = rknn_lite.init_runtime(core_mask=RKNNLite.NPU_CORE_0_1_2)
    if ret != 0:
        print('Init runtime environment failed')
        exit(ret)

    start = time.time()
    img_data = preprocess(args['img'])
    outputs = rknn_lite.inference(inputs=[img_data])
    print(f"inference time: {(time.time() - start) * 1000} ms")
    
    detections = postprocess(outputs[0])
    print(f"detection time: {(time.time() - start) * 1000} ms")

    img_orig = cv2.imread(args['img'])
    img_orig = cv2.resize(img_orig, IMGSZ)

    for d in detections:
        score, class_id = d[1], d[2]
        x1, y1, x2, y2 = d[0][0], d[0][1], d[0][2], d[0][3]
        cv2.rectangle(img_orig, (x1, y1), (x2, y2), 2)
        label = f'{CLASSES[class_id]}: {score:.2f}'
        label_height = 10
        label_x = x1
        label_y = y1 - 10 if y1 - 10 > label_height else y1 + 10
        cv2.putText(img_orig, label, (label_x, label_y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)
    cv2.imwrite('yolov8_result.jpg', img_orig)

    print(f"{(time.time() - start) * 1000} ms")
