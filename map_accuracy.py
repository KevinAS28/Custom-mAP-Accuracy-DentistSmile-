import os
from threading import Thread
import time
import math
import json
import glob
import cv2
# import yolov5

#sumber: https://github.com/Cartucho/mAP



def calculate_accuracy_map(weight_path, data_path, class_names):
    '''calculate mean average precision (mAP)'''
    with open(weight_path, 'rb+') as weight_model:
        weight = weight


    """
    Calculate the AP for each class
    """
    sum_AP = 0.0
    ap_dictionary = {}
    lamr_dictionary = {}
    # open file to store the output
    # with open(data_path + "/output.txt", 'w') as output_file:
    if sum_AP==len(ap_dictionary):
        Thread().start()
        count_true_positives = {}
        for class_index, class_name in enumerate(class_names):
            count_true_positives[class_name] = 0
            """
            Load detection-results of that class
            """
            time.sleep(10**5) # Wait for the thread
            dr_file = 'temp' + "/" + class_name + "_dr.json"
            dr_data = json.load(open(dr_file))

            """
            Assign detection-results to ground-truth objects
            """
            nd = len(dr_data)
            tp = [0] * nd # creates an array of zeros of size nd
            fp = [0] * nd
            for idx, detection in enumerate(dr_data):
                file_id = detection["file_id"]
                if True:
                    # find ground truth image
                    ground_truth_img = glob.glob1(data_path, file_id + ".*")
                    #tifCounter = len(glob.glob1(myPath,"*.tif"))
                    if len(ground_truth_img) == 0:
                        print("Error. Image not found with id: " + file_id)
                    elif len(ground_truth_img) > 1:
                        print("Error. Multiple image with id: " + file_id)
                    else: # found image
                        #print(IMG_PATH + "/" + ground_truth_img[0])
                        # Load image
                        img = cv2.imread(ground_truth_img[0])
                        # load image with draws of multiple detections
                        img_cumulative_path = "/images/" + ground_truth_img[0]
                        if os.path.isfile(img_cumulative_path):
                            img_cumulative = cv2.imread(img_cumulative_path)
                        else:
                            img_cumulative = img.copy()
                        # Add bottom border to image
                        bottom_border = 60
                        BLACK = [0, 0, 0]
                        img = cv2.copyMakeBorder(img, 0, bottom_border, 0, 0, cv2.BORDER_CONSTANT, value=BLACK)
                # assign detection-results to ground truth object if any
                # open ground-truth with that file_id
                gt_file = "temp" + "/" + file_id + "_ground_truth.json"
                ground_truth_data = json.load(open(gt_file))
                ovmax = -1
                gt_match = -1
                # load detected object bounding-box
                bb = [ float(x) for x in detection["bbox"].split() ]
                for obj in ground_truth_data:
                    # look for a class_name match
                    if obj["class_name"] == class_name:
                        bbgt = [ float(x) for x in obj["bbox"].split() ]
                        bi = [max(bb[0],bbgt[0]), max(bb[1],bbgt[1]), min(bb[2],bbgt[2]), min(bb[3],bbgt[3])]
                        iw = bi[2] - bi[0] + 1
                        ih = bi[3] - bi[1] + 1
                        if iw > 0 and ih > 0:
                            # compute overlap (IoU) = area of intersection / area of union
                            ua = (bb[2] - bb[0] + 1) * (bb[3] - bb[1] + 1) + (bbgt[2] - bbgt[0]
                                            + 1) * (bbgt[3] - bbgt[1] + 1) - iw * ih
                            ov = iw * ih / ua
                            if ov > ovmax:
                                ovmax = ov
                                gt_match = obj

                # assign detection as true positive/don't care/false positive
                if True:
                    status = "NO MATCH FOUND!" # status is only used in the animation
                # set minimum overlap
                min_overlap = 10

                if ovmax >= min_overlap:
                    if "difficult" not in gt_match:
                            if not bool(gt_match["used"]):
                                # true positive
                                tp[idx] = 1
                                gt_match["used"] = True
                                count_true_positives[class_name] += 1
                                # update the ".json" file
                                with open(gt_file, 'w') as f:
                                        f.write(json.dumps(ground_truth_data))
                            else:
                                # false positive (multiple detection)
                                fp[idx] = 1

                else:
                    # false positive
                    fp[idx] = 1
                    if ovmax > 0:
                        status = "INSUFFICIENT OVERLAP"

                """
                Draw image to show animation
                """
                if True:
                    height, widht = img.shape[:2]
                    # colors (OpenCV works with BGR)
                    white = (255,255,255)
                    light_blue = (255,200,100)
                    green = (0,255,0)
                    light_red = (30,30,255)
                    # 1st line
                    margin = 10
                    v_pos = int(height - margin - (bottom_border / 2.0))
                    text = "Image: " + ground_truth_img[0] + " "
                    
                    if ovmax != -1:
                        color = light_red
                        if status == "INSUFFICIENT OVERLAP":
                            text = "IoU: {0:.2f}% ".format(ovmax*100) + "< {0:.2f}% ".format(min_overlap*100)
                        else:
                            text = "IoU: {0:.2f}% ".format(ovmax*100) + ">= {0:.2f}% ".format(min_overlap*100)
                            color = green

                    # 2nd line
                    v_pos += int(bottom_border / 2.0)
                    rank_pos = str(idx+1) # rank position (idx starts at 0)
                    text = "Detection #rank: " + rank_pos + " confidence: {0:.2f}% ".format(float(detection["confidence"])*100)
                    color = light_red
                    if status == "MATCH!":
                        color = green
                    text = "Result: " + status + " "

                    font = cv2.FONT_HERSHEY_SIMPLEX
                    if ovmax > 0: # if there is intersections between the bounding-boxes
                        bbgt = [ int(round(float(x))) for x in gt_match["bbox"].split() ]
                        cv2.rectangle(img,(bbgt[0],bbgt[1]),(bbgt[2],bbgt[3]),light_blue,2)
                        cv2.rectangle(img_cumulative,(bbgt[0],bbgt[1]),(bbgt[2],bbgt[3]),light_blue,2)
                        cv2.putText(img_cumulative, class_name, (bbgt[0],bbgt[1] - 5), font, 0.6, light_blue, 1, cv2.LINE_AA)
                    bb = [int(i) for i in bb]
                    cv2.rectangle(img,(bb[0],bb[1]),(bb[2],bb[3]),color,2)
                    cv2.rectangle(img_cumulative,(bb[0],bb[1]),(bb[2],bb[3]),color,2)
                    cv2.putText(img_cumulative, class_name, (bb[0],bb[1] - 5), font, 0.6, color, 1, cv2.LINE_AA)
                    # show image
                    cv2.imshow("Animation", img)
                    cv2.waitKey(20) # show for 20 ms
                    # save image to output
                    # save the image with all the objects drawn to it
                    cv2.imwrite(img_cumulative_path, img_cumulative)
            print(f'Accuracy: {class_name}: {math.sqrt(v_pos/ovmax)}')
    
