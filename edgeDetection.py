'''
N96091172 HSING-YUN, TSAI
Multimedia HW1 (shot change boundary detection)
'''
import os
import cv2
import datetime
import numpy as np
import matplotlib.pyplot as plt

def get_EuclideanDistance(x, y):
    myX = np.array(x)
    myY = np.array(y)
    return np.sqrt(np.sum(( myX - myY ) * ( myX - myY )))

# returns the edge-change-ratio
# the dilate_rate parameter controls the distance of the pixels between the frame
def ECR(edge_frame, edge_prevframe):
    # 傳入 x, y 參數，如果 y 是 0 則回傳 0；
    pixelRatio = lambda x, y : 0 if y == 0 else x / y

    dilated = cv2.dilate(edge_frame, np.ones((1,1)))
    inverted = (255 - dilated)
    dilated2 = cv2.dilate(edge_prevframe, np.ones((1,1)))
    inverted2 = (255 - dilated2)

    diff_prev = (edge_prevframe & inverted)
    diff_now = (edge_frame & inverted2)

    pixels_sum_old = np.sum(edge_prevframe)
    pixels_sum_new = np.sum(edge_frame)

    out_pixels = np.sum(diff_prev)
    in_pixels = np.sum(diff_now)

    return pixelRatio(float(in_pixels),float(pixels_sum_new)), pixelRatio(float(out_pixels),float(pixels_sum_old))

def get_EdgePic(img, black, white, centercolor):
    weight = img.shape[1]
    height = img.shape[0]
    # 建立空白圖
    edge_img = np.zeros((height, weight, 3), np.uint8)

    for y in range(0, height-1):
        for x in range(0, weight-1):
            mydown = img[y+1,x,:]
            myright = img[y,x+1,:]
            myhere = img[y, x, :]
            if get_EuclideanDistance(myhere, mydown) > 16 and get_EuclideanDistance(myhere, myright) > 16:
                edge_img[y, x, :] = black
            elif get_EuclideanDistance(myhere, mydown) <= 16 and get_EuclideanDistance(myhere, myright) <= 16:
                edge_img[y, x, :] = white
            else: edge_img[y, x, :] = centercolor

    return edge_img

def compare_Edge(img1, img2):
    weight1 = img1.shape[1]
    height1 = img1.shape[0]

    same = 0
    for y in range(0, height1-1):
        for x in range(0, weight1-1):
            arrImg1 = img1[y, x, :]
            arrImg2 = img2[y, x, :]

            if (np.array_equal(arrImg1, arrImg2)):
                same += 1
    return same

def pr_cruve(answer_image_number, my_answer_image_number):
    TP = 0
    FP = 0
    FN = 0

    np_answer_image_number = np.array(answer_image_number)
    np_my_answer_image_number = np.array(my_answer_image_number)

    TP_array_result = np.in1d(answer_image_number, my_answer_image_number)
    TP_array = np_answer_image_number[TP_array_result]
    # print(TP_array)
    TP = len(TP_array)
    FP = len(my_answer_image_number) - TP
    FN = len(answer_image_number) - TP
    precision = TP / (TP + FP)
    recall = TP / (TP + FN)
    return (precision, recall)

if __name__ == '__main__':
    starttime = datetime.datetime.now()
    fileName = "./soccer_out"
    filePath = "soccer_out"
    imgTitle = "soccer"

    # fileName = "./ngc_out"
    # filePath = "ngc_out"
    # imgTitle = "ngc-"

    # 產生對比線條
    black = np.array([0, 0, 0])
    white = np.array([255, 255, 255])
    centercolor = np.array([125, 125, 125])

    '''Read file'''
    ansNum = []
    for num in range(0, len(os.listdir(fileName))-5):
        strNum= str(num)
        zeroNum = 7 - len(strNum)
        imgName = imgTitle + "0"*zeroNum + strNum + ".jpg"
        path = os.path.join(filePath, imgName)
        img1 = cv2.imread(path)

        strNum2 = str(num+5)
        zeroNum = 7 - len(strNum2)
        imgName = imgTitle + "0" * zeroNum + strNum2 + ".jpg"
        path = os.path.join(filePath, imgName)
        img2 = cv2.imread(path)

        edge_img1 = cv2.Canny(img1, 100, 200)
        edge_img2 = cv2.Canny(img2, 100, 200)
        result = compare_Edge(img1, img2)
        if (result <= 650):
            print("Number:", strNum, "相同像素量:", result)
            ansNum.append(int(strNum))
        # else: print("XXXX Number:", strNum, "相同像素量:", result)

    soccer_ground = [89, 90, 91, 92, 93, 94, 95, 96, 378, 379, 380, 381, 382, 383, 384, 385, 567, 568, 569, 570, 571, 572, 573]
    soccer_out_precision, soccer_out_recall = pr_cruve(soccer_ground, ansNum)
    print('soccer out precision: ', soccer_out_precision)
    print('soccer out recall: ', soccer_out_recall)

    endtime = datetime.datetime.now()
    print("執行時間：", (endtime - starttime))