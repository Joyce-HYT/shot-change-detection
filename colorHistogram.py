'''
N96091172 HSING-YUN, TSAI
Multimedia HW1 (shot change boundary detection)
'''
import os
import cv2
import datetime
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import precision_recall_curve

def cal_Color(image1, image2):
    # 轉成灰階圖像
    hist1 = cv2.calcHist([image1], [0], None, [256], [0, 256])
    hist2 = cv2.calcHist([image2], [0], None, [256], [0, 256])

    degree = 0
    for i in range(len(hist1)): # 跑每一條 histogram
        if hist1[i] != hist2[i]: # 不相同
            degree = degree + (1 - abs(hist1[i] - hist2[i]) / max(hist1[i], hist2[i]))
        else:degree = degree + 1 # 相同

    degree = degree / len(hist1)
    return degree

# compare_RGBhis is for news
def compare_RGBhis(image1, image2, size):
    '''每個通道 (R,G,B) 的 histogram 相似度'''
    # 將圖片 resize 成一致的大小, size=(寬, 高)
    image1 = cv2.resize(image1, size)
    image2 = cv2.resize(image2, size)

    # 將圖片分成RGB三個通道
    tempImage1 = cv2.split(image1)
    tempImage2 = cv2.split(image2)

    # 分別計算每個通道的相似度
    similiarity = 0

    for img1, img2 in zip(tempImage1, tempImage2):
        similiarity += cal_Color(img1, img2)

    similiarity = similiarity / 3
    return similiarity

# compare_GrayHis is for ngc
def compare_GrayHis(image1, image2):
    grayImg1 = cv2.cvtColor(image1, cv2.COLOR_BGR2GRAY)
    grayImg2 = cv2.cvtColor(image2, cv2.COLOR_BGR2GRAY)

    data1 = np.array(grayImg1)
    hist1 = np.zeros(256)
    for i in range(256):
        temp = (data1 == i)
        hist1[i] = temp.sum()

    data2 = np.array(grayImg2)
    hist2 = np.zeros(256)
    for i in range(256):
        temp = (data2 == i)
        hist2[i] = temp.sum()

    degree = 0
    for i in range(len(hist1)): # 跑每一條 histogram
        if hist1[i] != hist2[i]: # 不相同
            degree = degree + (1 - abs(hist1[i] - hist2[i]) / max(hist1[i], hist2[i]))
        else:degree = degree + 1 # 相同

    degree = degree / len(hist1)
    return degree

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
    
    # 像素值分佈直方圖 (OpenCV 的 calcHist 函數可用來計算直方圖的數值)
    # cv2.calcHist(影像, 通道, 遮罩, 區間數量, 數值範圍)
    # 通道 [0]=blue [1]=green [2]=red
    
    fileName = "./news_out"
    filePath = "news_out"
    imgTitle = "news-"
    # fileName = "./ngc_out"
    # filePath = "ngc_out"
    # imgTitle = "ngc-"
    
    ansNum = []
    for num in range(len(os.listdir(fileName))-1):
        strNum= str(num)
        # zeroNum = 4 - len(strNum)
        # imgName = imgTitle + "0"*zeroNum + strNum + ".jpeg"
        zeroNum = 7 - len(strNum)
        imgName = imgTitle + "0"*zeroNum + strNum + ".jpg"
        path = os.path.join(filePath, imgName)
        img1 = cv2.imread(path)
    
        strNum = str(num+1)
        # zeroNum = 4 - len(strNum)
        # imgName = imgTitle + "0"*zeroNum + strNum + ".jpeg"
        zeroNum = 7 - len(strNum)
        imgName = imgTitle + "0"*zeroNum + strNum + ".jpg"
        path = os.path.join(filePath, imgName)
        img2 = cv2.imread(path)
    
        result = compare_RGBhis(img1, img2, (300, 300))
        # result = compare_GrayHis(img1, img2)
        if (result < 0.8) :
            ansNum.append(int(strNum))
            print("Number:", strNum, " 相似度:", '%.3f'%result)
        # else:print(strNum," 相似度:", '%.3f'%result )
    
    news_out_precision, news_out_recall = pr_cruve([73, 235, 301, 370, 452, 861, 1281], ansNum)
    print('news out precision: ', news_out_precision)
    print('news out recall: ', news_out_recall)
    
    # long running
    endtime = datetime.datetime.now()
    print("執行時間：", (endtime - starttime))

    # plt.figure(1)  # 创建图表1
    # plt.title('PR Curve')  # give plot a title
    # plt.xlabel('Recall')  # make axis labels
    # plt.ylabel('Precision')

    ## y_true和y_scores分别是gt label和predict score
    # y_true = np.array([0, 0, 1, 1])
    # y_scores = np.array([0.1, 0.4, 0.35, 0.8])
    # precision, recall, thresholds = precision_recall_curve(y_true, y_scores)
    # plt.figure(1)
    # plt.plot(recall, precision)
    # plt.show()
    # plt.savefig('p-r.png')