# Video Shot Change Detection
###### tags: `Course`
If there anything wrong or any question, welcome share with me.

## 作業說明
給指定三部影片及對應的video frames、shot change boundary，請撰寫程式完成video shot change detection，程式可直接輸入mpg或video frames。

## 程式執行環境
作業系統：Windows 10
程式語言：Python 3.8
IDE：PyCharm 2017.3.2

## Visual Feature
news 及 ngc 使用 color histogram
soccer 使用 edge detection

## 演算法
### color histogram 
先將兩張圖片轉成一致的大小，而圖片由RGB三色所構成，我將兩張圖片切各分為Red、Green、Blue三個通道，並丟入cal_Color() 比較兩者各自的**Red相似度**、**Green相似度**以及**Blue相似度**，最後以 similiarity儲存相似值，而因為該值是RGB三色相似值的總和，因次需要除以3再回傳結果，此結果即為**圖形相似度**。
```python
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
``` 

將圖片轉為灰階圖像並畫出直方圖，再比較兩個直方圖中，每一條值的差異。若兩條值不同，則兩者相減取絕對值，並以取最大值作為分母，來得出相似度；若兩值相同則直接加一，最後，因為是將直方圖每一條的值加總計算相似度並存放於degree中，必須除以 橫軸 總數 ( =有幾條值 = 256條 )，除完之後即為**該顏色的相似度**。
```python
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
```

閾值的部份，我是分析每張圖計算下來的相似度之後，取0.8作為門檻，但好的做法應該是透過某種算式來決定門檻才正確。
```python
        result = compare_RGBhis(img1, img2, (300, 300))
        # result = compare_GrayHis(img1, img2)
        if (result < 0.8) :
            ansNum.append(int(strNum))
            print("Number:", strNum, " 相似度:", '%.3f'%result)
```
 
### edge detection
利用Canny method將圖片轉化為邊緣圖，並丟入compare_Edge()比較兩圖之間的像素差異。

![](https://i.imgur.com/6h7G5xB.png) ![](https://i.imgur.com/cGEJJBU.png)

```python
        edge_img1 = cv2.Canny(img1, 100, 200)
        edge_img2 = cv2.Canny(img2, 100, 200)
        result = compare_Edge(img1, img2)
```

比較兩圖之間，每一格像素是否相同，若相同則相似度加一，存入same
```python
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
```

閾值的部份，在看完所有圖片的相同像素量之後，選擇650 (低於650個相同像素量則判定為dissolve) 為門檻。
```python
        result = compare_Edge(img1, img2)
        if (result <= 650):
            print("Number:", strNum, "相同像素量:", result)
            ansNum.append(int(strNum))
```
 
## 效能
| Video name | news | soccer | ngc |
| -------- | -------- | --- | -------- |
| Algorithm | Color histogram | Edge detection | Color histogram|
| Precision | 0.875 | 0.106 | 0.688 |
| Recall | 1.0  | 0.826 | 0.693 |
| Run time | 12.75 secs | 7mins | 25 secs |

### news 
| Precision | Recall | Run time |
| -------- | -------- | -------- |
| 0.875 | 1.0 | 12.75 secs |
 
與news_ground相比多了一張編號451的圖片，因為我是以顏色來做比對，從450轉變到451的時候有區塊從黑色變白色，因此被判斷為shot change

| ![](https://i.imgur.com/n9vylvw.png)| ![](https://i.imgur.com/yJhYhXP.png) |
| -------- | -------- |
| (A)編號450 | (B)編號451 |
   
### soccer
| Precision | Recall | Run time |
| -------- | -------- | -------- |
| 0.106 | 0.826 | 7mins |

> Li, Zongjie, Xiabi Liu, and Shuwen Zhang. "Shot boundary detection based on multilevel difference of colour histograms." 2016 First International Conference on Multimedia and Image Processing (ICMIP). IEEE, 2016. 

此篇論文中所提到解決gradual change的方法，該方法會間隔幾張圖來做漸進變化的比較，因此在soccer的影片中，我在試了間隔2至5圖來做比較之後，發現在precision差不多的情況下，間隔5張圖片的效果較佳

| ![](https://i.imgur.com/NEzLh1x.png)| ![](https://i.imgur.com/9dVTeSD.png)|
| -------- | -------- |
| 間隔2張圖片的效能  | 間隔5張圖片的效能 |

### ngc 
| Precision | Recall | Run time |
| -------- | -------- | -------- |
| 0.688 | 0.693 | 25secs |

此篇影片是使用Color Histogram來做shot change detection，使用原本預測的閾值0.8來執行演算法的話，Precision和Recall都能夠得到接近70% 的的結果。而我試著提高閾值，Recall可以提升至0.83但是Precision卻下降至0.596，我也有嘗試降低閾值，但Precision和Recall皆會下降，於是就不放上此份報告中了。
ngc的影片不像是單純以顏色判斷或是邊緣偵測就能夠判斷shot change的，本來也有嘗試使用edge detection的方法，但是30分鐘過去卻只跑了300張左右的圖片，於是放棄，而我有另外找到ECR (edge change ratio) 的方法判斷此部影片的場景變化，但是效果也不是很好…。

| ![](https://i.imgur.com/VBaW4pF.png)| ![](https://i.imgur.com/JCG7moI.png)|
| -------- | -------- |
| threshold : 0.8  | threshold : 0.85 |
