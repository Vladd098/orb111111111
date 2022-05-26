import numpy as np
import cv2
import time
from matplotlib import pyplot as plt
import math 


MIN_MATCH_COUNT = 7
    
cap = cv2.VideoCapture(0)

t = 0
while True:
    t += 1

    ret, frame = cap.read()
    if not ret:
        print("чет не робит")
        break

    if t == 10:
        t = 0
        query_img = cv2.imread('cam.png',0)
        train_img = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        # query_img_bw = query_img
        # train_img_bw = cv2.cvtColor(train_img, cv2.COLOR_BGR2GRAY)


        # эта фихня дилает кантрольные точки
        sift = cv2.SIFT_create()
        kp1, des1 = sift.detectAndCompute(query_img,None)
        kp2, des2 = sift.detectAndCompute(train_img,None)
        matcher = cv2.BFMatcher()
        matches = matcher.knnMatch(des1, des2, k=2)

        good = []
        for m,n in matches:
            if m.distance < 0.7*n.distance:
                good.append(m)


        if len(good)>MIN_MATCH_COUNT:
            src_pts = np.float32([ kp1[m.queryIdx].pt for m in good ]).reshape(-1,1,2)    #---бессмыслено, только крашит программу
            dst_pts = np.float32([ kp2[m.trainIdx].pt for m in good ]).reshape(-1,1,2)
            M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC,5.0)
            matchesMask = mask.ravel().tolist()
            # print(query_img.shape)
            h,w = query_img.shape
            # time.sleep(1)



            pts = np.float32([ [0,0],[0,h-1],[w-1,h-1],[w-1,0] ]).reshape(-1,1,2)
            try:
                dst = cv2.perspectiveTransform(pts,M)  #тут проблема
            except cv2.error:
                dst = []
            train_img = cv2.polylines(train_img,[np.int32(dst)],True,255,3, cv2.LINE_AA)
            # print('совпадение:', len(good))

        else:
            print( "мало кт - {}/{}".format(len(good), MIN_MATCH_COUNT) )
            matchesMask = None
        draw_params = dict(matchColor = (0,255,0), # draw matches in green color
                           singlePointColor = None,
                           matchesMask = matchesMask, # draw only inliers
                           flags = 2)


        img3 = cv2.drawMatches(query_img,kp1,train_img,kp2,good,None,**draw_params)


        # фенальнае изабражение
        # final_img = cv2.drawMatches(query_img, queryKeypoints,
        # train_img, trainKeypoints, matches[:20],None)
        final_img = cv2.resize(img3, (1000,650))


        cv2.imshow("Matches", final_img)
        cv2.waitKey(1)

cap.release()
cv2.DestroyAllWindows()