# 동영상 파일 읽기 (video_play.py)
import os
import cv2
import numpy as np
from function import *

video_file = os.path.join('dataset', 'aaa.mp4') # 동영상 파일 경로
dtect_object = Dtect_object()  # 객체 탐지 모델 생성
detect_straightLine = Detect_straightLine()
cap = cv2.VideoCapture(video_file)

if cap.isOpened():
    while True:
        ret, img = cap.read()   # 프레임 읽기
        # 1. 객체 탐지 좌표 획득 및 거리 예측
        frame = Dtect_object.Get_coordinates(dtect_object, img.copy())

        # 마스크 생성
        height__roi = np.array([[int(img.shape[1]*0.15), img.shape[0]-1], [(img.shape[1]*0.4), int(img.shape[0]*0.6)],
                            [img.shape[1]*0.6, int(img.shape[0]*0.6)], [int(img.shape[1]*0.85), img.shape[0]-1]], np.int32)
        
        # 2. 차선 라인 검출하기
        yellow_roi = Filter_yellow(img, height__roi)
        white_roi = Filter_white(img, height__roi)
        color_mask = cv2.bitwise_or(yellow_roi, white_roi)

        # 3. 화면에 직선 크로스 긋기 
        detected_line = Detect_straightLine.Main(detect_straightLine, color_mask, height__roi, original_img=frame)
        
        # 4. 객체 탐지 박스 그리기
        detected_line = Dtect_object.Draw_box(dtect_object, frame=detected_line)

        # 5. 출력
        if ret:                     
            cv2.imshow(video_file, detected_line)
            if cv2.waitKey(7) & 0xFF == 27: # ESC 입력시 종료
                break
        else:
            break
else:
    print("can't open video.")

cap.release()
cv2.destroyAllWindows()



# distance()

# cv2.imshow("The Sea and the Swing", channel_sclaing_img)
# cv2.waitKey() # 키 입력 대기
# cv2.destroyAllWindows() # 윈도우창 제거