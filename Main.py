# 동영상 파일 읽기 (video_play.py)
import os
import cv2
import numpy as np
from function import Detect_straightLine, mouse, Filter_yellow, Filter_white, Dtect_object

video_file = os.path.join('dataset', 'aaa.mp4') # 동영상 파일 경로
model = Dtect_object()  # 객체 탐지 모델 생성
cap = cv2.VideoCapture(video_file)

if cap.isOpened():
    while True:
        ret, img = cap.read()   # 프레임 읽기
        # 1. 객체 탐지
        detected_obejct = Dtect_object.detect(model, img)

        # mouse(img)
        # 관심영역 지정
        width_roi = {'x' : 0, 'y' : int(img.shape[0]/2)+50, 'w' : img.shape[1], 'h' : img.shape[0]}
        height_roi = {'x' : int(img.shape[1]*0.15), 'y' : int(img.shape[0]/2)+50, 'w' : int(img.shape[1]*0.85), 'h' : img.shape[0]}
        pts_roi = np.array([[int(img.shape[1]*0.2), img.shape[0]-1], [(img.shape[1]*0.4), int(img.shape[0]*0.6)],
                            [img.shape[1]*0.6, int(img.shape[0]*0.6)], [int(img.shape[1]*0.8), img.shape[0]-1]], np.int32)
        # img = cv2.polylines(img, [pts_roi], True, (0, 255, 255), 2)
        # cv2.rectangle(img, (width_roi['x'],width_roi['y']), (width_roi['w'], width_roi['h']), (0,255,0))
        # cv2.rectangle(img, (height_roi['x'], height_roi['y']), (height_roi['w'], height_roi['h']), (0,255,0))
        # mouse(img)

        # 2. 라인검출
        # 관심영역 색 추출하기
        img_copy = img.copy()
        yellow_roi = Filter_yellow(img_copy, width_roi)
        white_roi = Filter_white(img_copy, width_roi)
        color_mask = cv2.bitwise_or(yellow_roi, white_roi)
        copy_color_mask = color_mask.copy()

        # 직선 긋기
        detected_line = Detect_straightLine(copy_color_mask, pts_roi, original_img=img)
        # detected_reuslt = cv2.bitwise_or(detected_obejct, detected_line)
        # a = np.concatenate([line_mask[..., np.newaxis] for _ in range(3)], axis=2)
        # line_mask = cv2.cvtColor(line_mask, cv2.COLOR_GRAY2BGR)

        # mask의 레이어를 크기 img에 맞추기
        # mask = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        # detected_line = cv2.bitwise_or(img, detected_line)
        # img = cv2.bitwise_or(img, line_mask)
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



# cv2.imshow("The Sea and the Swing", channel_sclaing_img)
# cv2.waitKey() # 키 입력 대기
# cv2.destroyAllWindows() # 윈도우창 제거