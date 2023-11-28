from typing import Optional, Tuple
import numpy as np
import cv2
import matplotlib.pyplot as plt
import torch
from keras.models import load_model
import os
import pandas as pd
from keras.models import model_from_json
from keras.models import Sequential
from sklearn.preprocessing import StandardScaler
import csv

model = 'models/model@1535470106.json'
weights = 'models/model@1535470106.h5'
csvfile_path = './data.csv'
results_dir = '.'

# def distance():
#     # get data
#     df_test = pd.read_csv(csvfile_path)
#     x_test = df_test[['scaled_xmin', 'scaled_ymin', 'scaled_xmax', 'scaled_ymax']].values

#     # standardized data
#     scalar = StandardScaler()
#     x_test = scalar.fit_transform(x_test)
#     scalar.fit_transform((df_test[['scaled_ymax']].values - df_test[['scaled_ymin']])/3)

#     # load json and create model
#     json_file = open(model, 'r')
#     loaded_model_json = json_file.read()
#     json_file.close()
#     loaded_model = model_from_json(loaded_model_json)

#     # load weights into new model
#     loaded_model.load_weights(weights)
#     print("Loaded model from disk")

#     # evaluate loaded model on test data
#     loaded_model.compile(loss='mean_squared_error', optimizer='adam')
#     distance_pred = loaded_model.predict(x_test)

#     # scale up predictions to original values
#     distance_pred = scalar.inverse_transform(distance_pred)

#     # save predictions
#     df_result = df_test
#     df_result['distance'] = None
    
#     for idx, row in df_result.iterrows():
#         df_result.at[idx, 'distance'] = distance_pred[idx]
#     df_result.to_csv(os.path.join(results_dir, 'predictions.csv'), index=False)


# 차 객체와의 거리
class Dtect_object():
    def __init__(self) -> None:
        self.yolo_model = torch.hub.load('WongKinYiu/yolov7', 'custom', path_or_model='models/yolov7_best.pt', force_reload=False)
        self.device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
        self.yolo_model.to(self.device)
        self.yolo_model.eval()
        self.coordinates = 0
        self.results = 0
        self.num = 0
        
        # Get_distance
        self.results_bundle = [['scaled_xmin', 'scaled_ymin', 'scaled_xmax', 'scaled_ymax', 'distance']]

    def Get_coordinates(self, frame: np.ndarray) -> None:
        self.results = self.yolo_model(frame)

        self.coordinates = self.results.pandas().xyxy[0]
        # self.coordinates.drop(self.coordinates[self.coordinates['confidence'] <0.35].index, inplace=True)   # 35% 이하의 확률인 박스는 제거
        # self.Get_distance(frame, self.coordinates)
        object_cars = []
        # for _ in range(len(self.coordinates)):
        #     # car만 추출
        #     if self.coordinates['name'][_] == 'car':
        #         # results의 _번째 인덱스의 xmin, ymin, xmax, ymax를 리스트로 저장
        #         object_cars.append([int(self.coordinates['xmin'][_]),int(self.coordinates['ymin'][_]),
        #                             int(self.coordinates['xmax'][_]), int(self.coordinates['ymax'][_])])

        return Dtect_object.Get_distance(self, frame, self.coordinates)

    def Get_distance(self, frame: np.ndarray, coordinates: pd.DataFrame) -> None:
        self.num += 1
        originalvideoSize = (375, 1242)
        originalvideoHieght = originalvideoSize[0]
        originalvideoWidth = originalvideoSize[1]
        imgHeight = frame.shape[0]
        imgWidth = frame.shape[1]
        scaledDict = {'scaled_xmin':[], 'scaled_ymin':[], 'scaled_xmax':[], 'scaled_ymax':[]}

        for i in range(len(coordinates)):
            x1 = int(coordinates['xmin'][i])
            y1 = int(coordinates['ymin'][i])
            x2 = int(coordinates['xmax'][i])
            y2 = int(coordinates['ymax'][i])
            scaledX1 = (x1 / imgWidth) * originalvideoWidth
            scaledX2 = (x2 / imgWidth) * originalvideoWidth
            scaledY1 = (y1 / imgHeight) * originalvideoHieght
            scaledY2 = (y2 / imgHeight) * originalvideoHieght

            scaledDict['scaled_xmin'].append(scaledX1)
            scaledDict['scaled_ymin'].append(scaledY1)
            scaledDict['scaled_xmax'].append(scaledX2)
            scaledDict['scaled_ymax'].append(scaledY2)
            
        scaledDf = pd.DataFrame(scaledDict)
        x_test = scaledDf[['scaled_xmin', 'scaled_ymin', 'scaled_xmax', 'scaled_ymax']].values
        scalar = StandardScaler()
        x_test = scalar.fit_transform(x_test)
        scalar.fit_transform((scaledDf[['scaled_ymax']].values - scaledDf[['scaled_ymin']])/3)

        # load json and create model
        json_file = open(model, 'r')
        loaded_model_json = json_file.read()
        json_file.close()
        loaded_model = model_from_json(loaded_model_json)

        # load weights into new model
        loaded_model.load_weights(weights)
        print("Loaded model from disk")

        # evaluate loaded model on test data
        loaded_model.compile(loss='mean_squared_error', optimizer='adam')
        distance_pred = loaded_model.predict(x_test)

        # scale up predictions to original values
        distance_pred = scalar.inverse_transform(distance_pred)
        
        # 이미지에 거리 표시
        for idx, row in scaledDf.iterrows():
            cv2.putText(frame, str(round(distance_pred[idx][0], 2))+'M', (int(coordinates['xmin'][idx]), int(coordinates['ymin'][idx])-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)

        return frame

    def Draw_box(self, frame: np.ndarray) -> np.ndarray:
        for _ in range(len(self.coordinates)):
            cv2.rectangle(frame, (int(self.coordinates['xmin'][_]),int(self.coordinates['ymin'][_])),
                            (int(self.coordinates['xmax'][_]), int(self.coordinates['ymax'][_])), (0,255,0), 1)
            
        return frame

def Filter_by_hsv(frame: np.ndarray, lower: np.ndarray, upper: np.ndarray, roi: dict) -> np.ndarray:
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    mask_reulst = cv2.inRange(hsv, lower, upper)

    mask_black = np.zeros(frame.shape[:2], dtype=np.uint8)
    cv2.fillConvexPoly(mask_black, roi, (255, 255, 255))

    masked_reults = cv2.bitwise_and(mask_reulst, mask_reulst, mask=mask_black)

    return masked_reults

def Filter_white(frame: np.ndarray, width_roi: dict) -> np.ndarray:
    return Filter_by_hsv(
        frame=frame, 
        lower=np.array([0, 0, 200]), 
        upper=np.array([180, 100, 255]),
        roi=width_roi,
    )

def Filter_yellow(frame: np.ndarray, width_roi: dict) -> np.ndarray:
    return Filter_by_hsv(
        frame=frame, 
        lower=np.array([20, 80, 80]), 
        upper=np.array([60, 150, 200]),
        roi=width_roi,
    )

# 직선 라인 검출/ 점으로

class Detect_straightLine():
    def __init__(self) -> None:
        self.previous_t = 0

    def Main(self, mask_image: np.ndarray, pts_roi: dict, original_img: np.ndarray) -> np.ndarray:
        # 왼쪽 라인 검출을 위한 마스킹
        left_mask_black = np.zeros(mask_image.shape[:2], dtype=np.uint8)
        left_pts = pts_roi.copy()
        left_pts[1][0] = int(original_img.shape[1]*0.37)
        left_pts[2][0] = int(original_img.shape[1]*0.50)
        left_pts[3][0] = int(original_img.shape[1]*0.35)
        cv2.fillConvexPoly(left_mask_black, left_pts, (255, 255, 255))
        # 오른쪽 라인 검출을 위한 마스킹
        right_mask_black = np.zeros(mask_image.shape[:2], dtype=np.uint8)
        right_pts = pts_roi.copy()

        right_pts[0][0] = int(original_img.shape[1]*0.65)
        right_pts[1][0] = int(original_img.shape[1]*0.50)
        right_pts[2][0] = int(original_img.shape[1]*0.63)
        cv2.fillConvexPoly(right_mask_black, right_pts, (255, 255, 255))

        left_masked = cv2.bitwise_and(mask_image, mask_image, mask=left_mask_black)
        right_masked = cv2.bitwise_and(mask_image, mask_image, mask=right_mask_black)
        left_edges = cv2.Canny(left_masked, 50, 130)
        right_edges = cv2.Canny(right_masked, 50, 130)
        
        try:
            # 직선 그리기
            for threshold in range(180, 0, -20):
                left_lines = cv2.HoughLines(left_edges, 2, np.pi/180, threshold)
                right_lines = cv2.HoughLines(right_edges, 2, np.pi/180, threshold)
                left_rho, left_theta, ok = Get_line_params(left_lines)
                if not ok:
                    continue
                right_rho, right_theta, ok = Get_line_params(right_lines)
                if not ok:
                    continue
                Draw_line(original_img, left_rho, left_theta)
                Draw_line(original_img, right_rho, right_theta)

                t = (left_rho * np.cos(left_theta) - right_rho * np.cos(right_theta)) / (np.sin(left_theta) - np.sin(right_theta))

                if t == np.float32('inf') or t == 0:
                    t = self.previous_t
                x, y = Get_xy_on_line(left_rho, left_theta, t)
                cv2.circle(original_img, (int(x), int(y)), 5, (0, 255, 0), -1)
                left_t, right_t = Solve(left_rho, left_theta, right_rho, right_theta)
                left_x, left_y = Get_xy_on_line(left_rho, left_theta, left_t)
                right_x, right_y = Get_xy_on_line(right_rho, right_theta, right_t)
                left_bottom_t = (original_img.shape[0]-left_rho * np.sin(left_theta)) / np.cos(left_theta)
                right_bottom_t = (original_img.shape[0]-right_rho * np.sin(right_theta)) / np.cos(right_theta)
                left_bottom_x, left_bottom_y = Get_xy_on_line(left_rho, left_theta, left_bottom_t)
                right_bottom_x, right_bottom_y = Get_xy_on_line(right_rho, right_theta, right_bottom_t)
                left_bottom_x, left_bottom_y = int(left_bottom_x), int(left_bottom_y)
                right_bottom_x, right_bottom_y = int(right_bottom_x), int(right_bottom_y)
                cv2.circle(original_img, (left_bottom_x, left_bottom_y), 5, (0, 0, 255), -1)
                cv2.circle(original_img, (right_bottom_x, right_bottom_y), 5, (0, 0, 255), -1)
                cv2.circle(original_img, (int(left_x), int(left_y)), 5, (0, 0, 255), -1)
                cv2.circle(original_img, (int(right_x), int(right_y)), 5, (0, 0, 255), -1)
                
                pts = np.array([[left_bottom_x, left_bottom_y], [int(left_x), int(left_y)], [right_bottom_x, right_bottom_y]], np.int32)
                # cv2.polylines(original_img, [pts], True, (0, 255, 255), -2)
                poly_canvas = original_img.copy()
                cv2.fillPoly(poly_canvas, [pts], color=(255, 200, 0))
                original_img = cv2.addWeighted(original_img, 0.7, poly_canvas, 0.3, 0)

                if t != 0:
                    self.previous_t = t
                break
        except:
            pass
        return original_img

def Solve(r1, t1, r2, t2):
    s1, c1 = np.sin(t1), np.cos(t1)
    s2, c2 = np.sin(t2), np.cos(t2)
    # v = (r1 - r2 * (c1 * c2 + s1 * s2)) / (s1 * c2 - s2 * c1)
    # return v, v
    a = np.array([[s1, -s2], [c1, -c2]])
    b = np.array([r1*c1-r2*c2, -r1*s1+r2*s2])
    sol  = np.linalg.solve(a, b)
    return sol[0], sol[1]

def Get_crosspt(x11,y11, x12,y12, x21,y21, x22,y22):
    if x12==x11 or x22==x21:
        print('delta x=0')
        if x12==x11:
            cx = x12
            m2 = (y22 - y21) / (x22 - x21)
            cy = m2 * (cx - x21) + y21
            return cx, cy
        if x22==x21:
            cx = x22
            m1 = (y12 - y11) / (x12 - x11)
            cy = m1 * (cx - x11) + y11
            return cx, cy

    m1 = (y12 - y11) / (x12 - x11)
    m2 = (y22 - y21) / (x22 - x21)
    if m1==m2:
        print('parallel')
        return None
    print(x11,y11, x12, y12, x21, y21, x22, y22, m1, m2)
    cx = (x11 * m1 - y11 - x21 * m2 + y21) / (m1 - m2)
    cy = m1 * (cx - x11) + y11

    return cx, cy


def Get_line_params(lines) -> Tuple[float, float, bool]:
    try:
        rho = lines[0][0][0]
        theta = lines[0][0][1]
        return rho, theta, True
    except:
        return 0, 0, False
    
def Get_xy_on_line(rho: float, theta: float, t: float) -> Tuple[float, float]:
    x = rho * np.cos(theta) - t * np.sin(theta)
    y = rho * np.sin(theta) + t * np.cos(theta)
    return x, y

def Draw_line(original_img, rho, theta) -> None:
    x1, y1 = Get_xy_on_line(rho, theta, -10000)
    x2, y2 = Get_xy_on_line(rho, theta, 10000)
    x1, y1 = int(x1), int(y1)
    x2, y2 = int(x2), int(y2)
    cv2.line(original_img, (x1, y1), (x2, y2), (0, 0, 255), 1)


def mouse(src):
    # 원본 이미지를 띄우고, 마우스 이벤트 처리도 도와줌
    roi = cv2.selectROI(src)    # 원하는 부분(관심영역,  roi)을 선택하기
    print('roi =', roi)	# (x 시작 지점, y 시작 지점, x축 드래그 한 길이, y축 드래그 한 길이)
    #print(type(roi))    # <class 'tuple'>
    
    img = src[roi[1]:roi[1] + roi[3],
                roi[0]:roi[0] + roi[2]]
                
    #print(type(img))    # <class 'numpy.ndarray'>
    
    # 원본 이미지인 img를 띄워주는 코드는 없음
    cv2.imshow('Img', img)  # 관심영역을 새 창으로 띄워주기
    cv2.waitKey()
    cv2.destroyAllWindows()