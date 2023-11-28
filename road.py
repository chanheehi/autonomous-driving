import numpy as np
import cv2
from PIL import Image
from matplotlib import pyplot as plt
from typing import *

def split_sky_and_ground(frame: np.ndarray, sky_ratio: float) -> Tuple[np.ndarray, np.ndarray]:
    frame_height = frame.shape[0]
    return frame[:int(frame_height * sky_ratio), :], frame[int(frame_height * sky_ratio):, :]

def filter_by_hsv(frame: np.ndarray, lower: np.ndarray, upper: np.ndarray) -> np.ndarray:
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, lower, upper)
    return mask

def filter_white(frame: np.ndarray) -> np.ndarray:
    return filter_by_hsv(
        frame=frame, 
        lower=np.array([0, 0, 200]), 
        upper=np.array([180, 100, 255]),
    )

def filter_yellow(frame: np.ndarray) -> np.ndarray:
    return filter_by_hsv(
        frame=frame, 
        lower=np.array([20, 100, 150]), 
        upper=np.array([60, 255, 255]),
    )

def show_mask_overlay(sky: np.ndarray, ground: np.ndarray, ground_mask: np.ndarray) -> np.ndarray:
    image_full_size = np.vstack((sky, ground))
    mask_full_size = np.vstack((np.zeros(sky.shape[:2], dtype=np.uint8), ground_mask))
    mask_full_size = cv2.applyColorMap(mask_full_size, cv2.COLORMAP_JET)
    vis = cv2.addWeighted(image_full_size, 0.5, mask_full_size, 0.5, 0.0)
    return vis

def get_points_to_track(mask: np.ndarray) -> np.ndarray:
    px, py = mask.nonzero()
    points = np.array(list(zip(py, px)), np.float32)
    return points.reshape(-1, 1, 2)

def draw_points_to_track(frame: np.ndarray, points: np.ndarray, color: Tuple[int, int, int], radius: int, thickness: int) -> np.ndarray:
    for point in points:
        cv2.circle(frame, tuple(point[0].astype(np.int32)), radius, color, thickness)
    return frame

def filter_only_mask_corner_points(mask: np.ndarray) -> np.ndarray:
    dst = cv2.cornerHarris(mask, 2, 3, 0.04)
    # filtered_points = mask[dst > 0.01 * dst.max()]
    filtered_points = np.where(dst > 0.05 * dst.max(), 255, 0).astype(np.uint8)
    return filtered_points

def main():
    filename = "input.mp4"
    cap = cv2.VideoCapture(filename)
    writer = cv2.VideoWriter("output.mp4", cv2.VideoWriter_fourcc(*'MPV4'), 30, (1280, 720))
    sky_ratio = 0.6
    frame_idx = 0
    prev_state = {}
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        sky, ground = split_sky_and_ground(frame, sky_ratio)
        white_mask = filter_white(ground)
        yellow_mask = filter_yellow(ground)
        lane_mask = cv2.bitwise_or(white_mask, yellow_mask)
        lane_mask = filter_only_mask_corner_points(lane_mask)
        points_to_track = get_points_to_track(lane_mask)

        if frame_idx > 0:
            current_points, status, error = cv2.calcOpticalFlowPyrLK(
                prevImg=prev_state["gray"],
                nextImg=lane_mask,
                prevPts=prev_state["points"],
                nextPts=None,
                winSize=(21, 21),
                maxLevel=5,
                # criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03),
            )
            if current_points is None:
                continue
            prev_valid_points = prev_state["points"][status == 1]
            current_valid_points = current_points[status == 1]
            valid_errors = error[status == 1]
            dy = current_valid_points[:, 1] - prev_valid_points[:, 1]
            dx = current_valid_points[:, 0] - prev_valid_points[:, 0]
            angle = np.arctan2(dy, dx)
            valid_left_angle = np.abs(np.arctan2(dy, dx) - 45 * np.pi / 180) < 10 * np.pi / 180
            valid_right_angle = np.abs(np.arctan2(dy, dx) - 135 * np.pi / 180) < 10 * np.pi / 180
            magnitude = np.sqrt(dy ** 2 + dx ** 2)
            magnitude_lower = 1
            magnitude_upper = 10
            valid_magnitude = (magnitude > magnitude_lower) & (magnitude < magnitude_upper)
            error_lower = 0.05
            valid_error = valid_errors > error_lower
            # lane_points = prev_valid_points[valid_angle & valid_magnitude]
            # prev_lane_points = prev_valid_points[valid_angle & valid_magnitude & valid_error]
            # lane_points = current_valid_points[valid_angle & valid_magnitude & valid_error]
            # lane_points = lane_points.reshape(-1, 1, 2)
            prev_left_lane_points = prev_valid_points[valid_left_angle & valid_magnitude & valid_error]
            prev_right_lane_points = prev_valid_points[valid_right_angle & valid_magnitude & valid_error]
            left_lane_points = current_valid_points[valid_left_angle & valid_magnitude & valid_error]
            right_lane_points = current_valid_points[valid_right_angle & valid_magnitude & valid_error]
            prev_lane_points = np.vstack((prev_left_lane_points, prev_right_lane_points))
            lane_points = np.vstack((left_lane_points, right_lane_points))
            lane_points = lane_points.reshape(-1, 1, 2)
            prev_lane_points = prev_lane_points.reshape(-1, 1, 2)

        if frame_idx > 0:
            for cur_pt in left_lane_points:
                cv2.circle(ground, tuple(cur_pt.astype(np.int32)), 3, (0, 0, 255), 1)
            for cur_pt in right_lane_points:
                cv2.circle(ground, tuple(cur_pt.astype(np.int32)), 3, (255, 0, 0), 1)
            # for prev_pt, cur_pt in zip(prev_left_lane_points, left_lane_points):
            #     cv2.arrowedLine(ground, tuple(prev_pt.astype(np.int32)), tuple(cur_pt.astype(np.int32)), (0, 0, 255), 1)
            # for prev_pt, cur_pt in zip(prev_right_lane_points, right_lane_points):
            #     cv2.arrowedLine(ground, tuple(prev_pt.astype(np.int32)), tuple(cur_pt.astype(np.int32)), (255, 0, 0), 1)
            # draw_points_to_track(ground, lane_points, (0, 255, 0), radius=3, thickness=1)

        # overall_vis = show_mask_overlay(sky, ground, lane_mask)
        # cv2.imshow("vis", overall_vis)
        result = np.vstack((sky, ground))
        cv2.imshow("vis", result)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
        writer.write(result)
        prev_state["gray"] = lane_mask
        prev_state["points"] = points_to_track
        frame_idx += 1
    cap.release()
    writer.release()

if __name__ == "__main__":
    main()
