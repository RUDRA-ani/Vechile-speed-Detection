import cv2
import torch
import numpy as np
import time
import math
from ultralytics import YOLO
from deep_sort_realtime.deepsort_tracker import DeepSort
import supervision as sv
# from yolov5.utils.general import scale_boxes  # Import scale_coords function
import sqlite3 as sq

# --- Settings ---
pixel_meter_ratio = 0.05  # meters per pixel
SPEED_THRESHOLD = 50  # km/h threshold

# --- Load Models ---
model_vehicle = torch.hub.load('ultralytics/yolov5', 'custom',
                               path=r'Your path here ',
                               force_reload=True, _verbose=False)
model_vehicle.conf = 0.55
model_vehicle.iou = 0.6
model_vehicle.eval()

model_chars = YOLO(r'Your path here').to('cuda')

tracker = DeepSort(max_age=30)

# --- Speed Estimation ---
track_speed_data = {}


# --- SQLite Database Setup ---
conn = sq.connect(r'Enter required file path')  # Create or connect to the SQLite database
c = conn.cursor()

# Create the table to store detected license plates
c.execute('''CREATE TABLE IF NOT EXISTS plate_data (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                tracker_id INTEGER,
                recognized_text TEXT,
                detection_time TEXT
            )''')
conn.commit()


# --- Video Setup ---
cap = cv2.VideoCapture(r'Path for the video file')
fps = cap.get(cv2.CAP_PROP_FPS)
orig_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
orig_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter(r 'path for output video', fourcc, fps, (orig_width, orig_height))

frame_count = 0
start_time = time.time()

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break
    frame_count += 1
    current_time = time.time()

    orig_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    orig_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    with torch.no_grad():
        results = model_vehicle(frame)

    # Scale detections correctly
    # results.xyxy[0] = scale_boxes(frame.shape[:2], results.xyxy[0].clone(), (orig_height, orig_width)).round()

    # Get detections in xyxy format
    detections = results.xyxy[0].cpu().numpy()
    vehicles = []
    for det in detections:
        x1, y1, x2, y2, conf, cls = det
        if int(cls) != 0:  # Ignore class 0 (non-vehicles)
            # x1, y1, x2, y2 = map(int, (x1, y1, x2, y2))  # Ensure integer values
            vehicles.append([x1, y1, x2-x1, y2-y1, float(conf), int(cls)])

    # Track vehicles
    if len(vehicles) > 0:
        vehicle_boxes = [([v[0], v[1], v[2], v[3]], v[4], v[5]) for v in vehicles]
        tracks = tracker.update_tracks(vehicle_boxes, frame=frame)
    else:
        tracks = []

    # Process each tracked vehicle
    for track in tracks:
        track_id = track.track_id
        x1, y1, x2, y2 = track.to_ltrb()
        ltrb = track.to_ltrb()
        xmin, ymin, xmax, ymax = int(ltrb[0]), int(
            ltrb[1]), int(ltrb[2]), int(ltrb[3])


        # Speed Estimation
        center = ((x1 + x2) / 2, (y1 + y2) / 2)
        if track_id in track_speed_data:
            prev_center = track_speed_data[track_id]['center']
            prev_time = track_speed_data[track_id]['time']
            dt = current_time - prev_time
            if dt > 0:
                dx = center[0] - prev_center[0]
                dy = center[1] - prev_center[1]
                distance_pixels = math.sqrt(dx * dx + dy * dy)
                speed_m_s = (distance_pixels * pixel_meter_ratio) / dt
                speed_kmph = speed_m_s * 3.6
                track_speed_data[track_id]['speed'] = speed_kmph
            track_speed_data[track_id]['center'] = center
            track_speed_data[track_id]['time'] = current_time
        else:
            track_speed_data[track_id] = {'center': center, 'time': current_time, 'speed': 0}

        # Draw vehicle bounding box
        vehicle_detection = sv.Detections(
            xyxy=np.array([[xmin, ymin, xmax, ymax]]),
            tracker_id=np.array([track_id]),
            confidence=np.array([1.0])
        )
        vehicle_label = f"ID:{track_id} {int(track_speed_data[track_id]['speed'])} km/h"
        Ltframe = sv.BoxAnnotator(color=sv.Color.RED, thickness=2)
        frame = Ltframe.annotate(scene=frame, detections=vehicle_detection, labels=[vehicle_label])

        # License Plate Detection
        if track_speed_data[track_id].get('speed', 0) > SPEED_THRESHOLD:
            vx1, vy1, vx2, vy2 = max(0, int(x1)), max(0, int(y1)), min(orig_width, int(x2)), min(orig_height, int(y2))
            if vx2 - vx1 <= 0 or vy2 - vy1 <= 0:
                continue
            vehicle_roi = frame[vy1:vy2, vx1:vx2].copy()
            if vehicle_roi.size == 0:
                continue

            with torch.no_grad():
                lp_results = model_vehicle(vehicle_roi)

            lp_detections = lp_results.xyxy[0].cpu().numpy()
            lp_box = None
            for d in lp_detections:
                cx1, cy1, cx2, cy2, cconf, ccls = d
                if int(ccls) == 0:  # License plate detected
                    lp_box = [int(cx1), int(cy1), int(cx2), int(cy2)]
                    break

            if lp_box is not None:
                px1, py1, px2, py2 = lp_box
                px1, py1, px2, py2 = max(0, px1), max(0, py1), min(vx2-vx1, px2), min(vy2-vy1, py2)
                if px2 - px1 <= 0 or py2 - py1 <= 0:
                    continue
                lp_crop = vehicle_roi[py1:py2, px1:px2].copy()
                if lp_crop.size == 0:
                    continue

                # Character Recognition
                char_results = model_chars.predict(lp_crop, conf=0.5, device='cuda')
                recognized_chars = []
                for result in char_results:
                    for box in result.boxes:
                        x_coord = box.xyxy[0][0].item()
                        label_char = result.names[int(box.cls[0].item())]
                        recognized_chars.append((x_coord, label_char))

                recognized_chars.sort(key=lambda x: x[0])
                recognized_text = ''.join([char for _, char in recognized_chars])

                detection_time = time.strftime('%Y-%m-%d %H:%M:%S', time.gmtime(current_time))
                c.execute('''INSERT INTO plate_data (tracker_id, recognized_text, detection_time) 
                             VALUES (?, ?, ?)''', (track_id, recognized_text, detection_time))
                conn.commit()

                # Draw License Plate Bounding Box
                lp_detection = sv.Detections(
                    xyxy=np.array([[vx1 + px1, vy1 + py1, vx1 + px2, vy1 + py2]])
                )
                frame = sv.BoxAnnotator(color=sv.Color.BLUE, thickness=2).annotate(scene=frame, detections=lp_detection, labels=[recognized_text])

    out.write(frame)

cap.release()
out.release()
conn.close()

end_time = time.time()
print("Processing complete.")
print("Total frames processed:", frame_count)
print("Total time (s):", end_time - start_time)