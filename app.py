from collections import deque
import time
import cv2
import numpy as np
from ultralytics import YOLO

# Khởi tạo model YOLOv8 Pose (tự động tải nhẹ ~6MB lần đầu)
model = YOLO("yolov8n-pose.pt")

cap = cv2.VideoCapture(0)

# Cấu hình ngưỡng phát hiện té ngã
FALL_ANGLE_THRESH = 50.0     # Góc thân nghiêng > 50 độ
INACTIVE_TIME_THRESH = 2.0   # Nằm bất động quá 2 giây -> Kích hoạt chuông đỏ

# Bộ đệm thời gian (Time Pillar)
history_y = deque(maxlen=6)
fall_start_time: float = 0.0
is_fallen: bool = False
prev_frame_time = time.time()

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    h, w, _ = frame.shape
    curr_time = time.time()
    dt = curr_time - prev_frame_time
    prev_frame_time = curr_time
    fps = 1.0 / dt if dt > 0 else 0

    # Dự đoán tư thế bằng YOLOv8-Pose
    results = model(frame, verbose=False)

    status = "BINH THUONG"
    status_color = (0, 255, 0) # Xanh lá
    angle = 0.0
    vy = 0.0
    display_frame = frame

    # Kiểm tra xem có người và các điểm khớp không
    if len(results) > 0 and results[0].keypoints is not None and len(results[0].keypoints.data) > 0:
        # Tự động vẽ khung xương của YOLO
        display_frame = results[0].plot()

        # Lấy mốc của người đầu tiên trong khung hình (Shape: 17 keypoints x 3 [x, y, conf])
        kpts = results[0].keypoints.data[0].cpu().numpy()

        # Mốc COCO: 5: vai trái, 6: vai phải | 11: hông trái, 12: hông phải
        conf_shoulder = (kpts[5][2] + kpts[6][2]) / 2.0
        conf_hip = (kpts[11][2] + kpts[12][2]) / 2.0

        # Chỉ tính toán nếu độ tin cậy của các mốc đủ rõ (> 0.35)
        if conf_shoulder > 0.35 and conf_hip > 0.35:
            # 1. BIỂU DIỄN (Representation): Tọa độ tâm vai và tâm hông
            shoulder_x = (kpts[5][0] + kpts[6][0]) / 2.0
            shoulder_y = (kpts[5][1] + kpts[6][1]) / 2.0
            hip_x = (kpts[11][0] + kpts[12][0]) / 2.0
            hip_y = (kpts[11][1] + kpts[12][1]) / 2.0

            norm_hip_y = hip_y / h  # Vị trí hông chuẩn hóa theo chiều dọc (0 = trên cùng, 1 = đáy sàn)

            # 2. HÌNH HỌC (Geometry): Tính góc nghiêng thân người so với phương thẳng đứng
            dx = shoulder_x - hip_x
            dy = shoulder_y - hip_y
            angle = float(np.degrees(np.arctan2(abs(dx), abs(dy))))

            # 3. THỜI GIAN (Time): Vận tốc rơi của hông theo trục Y
            history_y.append((curr_time, norm_hip_y))
            if len(history_y) >= 2:
                dt_hist = history_y[-1][0] - history_y[0][0]
                if dt_hist > 0:
                    vy = (history_y[-1][1] - history_y[0][1]) / dt_hist

            # LOGIC PHÁT HIỆN TÉ NGÃ: Góc nghiêng lớn + Hông ở nửa dưới màn hình
            if angle > FALL_ANGLE_THRESH and norm_hip_y > 0.45:
                if not is_fallen:
                    is_fallen = True
                    fall_start_time = curr_time

                elapsed_down = curr_time - fall_start_time
                if elapsed_down > INACTIVE_TIME_THRESH:
                    status = "CANH BAO: BENH NHAN BAT DONG!"
                    status_color = (0, 0, 255) # Đỏ
                else:
                    status = "PHAT HIEN NGA (DANG THEO DOI)"
                    status_color = (0, 165, 255) # Cam
            else:
                is_fallen = False
                fall_start_time = 0.0
                status = "DANG HOAT DONG"
                status_color = (0, 255, 0)

            # Vẽ vector trục thân (nối giữa vai và hông)
            cv2.line(display_frame, (int(hip_x), int(hip_y)), 
                                    (int(shoulder_x), int(shoulder_y)), (0, 255, 255), 3)

    # BẢNG THÔNG SỐ HUD (Bằng chứng số đo định lượng)
    cv2.rectangle(display_frame, (10, 10), (470, 150), (20, 20, 20), -1)
    cv2.putText(display_frame, f"TRANG THAI: {status}", (20, 40), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.65, status_color, 2)
    cv2.putText(display_frame, f"Goc than: {angle:.1f} deg (Nguong: >{FALL_ANGLE_THRESH})", 
                (20, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (220, 220, 220), 1)
    cv2.putText(display_frame, f"Van toc roi (Vy): {vy:.2f}/s", 
                (20, 95), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (220, 220, 220), 1)
    cv2.putText(display_frame, f"He thong: {fps:.1f} FPS | VinMec ICU-01", 
                (20, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (100, 255, 100), 1)

    cv2.imshow("VinMec SafeGuard - Patient Fall Monitor", display_frame)

    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break
    elif key == ord('s'):
        cv2.imwrite(f"ca_loi_{int(time.time())}.jpg", display_frame)
        print("Da luu anh ca loi!")

cap.release()
cv2.destroyAllWindows()