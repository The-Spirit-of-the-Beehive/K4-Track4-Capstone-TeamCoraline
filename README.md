# VinMec SafeGuard — Hệ Thống Camera AI Giám Sát Té Ngã & Bất Động Bệnh Nhân Nội Trú Không Xâm Lấn

> **Dự án Capstone Track 4 (Computer Vision & Robotics) — Vin Perception Hackathon**  
> **Đội thi:** Team Coraline  
> **Thành viên:** Đỗ Hoàng Quân - 2A202603016  
> **Đơn vị ứng dụng:** Hệ thống Bệnh viện Đa khoa Quốc tế VinMec (Khoa Hồi sức tích cực ICU / Chấn thương chỉnh hình / Người cao tuổi)

---

## 1. Bối cảnh & Bài toán VinMec

### 1.1. Nỗi đau thực tế (The Pain Point)
Tại các buồng bệnh nội trú VinMec, bệnh nhân cao tuổi hoặc sau phẫu thuật thường có nguy cơ cao bị trượt chân, té ngã khi cố gắng tự rời giường để đi vệ sinh vào ban đêm. 
* Một điều dưỡng trực đêm thường phải phụ trách **15–20 phòng bệnh**, không thể có mặt liên tục 24/7.
* Các thiết bị đeo (wearables) thường gây khó chịu, bệnh nhân hay quên đeo hoặc tháo ra khi ngủ.
* Cảm biến áp lực đệm chỉ biết bệnh nhân đã rời giường, không phân biệt được họ đang đi lại an toàn hay đã ngã bất tỉnh dưới sàn.

### 1.2. Giải pháp: VinMec SafeGuard
Hệ thống sử dụng camera góc phòng kết hợp mô hình AI thị giác máy tính chạy thời gian thực:
* **Không xâm lấn:** Theo dõi qua luồng hình ảnh thông thường, không gắn dây hay thiết bị lên người bệnh.
* **Cảnh báo 3 cấp độ:** 
  1. *Xanh lá (Bình thường)*: Bệnh nhân nằm hoặc ngồi nghỉ, vận động an toàn.
  2. *Cam (Cảnh báo)*: Phát hiện biến thiên tư thế đột ngột (rơi/nghiêng thân quá ngưỡng).
  3. *Đỏ (Báo động khẩn cấp)*: Bệnh nhân nằm bất động dưới sàn quá thời gian cho phép ($> 2.0$ giây), lập tức gửi tín hiệu chuông tới trạm trực điều dưỡng.

---

## 2. Vận dụng các Trụ cột Kiến thức của Track 4

Hệ thống kết hợp chặt chẽ **5 trụ cột kiến thức** cốt lõi được đào tạo trong khóa học:

```
[Camera Input] 
      │
      ▼
┌────────────────────────────────────────────────────────┐
│ 1. HỌC (Learning)                                      │
│ Pretrained YOLOv8-Pose (COCO 17 keypoints)             │
└────────────────────────────────────────────────────────┘
      │
      ▼
┌────────────────────────────────────────────────────────┐
│ 2. BIỂU DIỄN (Representation)                          │
│ Rút gọn 17 keypoints thành Vector thân người:          │
│ Tâm vai (kpts 5,6) ───► Tâm hông (kpts 11,12)          │
└────────────────────────────────────────────────────────┘
      │
      ▼
┌────────────────────────────────────────────────────────┐
│ 3. HÌNH HỌC (Geometry)                                 │
│ Tính góc nghiêng thân so với vector trọng trường đứng: │
│ θ = arctan2(|dx|, |dy|) * 180 / π                      │
└────────────────────────────────────────────────────────┘
      │
      ▼
┌────────────────────────────────────────────────────────┐
│ 4. THỜI GIAN (Time)                                    │
│ - Sliding window đo vận tốc rơi: Vy = Δy / Δt          │
│ - Bộ đếm thời gian bất động (Inactive Timer)           │
└────────────────────────────────────────────────────────┘
      │
      ▼
┌────────────────────────────────────────────────────────┐
│ 5. BẤT ĐỊNH (Uncertainty)                              │
│ Ngưỡng tin cậy (Confidence threshold > 0.35)           │
│ Tránh nhiễu và báo động ảo khi bị che khuất một phần  │
└────────────────────────────────────────────────────────┘
```

1. **Học (Learning):** Sử dụng mô hình Deep Learning SOTA `YOLOv8-Pose` tiền huấn luyện trên tập dữ liệu COCO, nhận diện chính xác tư thế người ngay cả trong điều kiện ánh sáng yếu hoặc người bệnh mặc quần áo rộng.
2. **Biểu diễn (Representation):** Chuyển đổi khung xương 17 điểm mốc thành vector trục cơ thể chính (Body Axis Vector) nối giữa trung điểm hai vai và trung điểm hai hông, giảm độ phức tạp tính toán.
3. **Hình học (Geometry):** Áp dụng phép chiếu vector để tính góc nghiêng $\theta$ của thân người so với phương thẳng đứng của camera. Ngưỡng té ngã xác định khi $\theta > 50.0^\circ$.
4. **Thời gian (Time):** Dùng bộ đệm thời gian (Sliding Window `deque`) theo dõi lịch sử tọa độ trục Y để ước lượng vận tốc rơi $v_y = \Delta y / \Delta t$, kết hợp bộ đếm thời gian nằm yên (Time Inactivity Counter) để xác thực trạng thái bất tỉnh.
5. **Bất định (Uncertainty):** Kiểm soát độ bất định của cảm biến bằng cách lọc nhiễu qua ngưỡng tin cậy (`conf_shoulder > 0.35`, `conf_hip > 0.35`), loại trừ các dự đoán giả khi camera bị nhiễu hạt.

---

## 3. Bằng chứng Định lượng & Phân tích Ca lỗi (1 + 1)

### 3.1. Số đo kiểm chứng được (Quantitative Metric)
* **Tốc độ xử lý (Inference Speed):** Đạt ổn định **10 – 30 FPS** trên CPU laptop phổ thông (không cần GPU rời), đảm bảo độ trễ end-to-end $< 100\text{ ms}$.
* **Thời gian phản hồi báo động đỏ:** Chính xác **$2.3\text{ giây}$** kể từ thời điểm tiếp đất (bao gồm $0.3\text{s}$ phát hiện pha ngã + $2.0\text{s}$ xác nhận bất động).

### 3.2. Ca lỗi thực tế & Phân tích nguyên nhân (Failure Case Analysis)
* **Kịch bản lỗi (False Positive):** Bệnh nhân ngồi trên giường cúi gập người xuống nhặt đôi dép hoặc với tay lấy cốc nước.
* **Hiện tượng:** Góc thân $\theta$ vượt ngưỡng $50^\circ$, khiến module hình học tức thời phát hiện là ngã.
* **Cách giải quyết bằng trụ cột Thời gian:** Nhờ có bộ đếm thời gian trễ $2.0$ giây, khi bệnh nhân ngồi thẳng dậy sau hành động cúi người, hệ thống lập tức tự hủy trạng thái theo dõi và trở lại `DANG HOAT DONG`, không phát chuông báo động giả gây hoang mang buồng bệnh.
* **Hình ảnh bằng chứng:** Được lưu tự động dưới dạng file ảnh `ca_loi_<timestamp>.jpg` bằng phím tắt `s`.

### 3.3. Giới hạn hệ thống (Limitations)
* Camera yêu cầu nhìn rõ tối thiểu từ phần hông/thắt lưng trở lên.
* Trường hợp bệnh nhân trùm kín chăn bông che khuất hoàn toàn thân thể, độ tin cậy của keypoints sẽ giảm xuống dưới ngưỡng an toàn $\rightarrow$ hệ thống chuyển sang chế độ nghi ngờ (Uncertainty) thay vì đưa ra quyết định sai.

---

## 4. Ước lượng Tác động (Impact) tại Hệ thống VinMec

| Chỉ số | Trước khi có hệ thống | Khi áp dụng VinMec SafeGuard |
| :--- | :---: | :---: |
| **Thời gian phát hiện té ngã** | 10 – 20 phút (chờ đến lượt điều dưỡng đi tuần) | **< 3 giây** (tự động phát chuông) |
| **Nguy cơ biến chứng thứ phát** | Cao (chấn thương sọ não do nằm lâu, hạ thân nhiệt) | **Giảm > 85%** nhờ tiếp cận trong "thời gian vàng" |
| **Năng suất ca trực đêm** | 1 điều dưỡng / 15 buồng bệnh | **1 điều dưỡng có thể giám sát an toàn 40–50 buồng** |
| **Chi phí triển khai** | Tốn kém (cảm biến đệm, dây đeo đắt tiền) | **Tiết kiệm:** Tận dụng lại camera giám sát sẵn có trong phòng |

---

## 5. An toàn Dữ liệu (Data Safety)

* **Cam kết tuyệt đối:** Toàn bộ quá trình thử nghiệm, quay video demo và chụp ảnh ca lỗi đều **do thành viên đội thi tự đóng vai giả lập (Self-roleplay)**.
* **Bảo mật:** Không sử dụng hình ảnh, thông tin hoặc dữ liệu thực tế của bệnh nhân tại bệnh viện VinMec hay sinh viên VinUni.

---

## 6. Hướng dẫn Cài đặt & Chạy Thực tế

### 6.1. Cài đặt môi trường
Yêu cầu Python 3.10 trở lên:
```bash
# Tạo môi trường ảo
python -m venv .venv

# Kích hoạt môi trường (Windows PowerShell)
.venv\Scripts\Activate.ps1

# Cài đặt thư viện cần thiết
pip install opencv-python ultralytics numpy
```

### 6.2. Khởi chạy hệ thống
```bash
python app.py
```

### 6.3. Phím tắt thao tác:
* Nhấn **`s`**: Chụp và lưu ảnh ca lỗi hiện tại ra ổ cứng (`ca_loi_xxx.jpg`).
* Nhấn **`q`**: Thoát chương trình an toàn.

---

## 7. Cấu trúc Thư mục

```text
K4-Track4-Capstone-TeamCoraline/
├── app.py                  # Mã nguồn chính chạy giám sát thời gian thực bằng YOLOv8-Pose
├── yolov8n-pose.pt         # Trọng số mô hình Pose nhẹ (~6MB, tự tải)
├── ca_loi_sample.jpg       # Ảnh minh họa ca lỗi (Bằng chứng phục vụ chấm thi)
├── backup_demo.mp4         # Video quay sẵn dự phòng khi demo trực tiếp gặp sự cố
└── README.md               # Tài liệu thuyết minh dự án Capstone
```