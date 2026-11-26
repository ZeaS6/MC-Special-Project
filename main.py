"""
main.py — โปรแกรมหลักสำหรับ CCTV + YOLOv11 + ByteTrack
รันด้วยคำสั่ง: python main.py
ต้องวาง main.py และ config.py ไว้ในโฟลเดอร์เดียวกัน
"""
import os
import time
import cv2
from ultralytics import YOLO
import config
from line_notification import send_line_message
from datetime import datetime
from draw_detections import draw_detections_highest_confidence


def load_model():
    print("กำลังตรวจสอบไฟล์โมเดล...")
    print(f"Model path: {config.MODEL_PATH}")
    if not config.MODEL_PATH.is_file():
        raise FileNotFoundError(
            f"ไม่พบไฟล์ best.pt ที่: {config.MODEL_PATH}\n"
            "กรุณาตรวจสอบ MODEL_PATH ใน config.py"
        )
    model = YOLO(str(config.MODEL_PATH))
    print("โหลดโมเดลสำเร็จ:", model.names)
    return model


def connect_camera():
    print("กำลังเชื่อมต่อกล้อง CCTV...")
    print(f"Camera: {config.CCTV_IP}:{config.CCTV_PORT}, "
          f"Channel: {config.CCTV_CHANNEL}")
    cap = cv2.VideoCapture(config.RTSP_URL, cv2.CAP_FFMPEG)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    if not cap.isOpened():
        cap.release()
        raise ConnectionError(
            "เชื่อมต่อกล้องไม่สำเร็จ กรุณาตรวจสอบ IP, username, "
            "password, RTSP port, channel และเครือข่าย"
        )
    return cap


def main():
    model = YOLO(str(config.MODEL_PATH), task="detect")
    print("\n========== YOLO MODEL CLASSES ==========")
    for class_id, class_name in model.names.items():
        print(f"Class ID: {class_id} | Class Name: {class_name}")
    print("========================================\n")
    
    last_alert_time = {}
    alert_interval = 120
    
    cap = cv2.VideoCapture(config.RTSP_URL)
    send_line_message("ทดสอบการเชื่อมต่อ LINE จากระบบ AI Surveillance")
    
    fps = cap.get(cv2.CAP_PROP_FPS) or 20.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    print(f"เชื่อมต่อสำเร็จ: {width}x{height}, FPS: {fps:.1f}")
    print(f"ปรับภาพเป็น {config.RESIZE_WIDTH}x{config.RESIZE_HEIGHT}")
    print(f"Confidence threshold = {config.CONF_THRESHOLD}")
    print("กด q เพื่อหยุดโปรแกรม")

    # writer = None
    # if config.SAVE_OUTPUT:
    #     fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    #     frame_skip = max(1, int(config.SKIP_FRAMES))
    #     writer = cv2.VideoWriter(
    #         str(config.OUTPUT_VIDEO_PATH), fourcc, fps,
    #         (config.RESIZE_WIDTH, config.RESIZE_HEIGHT)
    #     )
    #     if not writer.isOpened():
    #         writer.release()
    #         writer = None
    #         print("คำเตือน: ไม่สามารถสร้างไฟล์วิดีโอผลลัพธ์ได้")

    frame_idx = 0
    prev_time = time.time()

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                print("อ่านเฟรมไม่ได้ กำลังลองเชื่อมต่อใหม่...")
                cap.release()
                time.sleep(1)
                cap = cv2.VideoCapture(config.RTSP_URL, cv2.CAP_FFMPEG)
                cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
                if not cap.isOpened():
                    print("ยังเชื่อมต่อกล้องไม่ได้ จะลองใหม่")
                    time.sleep(2)
                continue

            frame_idx += 1
            if config.SKIP_FRAMES > 1 and frame_idx % config.SKIP_FRAMES != 0:
                continue

            frame = cv2.resize(
                frame, (config.RESIZE_WIDTH, config.RESIZE_HEIGHT),
                interpolation=cv2.INTER_AREA
            )

            results = model.track(
                source=frame,
                tracker=config.TRACKER_CONFIG,
                persist=True,
                conf=config.CONF_THRESHOLD,
                verbose=False
            )
            
            result = results[0]

            
            if result.boxes is not None:
                for box in result.boxes:
                    class_id = int(box.cls[0])
                    confidence = float(box.conf[0])
                    class_name = result.names[class_id]

                #ตรวจสอบเฉพาะพฤติกรรมที่ต้องสงสัย
                    if class_name in ['Suspicious', 'rob', 'stealing']:
                        current_time = time.time()
                        last_time = last_alert_time.get(class_name, 0)
                        
                        if current_time - last_time >= alert_interval:
                            alert_time = datetime.now().strftime("%d/%m/%Y %H:%M:%S")
                            message = (
                                "ระบบตรวจพบพฤติกรรมต้องสงสัย: "
                                f"{class_name}\n"
                                f"ความมั่นใจ: {confidence:.2f}\n"
                                f"เวลา: {alert_time}"
                            )
                            image_path = "alert_image.jpg"  # กำหนด path ของภาพที่ต้องการส่ง
                            cv2.imwrite(image_path, frame)  # บันทึกภาพจากกล้องเป็นไฟล์
                            try:
                                send_line_message(message, image_path=image_path)
                            finally:
                                import os
                                if os.path.exists(image_path):
                                    os.remove(image_path)  # ลบไฟล์ภาพหลังส่งข้อความ
                            last_alert_time[class_name] = current_time
                    
            annotated = draw_detections_highest_confidence(results[0], frame.copy())

            now = time.time()
            actual_fps = 1.0 / (now - prev_time) if now > prev_time else 0.0
            prev_time = now
            cv2.putText(
                annotated, f"FPS: {actual_fps:.1f}", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2
            )

            # if writer is not None:
            #     writer.write(annotated)

            if config.SHOW_WINDOW:
                cv2.imshow(config.WINDOW_NAME, annotated)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    print("หยุดการทำงาน")
                    break

    except KeyboardInterrupt:
        print("หยุดโปรแกรมด้วย Ctrl+C")
    finally:
        cap.release()
        # if writer is not None:
        #     writer.release()
        #     print(f"บันทึกวิดีโอที่: {config.OUTPUT_VIDEO_PATH}")
        # cv2.destroyAllWindows()


if __name__ == "__main__":
    main()