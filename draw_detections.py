import cv2
import numpy as np

# ---------- ค่าคงที่สำหรับการวาด (แก้ที่เดียว มีผลทั้งฟังก์ชัน) ----------
BOX_COLOR = (0, 255, 0)      # BGR: เขียว
TEXT_COLOR = (0, 0, 0)       # BGR: ดำ
FONT = cv2.FONT_HERSHEY_SIMPLEX
FONT_SCALE = 0.55
FONT_THICKNESS = 2
BOX_THICKNESS = 2
LABEL_PAD = 4                # padding ของป้าย (ต่อด้าน)
INSIDE_OFFSET = 3            # ระยะห่างจากขอบบนกรอบ เมื่อต้องวางป้ายไว้ด้านใน


def _rects_overlap(a, b):
    """เช็กว่าสี่เหลี่ยม (x1, y1, x2, y2) สองอันทับกันหรือไม่"""
    return a[0] < b[2] and a[2] > b[0] and a[1] < b[3] and a[3] > b[1]


def _calc_label_rect(x1, y1, text_w, text_h, baseline, frame_w, frame_h):
    """คำนวณตำแหน่งป้าย: วางเหนือกรอบ ถ้าไม่พอให้ย้ายเข้าไปด้านในกรอบ"""
    label_w = text_w + 2 * LABEL_PAD
    label_h = text_h + baseline + 2 * LABEL_PAD

    lx1 = max(0, min(x1, frame_w - label_w))   # กันล้นขอบซ้าย/ขวา
    ly1 = y1 - label_h                          # วางเหนือกรอบ

    if ly1 < 0:                                 # ล้นขอบบน -> ย้ายเข้าด้านใน
        ly1 = min(frame_h - label_h, y1 + INSIDE_OFFSET)

    return lx1, ly1, lx1 + label_w, ly1 + label_h


def draw_detections_highest_confidence(result, frame):
    """วาดกรอบทุกวัตถุ และวาดป้ายตามลำดับ confidence สูง -> ต่ำ
    (ป้ายที่ทับกับป้ายที่ confidence สูงกว่าจะถูกข้าม)
    แก้ไข frame ในที่เดิม และคืน frame กลับมา
    """
    boxes = result.boxes
    if boxes is None or len(boxes) == 0:
        return frame

    # แปลง tensor เป็น list ทีเดียวต่อฟิลด์ (เร็วกว่าวนดึงทีละ box
    # โดยเฉพาะเมื่อโมเดลรันบน GPU เพราะลดจำนวนครั้งที่ต้อง sync)
    coords = boxes.xyxy.cpu().numpy().astype(int).tolist()
    confs = boxes.conf.cpu().numpy().tolist()
    class_ids = boxes.cls.cpu().numpy().astype(int).tolist()
    track_ids = (
        boxes.id.cpu().numpy().astype(int).tolist()
        if boxes.id is not None else None
    )

    names = result.names
    frame_h, frame_w = frame.shape[:2]

    # 1) วาดกรอบของทุกวัตถุ
    for x1, y1, x2, y2 in coords:
        cv2.rectangle(frame, (x1, y1), (x2, y2), BOX_COLOR, BOX_THICKNESS)

    # 2) เรียงลำดับ index ตาม confidence มาก -> น้อย
    order = sorted(range(len(confs)), key=confs.__getitem__, reverse=True)

    # 3) วาดป้ายตามลำดับ โดยข้ามป้ายที่ทับกัน
    placed_rects = []
    for i in order:
        label = f"{names.get(class_ids[i], class_ids[i])} {confs[i]:.2f}"
        if track_ids is not None:
            label = f"ID:{track_ids[i]} {label}"

        (text_w, text_h), baseline = cv2.getTextSize(
            label, FONT, FONT_SCALE, FONT_THICKNESS
        )
        x1, y1 = coords[i][0], coords[i][1]
        rect = _calc_label_rect(
            x1, y1, text_w, text_h, baseline, frame_w, frame_h
        )

        if any(_rects_overlap(rect, old) for old in placed_rects):
            continue
        placed_rects.append(rect)

        lx1, ly1, lx2, ly2 = rect
        cv2.rectangle(frame, (lx1, ly1), (lx2, ly2), BOX_COLOR, -1)
        cv2.putText(
            frame, label,
            (lx1 + LABEL_PAD, ly2 - baseline - LABEL_PAD),
            FONT, FONT_SCALE, TEXT_COLOR, FONT_THICKNESS, cv2.LINE_AA,
        )

    return frame
