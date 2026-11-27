import cv2
import numpy as np
from numpy.typing import NDArray
from ultralytics.engine.results import Results

# ค่าคงที่สำหรับการวาด
BOX_COLOR = (0, 255, 0)
TEXT_COLOR = (0, 0, 0)
FONT = cv2.FONT_HERSHEY_SIMPLEX
FONT_SCALE = 0.55
FONT_THICKNESS = 2
BOX_THICKNESS = 2
LABEL_PAD = 4
INSIDE_OFFSET = 3


def _rects_overlap(
    a: tuple[int, int, int, int],
    b: tuple[int, int, int, int],
) -> bool:
    """เช็กว่าสี่เหลี่ยม (x1, y1, x2, y2) สองอันทับกันหรือไม่"""
    a_x1, a_y1, a_x2, a_y2 = a
    b_x1, b_y1, b_x2, b_y2 = b

    return (
        a_x1 < b_x2
        and a_x2 > b_x1
        and a_y1 < b_y2
        and a_y2 > b_y1
    )


def _calc_label_rect(
    x1: int,
    y1: int,
    text_w: int,
    text_h: int,
    baseline: int,
    frame_w: int,
    frame_h: int,
) -> tuple[int, int, int, int]:
    """คำนวณตำแหน่งป้ายเหนือกรอบ พร้อมย้ายเข้าในกรอบเมื่อล้นขอบบน"""
    label_w = text_w + 2 * LABEL_PAD
    label_h = text_h + baseline + 2 * LABEL_PAD

    lx1 = max(0, min(x1, frame_w - label_w))
    ly1 = y1 - label_h

    if ly1 < 0:
        ly1 = min(frame_h - label_h, y1 + INSIDE_OFFSET)

    return lx1, ly1, lx1 + label_w, ly1 + label_h


def _build_label(
    names: dict[int, str],
    class_id: int,
    confidence: float,
    track_id: int | None,
) -> str:
    """สร้างข้อความป้ายสำหรับวัตถุหนึ่งชิ้น"""
    class_name = names.get(class_id, str(class_id))
    label = f"{class_name} {confidence:.2f}"

    if track_id is not None:
        label = f"ID:{track_id} {label}"

    return label


def _draw_label(
    frame: NDArray[np.uint8],
    label: str,
    rect: tuple[int, int, int, int],
    baseline: int,
) -> None:
    """วาดพื้นป้ายและข้อความลงใน frame"""
    lx1, ly1, lx2, ly2 = rect
    cv2.rectangle(frame, (lx1, ly1), (lx2, ly2), BOX_COLOR, -1)

    text_x = lx1 + LABEL_PAD
    text_y = ly2 - baseline - LABEL_PAD
    cv2.putText(
        frame,
        label,
        (text_x, text_y),
        FONT,
        FONT_SCALE,
        TEXT_COLOR,
        FONT_THICKNESS,
        cv2.LINE_AA,
    )


def draw_detections_highest_confidence(
    result: Results,
    frame: NDArray[np.uint8],
) -> NDArray[np.uint8]:
    """วาดกรอบทุกวัตถุ และวาดป้ายตามลำดับ confidence สูง -> ต่ำ
    (ป้ายที่ทับกับป้ายที่ confidence สูงกว่าจะถูกข้าม)
    แก้ไข frame ในที่เดิม และคืน frame กลับมา
    """
    boxes = result.boxes
    if boxes is None or len(boxes) == 0:
        return frame

    box_coords = boxes.xyxy.cpu().numpy().astype(int).tolist()
    confidences = boxes.conf.cpu().numpy().tolist()
    class_ids = boxes.cls.cpu().numpy().astype(int).tolist()
    track_ids = (
        boxes.id.cpu().numpy().astype(int).tolist()
        if boxes.id is not None else None
    )

    names = result.names
    frame_h, frame_w = frame.shape[:2]

    for x1, y1, x2, y2 in box_coords:
        cv2.rectangle(frame, (x1, y1), (x2, y2), BOX_COLOR, BOX_THICKNESS)

    confidence_order = sorted(
        range(len(confidences)),
        key=confidences.__getitem__,
        reverse=True,
    )

    placed_label_rects: list[tuple[int, int, int, int]] = []
    for index in confidence_order:
        track_id = track_ids[index] if track_ids is not None else None
        label = _build_label(
            names,
            class_ids[index],
            confidences[index],
            track_id,
        )

        (text_w, text_h), baseline = cv2.getTextSize(
            label, FONT, FONT_SCALE, FONT_THICKNESS
        )
        x1, y1 = box_coords[index][0], box_coords[index][1]
        label_rect = _calc_label_rect(
            x1,
            y1,
            text_w,
            text_h,
            baseline,
            frame_w,
            frame_h,
        )

        overlaps_placed_label = any(
            _rects_overlap(label_rect, placed_rect)
            for placed_rect in placed_label_rects
        )
        if overlaps_placed_label:
            continue
        placed_label_rects.append(label_rect)
        _draw_label(frame, label, label_rect, baseline)

    return frame
