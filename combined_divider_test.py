import cv2
from ultralytics import YOLO

# ============================================================
# MODELS
# ============================================================

DETECTOR_PATH = r"runs\detect\train\weights\best.pt"
CLASSIFIER_PATH = r"runs\classify\runs\classify\cleanliness_baseline\weights\best.pt"

print("Loading divider detector...")
detector = YOLO(DETECTOR_PATH)

print("Loading cleanliness classifier...")
classifier = YOLO(CLASSIFIER_PATH)

print("Both models loaded successfully.")


# ============================================================
# FIND WORKING CAMERA
# ============================================================

cap = None
camera_id = None

for cam_id in range(5):

    print(f"Testing camera {cam_id}...")

    test_cap = cv2.VideoCapture(cam_id, cv2.CAP_DSHOW)

    if not test_cap.isOpened():
        print(f"  Camera {cam_id}: cannot open")
        test_cap.release()
        continue

    ret, frame = test_cap.read()

    if not ret or frame is None:
        print(f"  Camera {cam_id}: no valid frame")
        test_cap.release()
        continue

    h, w = frame.shape[:2]

    print(f"  Camera {cam_id}: WORKING")
    print(f"  Resolution: {w} x {h}")

    cap = test_cap
    camera_id = cam_id
    break


if cap is None:
    print("\nERROR: No working camera found.")
    raise SystemExit


print()
print(f"Using camera: {camera_id}")
print("Press Q to quit.")


# ============================================================
# LIVE LOOP
# ============================================================

while True:

    ret, frame = cap.read()

    if not ret or frame is None:
        continue

    display = frame.copy()

    # --------------------------------------------------------
    # STEP 1: DETECT DIVIDER
    # --------------------------------------------------------

    detections = detector.predict(
        frame,
        imgsz=640,
        conf=0.15,
        device=0,
        verbose=False
    )[0]

    if len(detections.boxes) == 0:

        cv2.putText(
            display,
            "DIVIDER NOT DETECTED",
            (20, 45),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.0,
            (0, 0, 255),
            3
        )

    else:

        # ----------------------------------------------------
        # Select largest detected divider
        # ----------------------------------------------------

        best_box = None
        best_area = 0

        for box in detections.boxes:

            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()

            x1 = int(max(0, x1))
            y1 = int(max(0, y1))
            x2 = int(min(frame.shape[1], x2))
            y2 = int(min(frame.shape[0], y2))

            area = (x2 - x1) * (y2 - y1)

            if area > best_area:
                best_area = area
                best_box = (x1, y1, x2, y2, float(box.conf[0]))

        x1, y1, x2, y2, detector_conf = best_box

        # ----------------------------------------------------
        # STEP 2: CROP DIVIDER
        # ----------------------------------------------------

        divider_crop = frame[y1:y2, x1:x2]

        if divider_crop.size == 0:

            cv2.putText(
                display,
                "INVALID DIVIDER CROP",
                (20, 45),
                cv2.FONT_HERSHEY_SIMPLEX,
                1.0,
                (0, 0, 255),
                3
            )

        else:

            # ------------------------------------------------
            # STEP 3: CLASSIFY CLEANLINESS
            # ------------------------------------------------

            classification = classifier.predict(
                divider_crop,
                device=0,
                verbose=False
            )[0]

            class_id = classification.probs.top1
            confidence = float(classification.probs.top1conf)

            label = classification.names[class_id]

            # ------------------------------------------------
            # DRAW DETECTION BOX
            # ------------------------------------------------

            cv2.rectangle(
                display,
                (x1, y1),
                (x2, y2),
                (0, 255, 0),
                3
            )

            # ------------------------------------------------
            # DISPLAY RESULT
            # ------------------------------------------------

            result_text = f"{label.upper()}  {confidence * 100:.1f}%"
            detector_text = f"Divider detection: {detector_conf * 100:.1f}%"

            cv2.putText(
                display,
                result_text,
                (x1, max(40, y1 - 15)),
                cv2.FONT_HERSHEY_SIMPLEX,
                1.0,
                (0, 255, 0),
                3
            )

            cv2.putText(
                display,
                detector_text,
                (20, 35),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.65,
                (255, 255, 0),
                2
            )

    # --------------------------------------------------------
    # CAMERA RESOLUTION
    # --------------------------------------------------------

    h, w = frame.shape[:2]

    cv2.putText(
        display,
        f"Camera: {w} x {h}",
        (20, h - 20),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (255, 255, 255),
        2
    )

    cv2.imshow("SWACH - Divider Cleanliness", display)

    key = cv2.waitKey(1) & 0xFF

    if key == ord("q"):
        break


# ============================================================
# CLEANUP
# ============================================================

cap.release()
cv2.destroyAllWindows()

print("Combined test finished.")