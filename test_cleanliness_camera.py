import cv2
from ultralytics import YOLO

MODEL = r"runs\classify\runs\classify\cleanliness_baseline\weights\best.pt"

print("Loading cleanliness model...")
model = YOLO(MODEL)
print("Model loaded.")

# --------------------------------------------------
# Find a working camera automatically
# --------------------------------------------------

cap = None
camera_id = None

for cam_id in range(5):
    print(f"Testing camera {cam_id}...")

    test_cap = cv2.VideoCapture(cam_id, cv2.CAP_DSHOW)

    if not test_cap.isOpened():
        print(f"  Camera {cam_id}: could not open")
        test_cap.release()
        continue

    ret, frame = test_cap.read()

    if not ret or frame is None:
        print(f"  Camera {cam_id}: opened but no valid frame")
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
    print("Check that the USB camera is connected.")
    raise SystemExit

print()
print(f"Using camera: {camera_id}")

# Actual resolution supplied by camera
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

print(f"Camera resolution: {width} x {height}")
print("Press Q to quit.")

# --------------------------------------------------
# Camera + classifier
# --------------------------------------------------

while True:

    ret, frame = cap.read()

    if not ret or frame is None:
        print("Warning: invalid camera frame")
        continue

    # Actual frame resolution
    h, w = frame.shape[:2]

    # Run classifier
    results = model.predict(
        frame,
        device=0,
        verbose=False
    )

    result = results[0]

    class_id = result.probs.top1
    confidence = float(result.probs.top1conf)
    label = result.names[class_id]

    text = f"{label.upper()}  {confidence * 100:.1f}%"
    resolution_text = f"Camera: {w} x {h}"

    cv2.putText(
        frame,
        text,
        (20, 50),
        cv2.FONT_HERSHEY_SIMPLEX,
        1.2,
        (0, 255, 0),
        3
    )

    cv2.putText(
        frame,
        resolution_text,
        (20, 90),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (0, 255, 255),
        2
    )

    cv2.imshow("Divider Cleanliness Test", frame)

    key = cv2.waitKey(1) & 0xFF

    if key == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()

print("Camera test finished.")