import cv2
from ultralytics import YOLO
MODEL = r"C:\Users\Gajanan Ranmal\OneDrive\Desktop\SWACH\runs\detect\train\weights\best.pt"
# MODEL=r"C:\Users\Gajanan Ranmal\OneDrive\Desktop\divider cleaner\divider-cleaned-\runs\divider_final-3\weights\best.pt"

model = YOLO(MODEL)

# 1 = external USB camera
cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)

cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
cap.set(cv2.CAP_PROP_FPS, 30)

if not cap.isOpened():
    print("ERROR: External camera could not be opened.")
    raise SystemExit

print("Live YOLO26 demo started.")
print("Using external camera.")
print("Press Q to exit.")

while True:
    ret, frame = cap.read()

    if not ret:
        print("ERROR: Could not read camera frame.")
        break

    results = model.predict(
        source=frame,
        conf=0.25,
        verbose=False
    )

    annotated = results[0].plot()

    cv2.imshow(
        "SWACH - YOLO26 Divider Dirtiness Detection",
        annotated
    )

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()