import cv2

for camera_id in range(5):
    print(f"\nTesting camera {camera_id}...")
    
    cap = cv2.VideoCapture(camera_id, cv2.CAP_DSHOW)

    if not cap.isOpened():
        print("  Could not open")
        cap.release()
        continue

    ret, frame = cap.read()

    print("  Opened:", True)
    print("  Frame received:", ret)

    if ret and frame is not None:
        print("  Frame size:", frame.shape)

        cv2.imshow(f"Camera {camera_id}", frame)
        cv2.waitKey(1500)
        cv2.destroyAllWindows()

    cap.release()

print("\nCamera scan finished.")