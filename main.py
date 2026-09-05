import cv2
import mediapipe as mp
import time


# ================= MEDIAPIPE =================

BaseOptions = mp.tasks.BaseOptions
VisionRunningMode = mp.tasks.vision.RunningMode


# ================= HAND =================

HandLandmarker = mp.tasks.vision.HandLandmarker
HandLandmarkerOptions = mp.tasks.vision.HandLandmarkerOptions

hand_options = HandLandmarkerOptions(
    base_options=BaseOptions(
        model_asset_path="hand_landmarker.task"
    ),
    running_mode=VisionRunningMode.VIDEO,
    num_hands=2
)

hand_detector = HandLandmarker.create_from_options(hand_options)


# ================= POSE =================

PoseLandmarker = mp.tasks.vision.PoseLandmarker
PoseLandmarkerOptions = mp.tasks.vision.PoseLandmarkerOptions

pose_options = PoseLandmarkerOptions(
    base_options=BaseOptions(
        model_asset_path="pose_landmarker_full.task"
    ),
    running_mode=VisionRunningMode.VIDEO,
    num_poses=1
)

pose_detector = PoseLandmarker.create_from_options(pose_options)


# ================= FACE =================

FaceLandmarker = mp.tasks.vision.FaceLandmarker
FaceLandmarkerOptions = mp.tasks.vision.FaceLandmarkerOptions

face_options = FaceLandmarkerOptions(
    base_options=BaseOptions(
        model_asset_path="face_landmarker.task"
    ),
    running_mode=VisionRunningMode.VIDEO,
    num_faces=1
)

face_detector = FaceLandmarker.create_from_options(face_options)


# ================= CAMERA =================

cap = cv2.VideoCapture(0)

start_time = time.time()


while True:

    success, frame = cap.read()

    if not success:
        print("Camera not found")
        break


    # Flip camera
    frame = cv2.flip(frame, 1)

    # Convert BGR -> RGB
    rgb_frame = cv2.cvtColor(
        frame,
        cv2.COLOR_BGR2RGB
    )

    # Create MediaPipe image
    mp_image = mp.Image(
        image_format=mp.ImageFormat.SRGB,
        data=rgb_frame
    )

    # Timestamp in milliseconds
    timestamp = int(
        (time.time() - start_time) * 1000
    )


    # ================= DETECT =================

    hand_result = hand_detector.detect_for_video(
        mp_image,
        timestamp
    )

    pose_result = pose_detector.detect_for_video(
        mp_image,
        timestamp
    )

    face_result = face_detector.detect_for_video(
        mp_image,
        timestamp
    )


    # ================= DRAW HANDS =================

    for hand_landmarks in hand_result.hand_landmarks:

        for landmark in hand_landmarks:

            x = int(landmark.x * frame.shape[1])
            y = int(landmark.y * frame.shape[0])

            cv2.circle(
                frame,
                (x, y),
                4,
                (0, 255, 0),
                -1
            )


    # ================= DRAW POSE =================

    for pose_landmarks in pose_result.pose_landmarks:

        for landmark in pose_landmarks:

            x = int(landmark.x * frame.shape[1])
            y = int(landmark.y * frame.shape[0])

            cv2.circle(
                frame,
                (x, y),
                4,
                (255, 0, 0),
                -1
            )


    # ================= DRAW FACE =================

    for face_landmarks in face_result.face_landmarks:

        for landmark in face_landmarks:

            x = int(landmark.x * frame.shape[1])
            y = int(landmark.y * frame.shape[0])

            cv2.circle(
                frame,
                (x, y),
                1,
                (0, 0, 255),
                -1
            )


    # ================= TEXT =================

    cv2.putText(
        frame,
        f"Hands: {len(hand_result.hand_landmarks)}",
        (20, 40),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (0, 255, 0),
        2
    )

    cv2.putText(
        frame,
        f"Pose: {len(pose_result.pose_landmarks)}",
        (20, 70),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (255, 0, 0),
        2
    )

    cv2.putText(
        frame,
        f"Face: {len(face_result.face_landmarks)}",
        (20, 100),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (0, 0, 255),
        2
    )


    # Show result
    cv2.imshow(
        "ISL Full Landmark Detection",
        frame
    )


    # Press Q to quit
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break


# ================= CLEANUP =================

cap.release()

hand_detector.close()
pose_detector.close()
face_detector.close()

cv2.destroyAllWindows()