import cv2
import mediapipe as mp
import os
import numpy as np


# ================= PATH =================

INPUT_PATH = r"C:\Users\kunal\.cache\kagglehub\datasets\prathumarikeri\indian-sign-language-isl\versions\1\Indian"

OUTPUT_PATH = "landmark_dataset"


# ================= MEDIAPIPE =================

BaseOptions = mp.tasks.BaseOptions
HandLandmarker = mp.tasks.vision.HandLandmarker
HandLandmarkerOptions = mp.tasks.vision.HandLandmarkerOptions
VisionRunningMode = mp.tasks.vision.RunningMode


options = HandLandmarkerOptions(

    base_options=BaseOptions(
        model_asset_path="hand_landmarker.task"
    ),

    running_mode=VisionRunningMode.IMAGE,

    num_hands=2,

    min_hand_detection_confidence=0.5
)


landmarker = HandLandmarker.create_from_options(options)


# ================= CREATE OUTPUT =================

os.makedirs(OUTPUT_PATH, exist_ok=True)


classes = sorted(os.listdir(INPUT_PATH))


# Only A-Z
classes = [
    c for c in classes
    if c.isalpha() and len(c) == 1
]


print("Classes:", classes)


# ================= PROCESS DATASET =================

for class_index, class_name in enumerate(classes):

    input_folder = os.path.join(
        INPUT_PATH,
        class_name
    )


    print(f"\nProcessing {class_name}...")


    X = []
    y = []


    images = [

        file for file in os.listdir(input_folder)

        if file.lower().endswith(
            (".jpg", ".jpeg", ".png")
        )
    ]


    failed = 0


    for filename in images:


        image_path = os.path.join(
            input_folder,
            filename
        )


        image = cv2.imread(image_path)


        if image is None:

            failed += 1
            continue


        # Convert BGR → RGB
        rgb_image = cv2.cvtColor(
            image,
            cv2.COLOR_BGR2RGB
        )


        # MediaPipe Image
        mp_image = mp.Image(

            image_format=mp.ImageFormat.SRGB,

            data=rgb_image
        )


        # Detect hands
        result = landmarker.detect(
            mp_image
        )


        # ================= CREATE FIXED 126 FEATURES =================

        landmarks = []


        if result.hand_landmarks:


            # Use maximum 2 hands
            detected_hands = result.hand_landmarks[:2]


            for hand in detected_hands:

                for landmark in hand:

                    landmarks.extend([

                        landmark.x,

                        landmark.y,

                        landmark.z

                    ])


        # ================= FIX FEATURE SIZE =================

        # Maximum:
        # 2 hands × 21 landmarks × 3 values
        # = 126 values

        while len(landmarks) < 126:

            landmarks.append(0.0)


        # Skip unexpected data
        if len(landmarks) != 126:

            failed += 1
            continue


        X.append(landmarks)

        y.append(class_index)


    # ================= SAVE CLASS DATA =================

    X = np.array(
        X,
        dtype=np.float32
    )


    y = np.array(
        y,
        dtype=np.int64
    )


    np.save(

        os.path.join(
            OUTPUT_PATH,
            f"{class_name}_X.npy"
        ),

        X
    )


    np.save(

        os.path.join(
            OUTPUT_PATH,
            f"{class_name}_y.npy"
        ),

        y
    )


    print(

        f"Saved: {len(X)} | "
        f"Failed: {failed}"
    )


# ================= SAVE CLASSES =================

np.save(

    os.path.join(
        OUTPUT_PATH,
        "classes.npy"
    ),

    np.array(classes)
)


# ================= CLEANUP =================

landmarker.close()


print("\nDONE!")

print(
    "Landmark dataset saved in:",
    OUTPUT_PATH
)