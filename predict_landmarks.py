import cv2
import mediapipe as mp
import numpy as np
import torch
import torch.nn as nn
import time


# ================= MODEL =================

class LandmarkModel(nn.Module):

    def __init__(self):

        super().__init__()

        self.network = nn.Sequential(

            nn.Linear(126, 256),
            nn.ReLU(),
            nn.Dropout(0.3),

            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.3),

            nn.Linear(128, 64),
            nn.ReLU(),

            nn.Linear(64, 26)
        )

    def forward(self, x):

        return self.network(x)


# ================= LOAD MODEL =================

device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

checkpoint = torch.load(
    "isl_landmark_model.pth",
    map_location=device
)

classes = checkpoint["classes"]

model = LandmarkModel().to(device)

model.load_state_dict(
    checkpoint["model_state"]
)

model.eval()

print("Model loaded!")
print("Classes:", classes)


# ================= MEDIAPIPE =================

BaseOptions = mp.tasks.BaseOptions
HandLandmarker = mp.tasks.vision.HandLandmarker
HandLandmarkerOptions = mp.tasks.vision.HandLandmarkerOptions
VisionRunningMode = mp.tasks.vision.RunningMode


options = HandLandmarkerOptions(

    base_options=BaseOptions(
        model_asset_path="hand_landmarker.task"
    ),

    running_mode=VisionRunningMode.VIDEO,

    # Detect maximum 2 hands
    num_hands=2,

    min_hand_detection_confidence=0.5
)


landmarker = HandLandmarker.create_from_options(
    options
)


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


    # Convert BGR → RGB
    rgb_frame = cv2.cvtColor(
        frame,
        cv2.COLOR_BGR2RGB
    )


    # Create MediaPipe image
    mp_image = mp.Image(
        image_format=mp.ImageFormat.SRGB,
        data=rgb_frame
    )


    # Timestamp
    timestamp = int(
        (time.time() - start_time) * 1000
    )


    # Detect hands
    result = landmarker.detect_for_video(
        mp_image,
        timestamp
    )


    # ================= CREATE 126 FEATURES =================

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


        # Pad missing hand with zeros
        while len(landmarks) < 126:

            landmarks.append(0.0)


        # Convert to NumPy
        landmarks = np.array(
            landmarks,
            dtype=np.float32
        )


        # ================= PREDICT =================

        input_data = torch.tensor(
            landmarks
        ).unsqueeze(0).to(device)


        with torch.no_grad():

            output = model(
                input_data
            )


            probabilities = torch.softmax(
                output,
                dim=1
            )


            confidence, prediction = torch.max(
                probabilities,
                1
            )


        predicted_letter = classes[
            prediction.item()
        ]


        confidence_value = (
            confidence.item() * 100
        )


        # ================= DRAW LANDMARKS =================

        h, w, _ = frame.shape


        for hand in result.hand_landmarks:

            for landmark in hand:

                x = int(
                    landmark.x * w
                )

                y = int(
                    landmark.y * h
                )


                cv2.circle(

                    frame,

                    (x, y),

                    5,

                    (0, 255, 0),

                    -1
                )


        # ================= DISPLAY =================

        cv2.putText(

            frame,

            f"Hands: {len(result.hand_landmarks)}",

            (30, 40),

            cv2.FONT_HERSHEY_SIMPLEX,

            0.8,

            (255, 255, 0),

            2
        )


        cv2.putText(

            frame,

            f"{predicted_letter} ({confidence_value:.1f}%)",

            (30, 90),

            cv2.FONT_HERSHEY_SIMPLEX,

            1.2,

            (0, 255, 0),

            3
        )


    else:

        cv2.putText(

            frame,

            "Hands not detected",

            (30, 50),

            cv2.FONT_HERSHEY_SIMPLEX,

            1,

            (0, 0, 255),

            2
        )


    # ================= SHOW CAMERA =================

    cv2.imshow(
        "ISL Landmark Recognition",
        frame
    )


    # Press Q to quit
    if cv2.waitKey(1) & 0xFF == ord("q"):

        break


# ================= CLEANUP =================

cap.release()

landmarker.close()

cv2.destroyAllWindows()