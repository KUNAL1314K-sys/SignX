import cv2
import torch
import torch.nn as nn
from torchvision import transforms


# ================= MODEL =================

class AlphabetModel(nn.Module):

    def __init__(self):
        super().__init__()

        self.network = nn.Sequential(

            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Flatten(),

            nn.Linear(128 * 8 * 8, 256),
            nn.ReLU(),

            nn.Dropout(0.3),

            nn.Linear(256, 26)
        )

    def forward(self, x):
        return self.network(x)


# ================= LOAD MODEL =================

device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

checkpoint = torch.load(
    "isl_alphabet_model.pth",
    map_location=device
)

classes = checkpoint["classes"]

model = AlphabetModel().to(device)

model.load_state_dict(
    checkpoint["model_state"]
)

model.eval()


# ================= IMAGE TRANSFORM =================

transform = transforms.Compose([

    transforms.ToPILImage(),

    transforms.Resize((64, 64)),

    transforms.ToTensor(),

    transforms.Normalize(
        mean=[0.5, 0.5, 0.5],
        std=[0.5, 0.5, 0.5]
    )
])


# ================= CAMERA =================

cap = cv2.VideoCapture(0)


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


    # Prepare image
    image = transform(rgb_frame)

    image = image.unsqueeze(0).to(device)


    # ================= PREDICTION =================

    with torch.no_grad():

        output = model(image)

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


    # ================= DISPLAY =================

    cv2.putText(
        frame,
        f"Letter: {predicted_letter}",
        (30, 50),
        cv2.FONT_HERSHEY_SIMPLEX,
        1.5,
        (0, 255, 0),
        3
    )

    cv2.putText(
        frame,
        f"Confidence: {confidence_value:.1f}%",
        (30, 100),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (0, 255, 255),
        2
    )


    cv2.imshow(
        "ISL Alphabet Recognition",
        frame
    )


    # Press Q to quit
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break


cap.release()

cv2.destroyAllWindows()