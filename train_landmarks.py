import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader, random_split


# ================= DATA PATH =================

DATA_PATH = "landmark_dataset"


# ================= LOAD CLASSES =================

classes = np.load(
    f"{DATA_PATH}/classes.npy",
    allow_pickle=True
)

print("Classes:", classes)


# ================= LOAD ALL DATA =================

X_list = []
y_list = []


for class_name in classes:

    X_file = f"{DATA_PATH}/{class_name}_X.npy"
    y_file = f"{DATA_PATH}/{class_name}_y.npy"

    print(f"Loading {class_name}...")

    X = np.load(X_file)
    y = np.load(y_file)

    X_list.append(X)
    y_list.append(y)


# Combine everything

X = np.concatenate(X_list, axis=0)
y = np.concatenate(y_list, axis=0)


print("\nTotal samples:", len(X))
print("Feature shape:", X.shape)


# ================= CONVERT TO TENSORS =================

X = torch.tensor(
    X,
    dtype=torch.float32
)

y = torch.tensor(
    y,
    dtype=torch.long
)


dataset = TensorDataset(X, y)


# ================= TRAIN TEST SPLIT =================

train_size = int(
    len(dataset) * 0.8
)

test_size = len(dataset) - train_size


train_dataset, test_dataset = random_split(

    dataset,

    [
        train_size,
        test_size
    ]
)


# ================= DATA LOADERS =================

train_loader = DataLoader(

    train_dataset,

    batch_size=128,

    shuffle=True
)


test_loader = DataLoader(

    test_dataset,

    batch_size=128,

    shuffle=False
)


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


# ================= DEVICE =================

device = torch.device(

    "cuda"

    if torch.cuda.is_available()

    else "cpu"
)


print("\nUsing device:", device)


# ================= CREATE MODEL =================

model = LandmarkModel().to(device)


loss_function = nn.CrossEntropyLoss()


optimizer = torch.optim.Adam(

    model.parameters(),

    lr=0.001
)


# ================= TRAIN =================

epochs = 30


for epoch in range(epochs):


    model.train()


    total_loss = 0


    for inputs, labels in train_loader:


        inputs = inputs.to(device)

        labels = labels.to(device)


        optimizer.zero_grad()


        outputs = model(inputs)


        loss = loss_function(
            outputs,
            labels
        )


        loss.backward()


        optimizer.step()


        total_loss += loss.item()


    average_loss = (

        total_loss / len(train_loader)

    )


    print(

        f"Epoch {epoch + 1}/{epochs} "

        f"| Loss: {average_loss:.4f}"
    )


# ================= TEST =================

print("\nTesting model...")


model.eval()


correct = 0
total = 0


with torch.no_grad():

    for inputs, labels in test_loader:


        inputs = inputs.to(device)

        labels = labels.to(device)


        outputs = model(inputs)


        _, predicted = torch.max(
            outputs,
            1
        )


        total += labels.size(0)


        correct += (

            predicted == labels

        ).sum().item()


accuracy = (

    correct / total

) * 100


print(

    f"\nTest Accuracy: {accuracy:.2f}%"
)


# ================= SAVE MODEL =================

torch.save(

    {

        "model_state": model.state_dict(),

        "classes": classes.tolist()

    },

    "isl_landmark_model.pth"
)


print(
    "\nModel successfully saved!"
)

print(
    "File: isl_landmark_model.pth"
)