import os
import torch
import torch.nn as nn
import torch.optim as optim

from torchvision import datasets, transforms
from torch.utils.data import DataLoader, random_split


# ================= DATASET PATH =================

dataset_path = r"C:\Users\kunal\.cache\kagglehub\datasets\prathumarikeri\indian-sign-language-isl\versions\1\Indian"


# ================= IMAGE TRANSFORM =================

transform = transforms.Compose([

    transforms.Resize((64, 64)),

    transforms.ToTensor(),

    transforms.Normalize(
        mean=[0.5, 0.5, 0.5],
        std=[0.5, 0.5, 0.5]
    )

])


# ================= LOAD DATASET =================

dataset = datasets.ImageFolder(
    dataset_path,
    transform=transform
)


# Keep only alphabet folders

alphabet_indices = []

for index, class_name in enumerate(dataset.classes):

    if class_name.isalpha():

        alphabet_indices.append(index)


# Filter alphabet samples

dataset.samples = [

    sample for sample in dataset.samples
    if dataset.classes[sample[1]].isalpha()

]

dataset.targets = [

    target for _, target in dataset.samples

]


# ================= FIX LABELS =================

alphabet_classes = sorted([

    name for name in dataset.classes
    if name.isalpha()

])

class_to_new_index = {

    name: index
    for index, name in enumerate(alphabet_classes)

}


new_samples = []

for path, old_label in dataset.samples:

    class_name = dataset.classes[old_label]

    new_label = class_to_new_index[class_name]

    new_samples.append(
        (path, new_label)
    )


dataset.samples = new_samples

dataset.classes = alphabet_classes

dataset.class_to_idx = class_to_new_index

dataset.targets = [

    label for _, label in dataset.samples
]


print("Classes:", dataset.classes)
print("Number of images:", len(dataset))


# ================= TRAIN / TEST SPLIT =================

train_size = int(
    0.8 * len(dataset)
)

test_size = len(dataset) - train_size


train_dataset, test_dataset = random_split(
    dataset,
    [train_size, test_size]
)


train_loader = DataLoader(
    train_dataset,
    batch_size=32,
    shuffle=True,
    num_workers=0
)

test_loader = DataLoader(
    test_dataset,
    batch_size=32,
    shuffle=False,
    num_workers=0
)


# ================= CNN MODEL =================

class AlphabetModel(nn.Module):

    def __init__(self):

        super().__init__()

        self.network = nn.Sequential(

            nn.Conv2d(
                3,
                32,
                kernel_size=3,
                padding=1
            ),

            nn.ReLU(),

            nn.MaxPool2d(2),


            nn.Conv2d(
                32,
                64,
                kernel_size=3,
                padding=1
            ),

            nn.ReLU(),

            nn.MaxPool2d(2),


            nn.Conv2d(
                64,
                128,
                kernel_size=3,
                padding=1
            ),

            nn.ReLU(),

            nn.MaxPool2d(2),


            nn.Flatten(),


            nn.Linear(
                128 * 8 * 8,
                256
            ),

            nn.ReLU(),

            nn.Dropout(0.3),


            nn.Linear(
                256,
                26
            )
        )


    def forward(self, x):

        return self.network(x)


# ================= MODEL =================

device = torch.device(

    "cuda"
    if torch.cuda.is_available()
    else "cpu"
)

print("Using device:", device)


model = AlphabetModel().to(device)


criterion = nn.CrossEntropyLoss()

optimizer = optim.Adam(
    model.parameters(),
    lr=0.001
)


# ================= TRAIN =================

epochs = 10


for epoch in range(epochs):

    model.train()

    correct = 0
    total = 0
    running_loss = 0


    for images, labels in train_loader:

        images = images.to(device)

        labels = labels.to(device)


        optimizer.zero_grad()


        outputs = model(images)


        loss = criterion(
            outputs,
            labels
        )


        loss.backward()

        optimizer.step()


        running_loss += loss.item()


        _, predicted = torch.max(
            outputs,
            1
        )


        total += labels.size(0)


        correct += (

            predicted == labels

        ).sum().item()


    accuracy = (

        100 * correct / total

    )


    print(

        f"Epoch {epoch + 1}/{epochs} | "
        f"Loss: {running_loss:.4f} | "
        f"Train Accuracy: {accuracy:.2f}%"

    )


# ================= TEST =================

model.eval()

correct = 0
total = 0


with torch.no_grad():

    for images, labels in test_loader:

        images = images.to(device)

        labels = labels.to(device)


        outputs = model(images)


        _, predicted = torch.max(
            outputs,
            1
        )


        total += labels.size(0)


        correct += (

            predicted == labels

        ).sum().item()


accuracy = (

    100 * correct / total

)


print(
    f"\nTest Accuracy: {accuracy:.2f}%"
)


# ================= SAVE MODEL =================

torch.save(
    {
        "model_state": model.state_dict(),
        "classes": dataset.classes
    },
    "isl_alphabet_model.pth"
)


print(
    "\nModel saved: isl_alphabet_model.pth"
)