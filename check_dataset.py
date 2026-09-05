import os

dataset_path = r"C:\Users\kunal\.cache\kagglehub\datasets\prathumarikeri\indian-sign-language-isl\versions\1"

for root, dirs, files in os.walk(dataset_path):

    if files:
        jpg_files = [
            file for file in files
            if file.lower().endswith((".jpg", ".jpeg", ".png"))
        ]

        if jpg_files:
            print(root)
            print("Images:", len(jpg_files))
            print("Example:", jpg_files[:3])
            print()