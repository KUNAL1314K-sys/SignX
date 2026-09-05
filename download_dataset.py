import kagglehub

# Download dataset
path = kagglehub.dataset_download(
    "prathumarikeri/indian-sign-language-isl"
)

print("Path to dataset files:", path)