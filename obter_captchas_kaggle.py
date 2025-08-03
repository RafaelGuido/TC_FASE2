import kagglehub

# Download da última versão
path = kagglehub.dataset_download("fournierp/captcha-version-2-images")

print("Path to dataset files:", path)
