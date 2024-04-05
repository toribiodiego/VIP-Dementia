import os

corrected_paths = [
    "/Users/Work/PycharmProjects/VIP/pipeline/data/raw/audio/ad/adrso024.wav",
    "/Users/Work/PycharmProjects/VIP/pipeline/data/raw/audio/cn/adrso002.wav"
]

print("Current Working Directory:", os.getcwd())

for path in corrected_paths:
    print(f"Testing access to: {path}")
    if os.path.exists(path):
        print("Success: File exists.")
    else:
        print(f"Failure: File does not exist. Attempted path: {path}")