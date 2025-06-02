from runpod.serverless import start

def handler(event):
    print("✅ Event received:", event)
    return [
        {
            "id": "1",
            "name": "Debug Bot",
            "age": 23,
            "gender": "Female",
            "image_url": "https://via.placeholder.com/150",
            "personalityDescription": "Testing RunPod output.",
        }
    ]

start({"handler": handler})