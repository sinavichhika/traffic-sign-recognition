import torch
import cv2
import os
from pytube import YouTube
from torchvision import transforms

# Load the YOLOv5 model (assuming YOLOv5)
yolo_model = torch.hub.load('ultralytics/yolov5', 'yolov5m', pretrained=True)

# Load your trained ResNet-50 model
resnet50_model = torch.load('models/traffic_sign_recognition_resNet50_model.pth')
resnet50_model.eval()  # Set the model to evaluation mode

# Define image transformations for ResNet-50
transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

def download_youtube_video(url, output_path='videos'):
    yt = YouTube(url)
    stream = yt.streams.filter(file_extension='mp4').get_highest_resolution()
    output_file = stream.download(output_path)
    return output_file

def detect_and_recognize_from_video(video_path):
    # Open the video file
    cap = cv2.VideoCapture(video_path)
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        
        # Detect traffic signs with YOLOv5
        results = yolo_model(frame)
        
        for detection in results.xyxy[0]:  # Iterate through detections
            x1, y1, x2, y2, conf, cls = map(int, detection[:6])
            
            # Crop the detected traffic sign
            cropped_img = frame[y1:y2, x1:x2]
            
            # Preprocess the cropped image for ResNet-50
            processed_img = transform(cropped_img)
            processed_img = processed_img.unsqueeze(0)  # Add batch dimension
            
            # Predict the traffic sign using ResNet-50
            with torch.no_grad():
                prediction = resnet50_model(processed_img)
            
            # Process the output (e.g., get the predicted class)
            _, predicted_class = prediction.max(1)
            
            # Print or display the results
            label = f'Class: {predicted_class.item()} Conf: {conf:.2f}'
            print(f"Detected traffic sign with confidence {conf:.2f}. Recognized as class: {predicted_class.item()}")

            # Draw the bounding box and label on the frame
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (36, 255, 12), 2)
        
        # Display the frame with detections
        cv2.imshow('Detected and Recognized', frame)
        
        # Press 'q' to exit the video
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    # Release the video capture object and close display window
    cap.release()
    cv2.destroyAllWindows()

# Example usage
if __name__ == "__main__":
    # Provide the YouTube video URL
    video_url = 'https://www.youtube.com/watch?v=lZ_wmm6Ubik'
    
    # Download the video
    video_path = download_youtube_video(video_url)
    
    # Run detection and recognition on the downloaded video
    detect_and_recognize_from_video(video_path)
    
    # Optionally, clean up the downloaded video file
    os.remove(video_path)