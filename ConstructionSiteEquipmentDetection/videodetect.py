import os
from PIL import Image
from ultralytics import YOLO
import cv2

video_name = "workers"
video_path = os.path.join('test_videos_input', f'{video_name}.mp4')
video_path_out = os.path.join('test_videos_output', f'{video_name}_out.mp4')

cap = cv2.VideoCapture(video_path)
fps = int(cap.get(cv2.CAP_PROP_FPS))  # Get frames per second
ret, frame = cap.read()
H, W, _ = frame.shape
out = cv2.VideoWriter(video_path_out, cv2.VideoWriter_fourcc(*'XVID'), fps, (W, H))

model_path = os.path.join('.', 'best.pt')

# Load a model
model = YOLO(model_path)  # load a custom model

threshold = 0.5
counter = 1
frame_count = 0

folder_path = f'./frames/{video_name}/'
os.makedirs(folder_path, exist_ok=True)

while ret:
    try:
    
        results = model(frame)[0]
        
        if frame_count % fps == 0:  # Save annotated frame as an image once per second
            im_array = results.plot()  # plot a BGR numpy array of predictions
            im = Image.fromarray(im_array[..., ::-1])  # RGB PIL image
            im.save(f'./frames/{video_name}/{video_name}_results_{counter}.jpg')  # save image
            counter += 1
        
        for result in results.boxes.data.tolist():
            x1, y1, x2, y2, score, class_id = result
            rounded_score = round(score, 2)
            if score > threshold:
                cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 4)
                cv2.putText(frame, results.names[int(class_id)].upper() + f" {rounded_score}", (int(x1), int(y1 - 10)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 3, cv2.LINE_AA)

        

        out.write(frame)
        ret, frame = cap.read()
        frame_count += 1
    
    except cv2.error as e:
        print("Error occurred during video writing:", e)
        break

cap.release()
out.release()
cv2.destroyAllWindows()
