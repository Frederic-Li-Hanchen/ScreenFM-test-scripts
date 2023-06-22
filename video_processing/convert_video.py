import cv2

# Video path
video_path = r'D:\ScreenFM\Subtitled Videos\041\041 – 20220519T073731Z – Lidar_0001.mp - Friederike Pagel.mkv.mp4'

# Path where to save the result
result_path = r'D:\ScreenFM\Subtitled Videos\041\test.avi'

# Convert video
video_cap = cv2.VideoCapture(video_path)
fps = video_cap.get(cv2.CAP_PROP_FPS)
size = (int(video_cap.get(cv2.CAP_PROP_FRAME_WIDTH)),   
        int(video_cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))  
video_writer = cv2.VideoWriter(result_path, cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'), fps, size) 

success, frame = video_cap.read()
while success:
    video_writer.write(frame)
    success, frame = video_cap.read()