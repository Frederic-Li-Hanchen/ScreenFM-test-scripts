import pandas as pd
import os
from pdb import set_trace as st
import cv2
from time import time

############################################################################################################
# add_subtitles(): function to add the subtitles regarding FM annotations to a video
# Inputs:
# - [str] video_path: path to the video to annotate
# - [str] annotation_path1: path to the csv file containing the pairs of events annotated by one clinician
# - [str] annotation_path2: path to the csv file containing the pairs of events annotated by the second clinician
# - [str] save_path: path to save the annotated video.
# Outputs:
# - None
############################################################################################################
def add_subtitles(
        video_path,
        annotation_path1,
        annotation_path2,
        save_path,
):

    # Load the csv files and convert them into panda frames
    print('Loading the arrays of annotations ...')
    annotations1 = pd.read_csv(annotation_path1)
    annotations2 = pd.read_csv(annotation_path2)

    # Load the video and get its frame rate
    print('Loading the video ...')
    video = cv2.VideoCapture(video_path)
    num_frames = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = video.get(cv2.CAP_PROP_FPS)
    #duration = 1000*num_frames/fps # Duration of the video in seconds

    # Create a list of events to be shown as subtitle
    print('Creating subtitles ...')
    start = time()
    subtitles = pd.DataFrame(columns=['subtitle1','subtitle2'],index=range(num_frames))
    subtitles['subtitle1'] = ''*len(subtitles)
    subtitles['subtitle2'] = ''*len(subtitles)

    # Loop on annotations for subject 1
    for idx in range(len(annotations1)):
        current_event = annotations1.iloc[idx]
        start_ts = current_event['start_timestamp']/1000 # Starting timestamp in second
        start_frame = max(0,round(start_ts*fps)-1) # Starting frame ID
        end_ts = current_event['stop_timestamp']/1000 # Ending timestamp in second
        end_frame = min(round(end_ts*fps),num_frames-1) # Ending frame ID
        fm_type = current_event['start_event']
        pos = fm_type.find('_start')
        fm_type = fm_type[:pos]

        for idx2 in range(start_frame,end_frame):
            subtitles['subtitle1'][idx2] += ' '+fm_type

    # Loop on annotations for subject 2
    for idx in range(len(annotations2)):
        current_event = annotations2.iloc[idx]
        start_ts = current_event['start_timestamp']/1000 # Starting timestamp in second
        start_frame = max(0,round(start_ts*fps)-1) # Starting frame ID
        end_ts = current_event['stop_timestamp']/1000 # Ending timestamp in second
        end_frame = min(round(end_ts*fps),num_frames-1) # Ending frame ID
        fm_type = current_event['start_event']
        pos = fm_type.find('_start')
        fm_type = fm_type[:pos]

        for idx2 in range(start_frame,end_frame):
            subtitles['subtitle2'][idx2] += ' '+fm_type

    # Add prefix to subtitles
    for idx in range(len(subtitles)):
        if len(subtitles['subtitle1'][idx])>0:
            subtitles['subtitle1'][idx] = 'A1:'+subtitles['subtitle1'][idx]
        if len(subtitles['subtitle2'][idx])>0:
            subtitles['subtitle2'][idx] = 'A2:'+subtitles['subtitle2'][idx]
    
    end = time()
    print('Subtitles created in %.2f seconds.' % (end-start))

    # Get frame size
    _, frame = video.read()
    W, H, _ = frame.shape

    # Assign subtitles to video and save video
    video.set(cv2.CAP_PROP_POS_FRAMES, 0) # Set current frame back to 0
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    new_video = cv2.VideoWriter(save_path,fourcc,fps,(H,W)) # TODO: check this
    print('Assigning subtitles to video ...')
    start = time()
    for idx in range(num_frames):
        _, frame = video.read()
        if len(subtitles['subtitle1'][idx])>0:
            cv2.putText(frame,subtitles['subtitle1'][idx], (int(0.05*W),int(0.70*H)),cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2, cv2.LINE_AA, False)
        if len(subtitles['subtitle2'][idx])>0:
            cv2.putText(frame,subtitles['subtitle2'][idx], (int(0.05*W),int(0.72*H)),cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 170, 0), 2, cv2.LINE_AA, False)
        new_video.write(frame)
        # # DEBUG: show one frame at a time
        # cv2.imshow("video", frame)
        # key = cv2.waitKey(0)
        # while key not in [ord('q'), ord('k')]:
        #     key = cv2.waitKey(0)
    end = time()
    print('Subtitles assigned to video in %.2f seconds.' % (end-start))
    print('Subtitled video saved in %s' % (save_path))

    # Clean all variables
    cv2.destroyAllWindows()
    new_video.release()
    video.release()


############################################## Main function ###############################################
if __name__ == '__main__':
    # # CSV containing the events
    # file_path1 = r'./interrater_disagreement/annotations/Friederike_s033.csv'
    # file_path2 = r'./interrater_disagreement/annotations/Andrea_s033.csv'
    
    # # Path to video and where to save the result
    # video_path = r'./interrater_disagreement/video/033 – 20220420T123307Z – Lidar_0001.mp4'
    # save_path = r'./interrater_disagreement/video/annotated_s033.mp4'

    # CSV containing the events
    file_path1 = r'./interrater_disagreement/annotations/Friederike_s036.csv'
    file_path2 = r'./interrater_disagreement/annotations/Andrea_s036.csv'
    
    # Path to video and where to save the result
    video_path = r'./interrater_disagreement/video/036 – 20220427T121247Z – Lidar_0001.mp4'
    save_path = r'./interrater_disagreement/video/annotated_s036.mp4'

    # Plot the events
    add_subtitles(video_path=video_path,annotation_path1=file_path1,annotation_path2=file_path2,save_path=save_path)
############################################################################################################