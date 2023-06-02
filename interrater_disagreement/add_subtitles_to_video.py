import pandas as pd
import os
from pdb import set_trace as st
import cv2
from time import time
import check_FM_annotations as cfma

############################################################################################################
# add_subtitles(): function to add the subtitles regarding FM annotations to a video
# Inputs:
# - [str] video_path: path to the video to annotate
# - [str or pd frame] annotation_file1: path to the csv file or pandas frame containing the pairs of events annotated by one clinician
# - [str or pd frame or None] annotation_file2: path to the csv file or pandas frame containing the pairs of events annotated by the second clinician. If None, only annotations 1 are added to the video
# - [str] save_path: path to save the annotated video.
# Outputs:
# - None
############################################################################################################
def add_subtitles(
        video_path,
        annotation_file1,
        annotation_file2,
        save_path,
):

    # Load the csv files and convert them into panda frames
    print('Loading the arrays of annotations ...')
    if isinstance(annotation_file1,str):
        annotations1 = pd.read_csv(annotation_file1)
    else: # assumed to be pandas frame
        annotations1 = annotation_file1
    if annotation_file2 is not None:
        if isinstance(annotation_file2,str):
            annotations2 = pd.read_csv(annotation_file2)
        else: # assumed to be pandas frame
            annotations2 = annotation_file2

    # Load the video and get its frame rate
    print('Loading the video ...')
    video = cv2.VideoCapture(video_path)
    num_frames = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = video.get(cv2.CAP_PROP_FPS)
    #duration = 1000*num_frames/fps # Duration of the video in seconds

    # Create a list of events to be shown as subtitle
    print('Creating subtitles ...')
    start = time()
    #subtitles = pd.DataFrame(columns=['subtitle1','subtitle2'],index=range(num_frames))
    subtitles = pd.DataFrame(columns=['subtitle1_upper','subtitle1_lower','subtitle2_upper','subtitle2_lower'],index=range(num_frames))
    subtitles['subtitle1_upper'] = ''*len(subtitles)
    subtitles['subtitle2_upper'] = ''*len(subtitles)
    subtitles['subtitle1_lower'] = ''*len(subtitles)
    subtitles['subtitle2_lower'] = ''*len(subtitles)

    lower_fm = ['hips_l', 'hips_r', 'ankl_l', 'ankl_r']

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

        if fm_type in lower_fm:
            for idx2 in range(start_frame,end_frame):
                subtitles['subtitle1_lower'][idx2] += ' '+fm_type
        elif fm_type == 'invalid':
            for idx2 in range(start_frame,end_frame):
                subtitles['subtitle1_upper'][idx2] = ' ' + fm_type + subtitles['subtitle1_upper'][idx2]
                subtitles['subtitle1_lower'][idx2] = ' ' + fm_type + subtitles['subtitle1_lower'][idx2]
        else:
            for idx2 in range(start_frame,end_frame):
                subtitles['subtitle1_upper'][idx2] += ' '+fm_type

    # Loop on annotations for subject 2
    if annotation_file2 is not None:
        for idx in range(len(annotations2)):
            current_event = annotations2.iloc[idx]
            start_ts = current_event['start_timestamp']/1000 # Starting timestamp in second
            start_frame = max(0,round(start_ts*fps)-1) # Starting frame ID
            end_ts = current_event['stop_timestamp']/1000 # Ending timestamp in second
            end_frame = min(round(end_ts*fps),num_frames-1) # Ending frame ID
            fm_type = current_event['start_event']
            pos = fm_type.find('_start')
            fm_type = fm_type[:pos]

            if fm_type in lower_fm:
                for idx2 in range(start_frame,end_frame):
                    subtitles['subtitle2_lower'][idx2] += ' '+fm_type
            elif fm_type == 'invalid':
                for idx2 in range(start_frame,end_frame):
                    subtitles['subtitle2_upper'][idx2] = ' '+ fm_type + subtitles['subtitle2_upper'][idx2]
                    subtitles['subtitle2_lower'][idx2] = ' '+ fm_type + subtitles['subtitle2_lower'][idx2]
            else:
                for idx2 in range(start_frame,end_frame):
                    subtitles['subtitle2_upper'][idx2] += ' '+fm_type

    # Add prefix to subtitles
    for idx in range(len(subtitles)):
        if len(subtitles['subtitle1_upper'][idx])>0:
            subtitles['subtitle1_upper'][idx] = 'A1 upper:'+subtitles['subtitle1_upper'][idx]
        if len(subtitles['subtitle2_upper'][idx])>0:
            subtitles['subtitle2_upper'][idx] = 'A2 upper:'+subtitles['subtitle2_upper'][idx]
        if len(subtitles['subtitle1_lower'][idx])>0:
            subtitles['subtitle1_lower'][idx] = 'A1 lower:'+subtitles['subtitle1_lower'][idx]
        if len(subtitles['subtitle2_lower'][idx])>0:
            subtitles['subtitle2_lower'][idx] = 'A2 lower:'+subtitles['subtitle2_lower'][idx]
    
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
        if len(subtitles['subtitle1_upper'][idx])>0:
            cv2.putText(frame,subtitles['subtitle1_upper'][idx], (int(0.05*W),int(0.66*H)),cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2, cv2.LINE_AA, False)
        if len(subtitles['subtitle1_lower'][idx])>0:
            cv2.putText(frame,subtitles['subtitle1_lower'][idx], (int(0.05*W),int(0.68*H)),cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2, cv2.LINE_AA, False)
        if len(subtitles['subtitle2_upper'][idx])>0:
            cv2.putText(frame,subtitles['subtitle2_upper'][idx], (int(0.05*W),int(0.70*H)),cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 170, 0), 2, cv2.LINE_AA, False)
        if len(subtitles['subtitle2_lower'][idx])>0:
            cv2.putText(frame,subtitles['subtitle2_lower'][idx], (int(0.05*W),int(0.72*H)),cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 170, 0), 2, cv2.LINE_AA, False)
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
   
    ### Process several videos to subtitle
    # Subjects for which the subtitled videos should be processed (double annotations available)
    #subjects_to_process = ['034a','034b','035','036','037','038']
    subjects_to_process = ['049','052','053'] 
    
    # Folder containing the CSV annotation files
    annotation_path = r'Z:/Annotations'

    # Folder containing the videos
    video_path = r'D:/ScreenFM/Annotation Data'

    # Folder where to save the subtitled videos
    save_path = r'D:/ScreenFM/Subtitled Videos'

    # Loop on the videos to be procssed
    print('Starting the process of video subtitling ...')
    for subject_id in subjects_to_process:

        print('')
        print('Subject %s:' % subject_id)

        annotation_folder = os.path.join(annotation_path,subject_id).replace("\\","/")
        annotation_files = [e for e in os.listdir(annotation_folder) if ' – Comments' in e and '.csv' in e]

        # get path to the video
        video_folder = os.path.join(video_path,subject_id)
        video_files = [e for e in os.listdir(video_folder) if '.mp4' in e]
        tmp_video_name = video_files[0]
        iphone_video_path = os.path.join(video_folder,tmp_video_name)

        # check how many raters are there
        nb_raters = len(annotation_files)

        # get rater names
        rater_names = [] 
        for idx in range(nb_raters):
            tmp_name = annotation_files[idx]
            pos1 = tmp_name.find(' – Comments ')
            pos2 = tmp_name.find('.csv')
            rater_names += [tmp_name[pos1+len(' – Comments '):pos2]]

        # prepare save folder
        save_folder = os.path.join(save_path,subject_id)
        if not os.path.isdir(save_folder):
            os.makedirs(save_folder)

        if nb_raters == 2:  # nb_raters = 2
            pos = tmp_video_name.find('.mkv.mp4')
            save_name = tmp_video_name[:pos]+' - '+rater_names[0]+' '+rater_names[1]+'.mkv.mp4'
            full_save_path = os.path.join(save_folder,save_name).replace("\\","/")
            annotation1 = [e for e in annotation_files if rater_names[0] in e]
            file_path1 = os.path.join(annotation_folder,annotation1[0]).replace("\\","/")
            annotation2 = [e for e in annotation_files if rater_names[1] in e]
            file_path2 = os.path.join(annotation_folder,annotation2[0]).replace("\\","/")
            # Compute tables of correspondencies
            annotation_table1 = cfma.list_annotated_events(file_path1,time_threshold=60000,res_path='')
            annotation_table2 = cfma.list_annotated_events(file_path2,time_threshold=60000,res_path='')
            # Add subtitles to video
            add_subtitles(video_path=iphone_video_path,annotation_file1=annotation_table1,annotation_file2=annotation_table2,save_path=full_save_path)

        elif nb_raters == 3: # nb_raters = 3
            pos = tmp_video_name.find('.mkv.mp4')
            ### Raters 1 and 2
            save_name = tmp_video_name[:pos]+' - '+rater_names[0]+' '+rater_names[1]+'.mkv.mp4'
            full_save_path = os.path.join(save_folder,save_name).replace("\\","/")
            annotation1 = [e for e in annotation_files if rater_names[0] in e]
            file_path1 = os.path.join(annotation_folder,annotation1[0]).replace("\\","/")
            annotation2 = [e for e in annotation_files if rater_names[1] in e]
            file_path2 = os.path.join(annotation_folder,annotation2[0]).replace("\\","/")
            # Compute tables of correspondencies
            annotation_table1 = cfma.list_annotated_events(file_path1,time_threshold=60000,res_path='')
            annotation_table2 = cfma.list_annotated_events(file_path2,time_threshold=60000,res_path='')
            # Add subtitles to video
            add_subtitles(video_path=iphone_video_path,annotation_file1=annotation_table1,annotation_file2=annotation_table2,save_path=full_save_path)
            ### Raters 1 and 3
            save_name = tmp_video_name[:pos]+' - '+rater_names[0]+' '+rater_names[2]+'.mkv.mp4'
            full_save_path = os.path.join(save_folder,save_name).replace("\\","/")
            annotation1 = [e for e in annotation_files if rater_names[0] in e]
            file_path1 = os.path.join(annotation_folder,annotation1[0]).replace("\\","/")
            annotation2 = [e for e in annotation_files if rater_names[2] in e]
            file_path2 = os.path.join(annotation_folder,annotation2[0]).replace("\\","/")
            # Compute tables of correspondencies
            annotation_table1 = cfma.list_annotated_events(file_path1,time_threshold=60000,res_path='')
            annotation_table2 = cfma.list_annotated_events(file_path2,time_threshold=60000,res_path='')
            # Add subtitles to video
            add_subtitles(video_path=iphone_video_path,annotation_file1=annotation_table1,annotation_file2=annotation_table2,save_path=full_save_path)
            ### Raters 2 and 3
            save_name = tmp_video_name[:pos]+' - '+rater_names[1]+' '+rater_names[2]+'.mkv.mp4'
            full_save_path = os.path.join(save_folder,save_name).replace("\\","/")
            annotation1 = [e for e in annotation_files if rater_names[1] in e]
            file_path1 = os.path.join(annotation_folder,annotation1[0]).replace("\\","/")
            annotation2 = [e for e in annotation_files if rater_names[2] in e]
            file_path2 = os.path.join(annotation_folder,annotation2[0]).replace("\\","/")
            # Compute tables of correspondencies
            annotation_table1 = cfma.list_annotated_events(file_path1,time_threshold=60000,res_path='')
            annotation_table2 = cfma.list_annotated_events(file_path2,time_threshold=60000,res_path='')
            # Add subtitles to video
            add_subtitles(video_path=iphone_video_path,annotation_file1=annotation_table1,annotation_file2=annotation_table2,save_path=full_save_path)

        elif nb_raters == 1:
            #print('Warning: subject %s skipped because only one annotation available!' % (subject_id))
            pos = tmp_video_name.find('.mkv.mp4')
            save_name = tmp_video_name[:pos]+' - '+rater_names[0]+'.mkv.mp4'
            full_save_path = os.path.join(save_folder,save_name).replace("\\","/")
            annotation1 = [e for e in annotation_files if rater_names[0] in e]
            file_path1 = os.path.join(annotation_folder,annotation1[0]).replace("\\","/")
            # Compute tables of correspondencies
            annotation_table1 = cfma.list_annotated_events(file_path1,time_threshold=60000,res_path='')
            # Add subtitles to video
            add_subtitles(video_path=iphone_video_path,annotation_file1=annotation_table1,annotation_file2=None,save_path=full_save_path)
        
        else:
            print('Warning: subject %s skipped because no annotations found!' % (subject_id))

############################################################################################################