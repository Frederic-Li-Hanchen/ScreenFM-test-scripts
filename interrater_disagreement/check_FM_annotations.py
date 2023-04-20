############################################################################################################
# Script to process the FM annotations attributed by the clinicians 
############################################################################################################
import numpy as np
from pdb import set_trace as st
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches


############################################################################################################
# List of possible FM annotations
############################################################################################################
annotation_list = [
    'throat',
    'should_l', 'should_r', # 'should_both',
    'elbow_l', 'elbow_r',
    'wrst_l', 'wrst_r',
    'fing_l', 'fing_r',
    'hips_l', 'hips_r', # 'hips_both',
    'ankl_l', 'ankl_r',
    'invalid'
    ]

############################################################################################################
# list_annotated_events(): create a table of starting and stop events
# Inputs:
# - [str] file_path: path to the CSV file to be processed
# - [int] time_threshold: maximum duration in ms allowed between a 'FM_start' and 'FM_stop' event
# - [str] res_path: path where to save the table in CSV format. If empty, no saving is performed
# Outputs:
# - [df] data frame containing the table of starting and stop events
############################################################################################################
def list_annotated_events(file_path,time_threshold=60000,res_path=''):
    # Read csv file
    df = pd.read_csv(file_path,encoding='latin')

    # Filter all events that contain start or stop
    b1 = df['type']=='Text'
    b2 = df['content'].str.contains('_start')
    b3 = df['content'].str.contains('_stop')
    df_events = df[b1 & (b2|b3)]
    #df_start_events = df[b1&b2]
    df_end_events = df[b1&b3]

    # Parse the events to find pairs of annotations
    events = [] # List of tuples of format (event_start, timestamp_start, event_stop, timestamps_stop)
    not_processed = df_events.index.to_list()

    while len(not_processed)>0:
        idx = not_processed[0]
        end_event_idx = -1
        end_event = ''
        end_event_timestamp = -1
        
        if '_stop' in df_events.loc[idx]['content']: # The event is a stop event
            events += [['None','NaN',df_events.loc[idx]['content'],df_events.loc[idx]['timestamp'],'NaN']]
            #print('Deleted event %s: %d' % (df_events.loc[idx]['content'], idx))
            
        else: # The event is a start event
            # Get nature of event
            pos = df_events.loc[idx]['content'].find('_start')
            event_name = df_events.loc[idx]['content'][:pos]
            # Looking for the corresponding ending event
            if event_name != 'invalid': # Assumption that start and stop of a FM cannot be separated by more than threshold ms
                possible_end_events = df_end_events[(df_end_events['timestamp']>df_events.loc[idx]['timestamp']) & (df_end_events['timestamp']<=df_events.loc[idx]['timestamp']+time_threshold)]
            else: # Invalid periods can be of arbitrary duration
                possible_end_events = df_end_events
            stop_idx = 0  
            #print('         Length possible end events: %d' % len(possible_end_events))
            while end_event == '' and stop_idx < len(possible_end_events): # The first matching event found in the list of possible end event is considered as stopping event 
                #print('         %d' % stop_idx)
                pos = possible_end_events.iloc[stop_idx]['content'].find('_stop')
                end_event_name = possible_end_events.iloc[stop_idx]['content'][:pos]
                if end_event_name == event_name:
                    end_event = event_name
                    end_event_timestamp = possible_end_events.iloc[stop_idx]['timestamp']
                    end_event_idx = possible_end_events.index[stop_idx]
                stop_idx += 1

            #print("%d, %d"%(idx,end_event_idx))

            # If end event found, remove both start and end events from the lists of unprocessed and end events.
            if end_event != '':
                # if end_event_idx not in not_processed:
                #     st()
                df_end_events = df_end_events.drop(index=end_event_idx)
                not_processed.remove(end_event_idx) 
                # Add event to list of events
                events += [[df_events.loc[idx]['content'], df_events.loc[idx]['timestamp'],end_event+'_stop',end_event_timestamp, end_event_timestamp-df_events.loc[idx]['timestamp']]]
            else:
                # Add event to list of events
                events += [[df_events.loc[idx]['content'], df_events.loc[idx]['timestamp'],'None','NaN','Nan']]
    
        not_processed.remove(idx)
        # print(idx)
        # print('     %d' %end_event_idx)

    # Remove any event that contains NaN (either start event without end or end event without start)
    idx_to_keep = []
    for idx in range(len(events)):
        event = events[idx]
        if not 'NaN' in event:
            idx_to_keep += [idx]
    events = [e for i,e in enumerate(events) if i in idx_to_keep]

    # Convert 'both' events to their respective FM events
    new_events = []
    for idx in range(len(events)):
        event = events[idx]
        event_name = event[0]
        if '_both_' in event_name:
            # Find location of FM
            pos = event_name.find('_both_')
            fm_type = event_name[:pos]
            # Duplicate the event for left and right
            event_l = [fm_type+'_l_start',event[1],fm_type+'_l_stop',event[3],event[4]]
            event_r = [fm_type+'_r_start',event[1],fm_type+'_r_stop',event[3],event[4]]
            # Add left and right events to list
            new_events += [event_l]
            new_events += [event_r]

    # Remove both events from the list and concatenate them with the new events
    idx_to_keep = []
    for idx in range(len(events)):
        event = events[idx]
        event_name = event[0]
        if not '_both_' in event_name:
            idx_to_keep += [idx]
    events = [e for i,e in enumerate(events) if i in idx_to_keep]
    events += new_events

    # Convert the events to panda frame and order them by starting timestamp
    res = pd.DataFrame(events,columns=['start_event', 'start_timestamp', 'stop_event', 'stop_timestamp', 'duration'], index=range(len(events)))
    res = res.sort_values('start_timestamp')

    # Save the list of events in a csv file
    if len(res_path) > 1:
        res.to_csv(res_path,index=False)
    return res


############################################################################################################
# plot_annotated_events(): plots the annotated events of two clinicians
# Inputs:
# - [df] annotations1: data frame of events as annotated by one clinician
# - [df] annotations2: data frame of events as annotated by one clinician
# - [str] save_path: path to save the plotted figure. If empty, the figure is displayed instead of being saved
# - [str] title: title to give to the figure. No assigned if left empty
# Outputs:
# - None
############################################################################################################
def plot_annotated_events(annotations1, annotations2, save_path='', title=''):

    # Get the maximum timestamp range for plotting
    max_ts = max(max(annotations1['stop_timestamp']),max(annotations2['stop_timestamp']))
    # Get the number of different events to plot
    nb_events = len(annotation_list)
    #for idx in range(annotation_list):
    # Create figure
    offset = 0.20 # Offset regarding the plotting of annotations for both reviewers
    colors = ['blue', 'green']
    #plt.figure()
    # Loop on all events
    for idx in range(nb_events):
        current_event = annotation_list[idx]
        for idx1 in range(len(annotations1)):
            event1 = annotations1.iloc[idx1]
            if current_event in event1['start_event']:
                # Get start and end timestamps of event
                start_ts = event1['start_timestamp']
                end_ts = event1['stop_timestamp']
                # Plot line
                plt.hlines(y=idx+1+offset,xmin=start_ts,xmax=end_ts,color=colors[0],linestyle='-',linewidth=2.5)
                # Plot (possibly overlapping) lines for the FM annotation
                if current_event != 'invalid':
                    plt.hlines(y=nb_events+1+offset,xmin=start_ts,xmax=end_ts,color=colors[0],linestyle='-',linewidth=2.5)
        for idx2 in range(len(annotations2)):
            event2 = annotations2.iloc[idx2]
            if current_event in event2['start_event']:
                # Get start and end timestamps of event
                start_ts = event2['start_timestamp']
                end_ts = event2['stop_timestamp']
                # Plot line
                plt.hlines(y=idx+1-offset,xmin=start_ts,xmax=end_ts,color=colors[1],linestyle='-',linewidth=2.5)
                # Plot (possibly overlapping) lines for the FM annotation
                if current_event != 'invalid':
                    plt.hlines(y=nb_events+1-offset,xmin=start_ts,xmax=end_ts,color=colors[1],linestyle='-',linewidth=2.5)
    # Plot legend
    patch1 = mpatches.Patch(color=colors[0],label='Annotator 1')
    patch2 = mpatches.Patch(color=colors[1],label='Annotator 2')
    plt.legend(handles=[patch1,patch2])
    # Plot separations for each FM
    for idx in range(nb_events):
        plt.hlines(y=idx+1.5,xmin=0,xmax=max_ts,color='black',linestyle=':',linewidth=1)
    # Specify x range
    plt.xlim([0,max_ts])
    # Write axis labels
    plt.xlabel('Time (in ms)')
    plt.yticks(ticks=list(range(1,nb_events+2)),labels=annotation_list+['FM'])
    # Set title
    if len(title)>0:
        plt.title(title)
    # Save or display the figure
    if len(save_path)>0:
        plt.savefig(save_path)
    else:
        plt.show()


############################################## Main function ###############################################
if __name__ == '__main__':
    # Data files containing the labels
    # file_path1 = r'./annotations/033 – 20220420T123307Z – Comments Friederike Pagel.csv'
    # file_path2 = r'./annotations/033 – 20220420T123307Z – Comments Andrea Kock.csv'
    # file_path3 = r'./annotations/033 – 20220420T123307Z – Comments Margot Lau.csv'

    file_path1 = r'./annotations/036 – 20220427T121247Z – Comments Friederike Pagel.csv'
    file_path2 = r'./annotations/036 – 20220427T121247Z – Comments Andrea Kock.csv'
    file_path3 = r'./annotations/036 – 20220427T121247Z – Comments Margot Lau.csv'
    
    # # Path to result file
    # res_path = r'./042 – 20220519T090912Z – Comments Friederike Pagel - Analysis.csv'

    # Compute tables of events
    annotations1 = list_annotated_events(file_path1,res_path='./interrater_disagreement/annotations/Friederike_s036.csv')
    annotations2 = list_annotated_events(file_path2,res_path='./interrater_disagreement/annotations/Andrea_s036.csv')
    annotations3 = list_annotated_events(file_path3,res_path='./interrater_disagreement/annotations/Margot_s036.csv')

    # Plot the events
    #plot_annotated_events(annotations1, annotations2, save_path='', title='')
    plot_annotated_events(annotations1, annotations3, save_path='', title='')
############################################################################################################