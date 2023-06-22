import pandas as pd
from pdb import set_trace as st

# Path to the files to be merged
file_path1 = r'../annotations/078 – 20220818T114107Z – Comments lower part.csv'
file_path2 = r'../annotations/078 – 20220818T114107Z – Comments upper part.csv'

# Load the files as pd frame
df1 = pd.read_csv(file_path1,delimiter=',')
df2 = pd.read_csv(file_path2,delimiter=',')

# Remove the Lidar start and stop events from the second frame (because redundant)
df2.drop(0, inplace=True)
df2.drop(len(df2), inplace=True)

# Concatenate the two frames
new_df = pd.concat([df1,df2])

# Reorder the lines by increasing timestamp
new_df = new_df.sort_values('timestamp',ascending=True)

# Save as csv
new_df.to_csv('../078 – 20220818T114107Z – Comments.csv', sep=',', index=False)