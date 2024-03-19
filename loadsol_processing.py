import streamlit as st
import pandas as pd
import numpy as np
import math
import scipy.stats as stats
import matplotlib.pyplot as plt
import warnings
from scipy.signal import find_peaks,resample


# Set up warnings and pandas options
warnings.filterwarnings('ignore')
pd.set_option('display.max_columns', None)

# Initialize Streamlit app
st.set_page_config(page_title='Loadsol Processing UVA', layout='wide')

# Written by Dante Goss at the University of Virginia

################################### Functions #######################################

def stepfinder(dataframe):
    '''
    Takes a dataframe containing Force on the left and right side.
    Searches in time from a heel strike to a toe off between .4 and 1 seconds.
    Adds all steps meeting this condition to a list by limb.

    Returns two list of steps by limb.
    '''
    # Generate Stance Variables
    dataframe['RightStance'] = np.where(dataframe['RightForce[N]']>force_threshold,1,0)
    dataframe['LeftStance'] = np.where(dataframe['LeftForce[N]']>force_threshold,1,0)


    # Identify events where forceplate goes from loaded to unloaded, or unloaded to loaded
    dataframe['LeftHeelStrike'] = (dataframe['LeftStance']== 1) & (dataframe['LeftStance'].shift(1)== 0) & (dataframe['LeftStance'].shift(-1)== 1)
    dataframe['LeftToeOff'] = (dataframe['LeftStance']== 0) & (dataframe['LeftStance'].shift(1)== 1)  & (dataframe['LeftStance'].shift(2)==1) 

    dataframe['RightHeelStrike'] = (dataframe['RightStance']== 1) & (dataframe['RightStance'].shift(1)== 0) & (dataframe['RightStance'].shift(-1)== 1)
    dataframe['RightToeOff'] = (dataframe['RightStance']== 0) & (dataframe['RightStance'].shift(1)== 1)  & (dataframe['RightStance'].shift(2)==1) 


    # Search for left and right steps
    left_step_data = []
    Left_HS_idx = dataframe.index[dataframe['LeftHeelStrike']].tolist()
    Left_TO_idx = dataframe.index[dataframe['LeftToeOff']].tolist()
    # Search through the heel strikes
    for i in Left_HS_idx:
        earliest = i + (400/(1000/Sampling_Rate)) # If a step takes 40% of a second or 400 ms / 0.4 seconds
            # Since 1000 ms is 1 Second, 1000/ Sampling rate in Hz = Samples / Second
            # 400 ms / 4 ms = 100 samples for example at 250 Hz
        latest = i + (Sampling_Rate) # This is sampling rate since a step should take max 1 second.
        # Find the next toe off in the range
        for j in Left_TO_idx:
            if (j > earliest) and (j<latest) and (j>i):
                # Add to a list
                left_step_data.append(dataframe[i:j])
    ######################################## 
    right_step_data = []
    Right_HS_idx = dataframe.index[dataframe['RightHeelStrike']].tolist()
    Right_TO_idx = dataframe.index[dataframe['RightToeOff']].tolist()
    # Search through the heel strikes
    for k in Right_HS_idx:
        earliest = k + (400/ (1000/Sampling_Rate))
        latest = k + (Sampling_Rate)
        # Find the next toe off in the range
        for l in Right_TO_idx:
            if (l > earliest) and (l<latest) and (l>k):
                # Add to a list
                right_step_data.append(dataframe[k:l])
    return left_step_data, right_step_data

def spatio_calc(df,column):
    """
    Calculate spatiotemporal variables baseed on grf data
    """
    basecols = ['contact_time','time_to_peak','time_between_peaks','loading_rate','loading_rate_norm','first_peak_grf','first_peak_grf_norm','second_peak_grf','second_peak_grf_norm','mean_grf','mean_grf_norm','grf_impulse','grf_impulse_norm']

    grf_stats = pd.DataFrame(columns = basecols)
    contact_time = ((len(df[column]))/Sampling_Rate)
    # Find Peaks (internal variable)
    peaks, _ = find_peaks(df[column], distance=(150/(1000/Sampling_Rate)), height=(0.7*weight_N))
    # Time to peak
    time_to_peak = ((peaks[0])/Sampling_Rate)
    # Time between peaks 
    time_between_peaks = ((peaks[-1]-peaks[0])/Sampling_Rate)
    # Average Loading Rate
    loading_rate = df[column][df.index[0] + peaks[0]]/time_to_peak
    # norm
    loading_rate_norm = loading_rate / normalizer
    # Peak GRF
    first_peak_grf = df[column][df.index[0] + peaks[0]]
    # Norm
    first_peak_grf_norm = first_peak_grf/ normalizer
    # Second peak
    second_peak_grf = df[column][df.index[0] + peaks[-1]]
    # Norm
    second_peak_grf_norm = second_peak_grf/ normalizer
    
    # Mean
    mean_grf = df[column].mean()
    # Norm
    mean_grf_norm = mean_grf/ normalizer

    # GRF Impulse
    grf_impulse = np.trapz(df[column][df[column]>0], dx= (1000/Sampling_Rate)/ 1000)
    # Norm GRF Impulse
    grf_impulse_norm = grf_impulse / normalizer


    grf_stats.loc[0] = [contact_time,time_to_peak,time_between_peaks,loading_rate,loading_rate_norm, first_peak_grf, first_peak_grf_norm,second_peak_grf,second_peak_grf_norm,mean_grf,mean_grf_norm,grf_impulse,grf_impulse_norm]
    return grf_stats

def step_summarizer(good_steps_left,good_steps_right):
    
    left_all_steps = pd.DataFrame()
    for lstep in good_steps_left:
        left_spatio_step = spatio_calc(lstep,'LeftForce[N]')

        left_all_steps = pd.concat([left_all_steps, left_spatio_step])

    right_all_steps = pd.DataFrame()
    for rstep in good_steps_right:
        right_spatio_step = spatio_calc(rstep,'RightForce[N]')

        right_all_steps = pd.concat([right_all_steps, right_spatio_step]) 


    left_all_steps['Limb'] = 'Left'
    right_all_steps['Limb'] = 'Right'

    all_steps = pd.concat([left_all_steps,right_all_steps])

    return(all_steps)

############################     App    #########################################

# Streamlit UI for file upload
st.title('Loadsol Gait Processing Code')
st.write('This app allows the processing of gait data collected with single sensor loadsols.')
st.write('')
data_files = st.file_uploader('Upload the text file reports from loadsol', accept_multiple_files=True)

# Initialize an empty DataFrame to store all summaries
combined_summaries = pd.DataFrame()

force_threshold = st.sidebar.number_input('Force Threshold (N): ', min_value=20, value=50)
weight_kg = st.sidebar.number_input("What is the subject's weight in kilograms?",value=100)
weight_N = weight_kg * 9.81
normalize = st.sidebar.selectbox('What would you like to normalize by?',('Newtons','Kilograms'))

if normalize == 'Newtons':
    normalizer = weight_N
else:
    normalizer = weight_kg

# Initialize an empty DataFrame to store all summaries
combined_summaries = pd.DataFrame()

# Check if files have been uploaded
if data_files:

    # Process each file
    for data_file in data_files:
        data_file.seek(0)

        # Read data from the file
        data_file.seek(0)  
        df = pd.read_csv(data_file, delimiter='\t', header = 3, index_col=0 ,usecols=[1,3])
        df.index.name = 'Time (Sec)'
        df.reset_index(inplace=True)
        df.rename(columns={'Force[N]': 'LeftForce[N]','Force[N].1': 'RightForce[N]'},inplace=True)
        st.write('View Raw Data')
        st.dataframe(df)
        st.line_chart(df, x ='Time (Sec)', y=df[['LeftForce[N]','RightForce[N]']], color = ["#232D4B","#E57200"])

        # Get Sampling Rate 
        Sampling_Rate = 1 / df['Time (Sec)'][1]-df['Time (Sec)'][0]

        # Isolate Steps
        left_steps, right_steps = stepfinder(df)

        # How many steps on each side?
        st.write(f'There were {len(left_steps)} steps recorded on the left foot.')
        st.write(f'There were {len(right_steps)} steps recorded on the right foot.')
        st.write('---')

        trial_summary = step_summarizer(left_steps, right_steps)
        st.write('All Steps')
        st.write(trial_summary)

        st.write('Summarized by limb')
        limb_steps = trial_summary.groupby(['Limb']).agg('mean')
        st.write(limb_steps)

        st.write('Final Summary')
        nolimb = trial_summary.drop(columns='Limb')
        final_summary = nolimb.mean()
        #final_summary['Comment'] = comment
        # Convert final_summary Series to DataFrame for a single row
        final_summary_df = pd.DataFrame([final_summary])
        st.write(final_summary_df)

        
        
        combined_summaries = pd.concat([combined_summaries, final_summary_df], ignore_index=True)
        st.write('---')

    st.write('Combined Summary')
    st.write(combined_summaries)

else:
    st.error('Please upload the files corresponding to the trials.')



# There may need to be a check to make sure the data isn't zero for a prolonged period.