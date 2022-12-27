import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib.patheffects as PathEffects
from gazeheatplot_script import draw_heatmap
# Gazeheatplot is based on https://github.com/TobiasRoeddiger/GazePointHeatMap
from PIL import Image
from skimage.measure import regionprops
# To Generate "Center of Mass"
# https://stackoverflow.com/questions/48888239/finding-the-center-of-mass-in-an-image
import os
from findcentermass import *

DATASET_DIR = os.path.join("ChangeBlindnessTTM", "ChangeBlindnessTTM-ExpComplete", "Data", "CBDatabase")
OUTPUT_DIR_PROCESSED_DATAFRAME = "processed_data"
TOLERANCE_IN_MASK = 30
IMAGE_X_SIZE = 1024
IMAGE_Y_SIZE = 768

def get_postProcessedData():
    combined_all_dataframes = pd.DataFrame()
    for csv_path in os.listdir(OUTPUT_DIR_PROCESSED_DATAFRAME):
        if ".csv" in csv_path:
            df = pd.read_csv(os.path.join(OUTPUT_DIR_PROCESSED_DATAFRAME, csv_path))
            combined_all_dataframes = pd.concat([combined_all_dataframes,df])
    combined_all_dataframes = combined_all_dataframes.drop('Unnamed: 0', axis=1)
    return combined_all_dataframes

def get_dict_start_end(recording_name, data):
    assert recording_name in set(data["Recording name"]), "ERROR: Recording name not found in data."
    dict_with_start_end = dict()
    dict_start_end_found = dict()
    # We iterate in order, so it's garanteed that we will have the events in order
    recording_filter = data["Recording name"] == recording_name
    for i,row in data[recording_filter][['Event', 'Event value']][~data[recording_filter]['Event'].isnull()].iterrows():
        event_name_og = row['Event value']
        if row['Event'] == 'ImageStimulusStart':
            for i in range(4):
                event_name = event_name_og + f"_{i}"
                if event_name not in dict_with_start_end:
                    # If it's not in the event
                    break
            dict_with_start_end[event_name] = [row.name]
            dict_start_end_found[event_name] = False
        elif row['Event'] == 'ImageStimulusEnd':
            for i in range(4):
                event_name = event_name_og + f"_{i}"
                if not dict_start_end_found[event_name]:
                    # If it's not in the event
                    dict_start_end_found[event_name] = True
                    break
            dict_with_start_end[event_name].append(row.name)
    return dict_with_start_end

def get_dict_mouse_clicks(dict_start_end, data):
    dict_mouse_clicks = dict()
    for key, (start,end) in dict_start_end.items():
        subset_data = data.loc[start:end]
        if (subset_data["Event"] == "MouseEvent").any():
            # There is a mouse event
            mouse_click_i = subset_data[subset_data["Event"] == "MouseEvent"].index[0]
            after_mouse_click_subset = subset_data.loc[mouse_click_i:]
            after_mouse_click_subset = after_mouse_click_subset[after_mouse_click_subset["Sensor"] == "Mouse"]
            c_time, time, mX, mY = after_mouse_click_subset[["Computer timestamp", "Recording timestamp","Mouse position X", "Mouse position Y"]].iloc[0]
            dict_mouse_clicks[key] = (c_time, time, mX, mY)
    return dict_mouse_clicks

def get_dict_fixation(dict_start_end, data):
    dict_fix_data = dict()
    for key, (start,end) in dict_start_end.items():
        fixation_data = data.loc[start:end,["Computer timestamp", "Recording timestamp",  'Eye movement type', 'Eye movement type index','Gaze event duration', 'Fixation point X', 'Fixation point Y']]
        fixation_indexes = fixation_data[fixation_data["Eye movement type"] == "Fixation"]["Eye movement type index"].unique()
        if fixation_indexes.any():
            fix_list = []
            for i in fixation_indexes:
                fix_data = fixation_data[(fixation_data["Eye movement type index"]==i) & (fixation_data["Eye movement type"]=="Fixation")]
                c_time, time, _, _, duration, fX, fY = fix_data.max()
                # Manually calculated 
                duration = fix_data["Recording timestamp"].max() - fix_data["Recording timestamp"].min()
                fix_list.append([c_time, time, duration, fX, fY])
            dict_fix_data[key] = fix_list
    return dict_fix_data

def load_mask(path):
    img = Image.open(path).convert('L')
    img = ~np.array(img,dtype=bool)
    return img

def point_in_image(x,y):
    # For point to be in image 0 <= x <= 1024
    # and 0 <= y <= 768
    return (x >= 0 and x < IMAGE_X_SIZE) and (y >= 0 and y < IMAGE_Y_SIZE)

def is_in_mask(x, y, mask, tolerance=TOLERANCE_IN_MASK):
    for x_t in range(x-tolerance, x+tolerance+1):
        for y_t in range(y-tolerance, y+tolerance+1):
            if point_in_image(x_t,y_t):
                if mask[y_t,x_t]:
                    return True
    return False

def get_img_name_from_trial(trial_name):
    is_already_trial_name = False
    try:
        int(trial_name[-1])
    except:
        is_already_trial_name = True
    if is_already_trial_name:
        return trial_name
    # Filters the added strings    
    img_name = trial_name[:-2] # Remove the added substring from the dict_start_end
    img_name = img_name.replace(" - Kopi", "") # Remove in case of copy
    img_name = img_name.replace(" - Copy", "") # Remove in case of copy
    img_name = img_name.replace(" (1)", "") # Remove in case of renamed
    return img_name

def get_mask_name(trial_name):
    img_name = get_img_name_from_trial(trial_name)
    mask_name = ""
    for file in os.listdir(os.path.join(DATASET_DIR, "Masks")):
        if ".png" in file:
            file_n, file_side = file.split("_")[:2]
            img_n, img_side = img_name.split("_")[:2]
            if (int(file_n) == int(img_n) and 
               file_side == img_side):
                mask_name = file
                break
    return mask_name

def create_processed_dataframe(recording_name, data):

    """
        Takes a dataframe of raw data from Tobii and produces a dataframe for a recording which only includes:
        - For each stimuli (if any):
            - Fixation (time, duration, n, fX, fY)
            - MouseClick (time, mX, mY)
    """
    name = data[data["Recording name"] == recording_name].iloc[0]["Participant name"]
    data_dict_format = {
        "Computer_time": np.NaN,
        "Event_time":np.NaN,
        "Participant":name,
        "Recording":recording_name,
        "Stimulus_name" : np.NaN,
        "Image_n": np.NaN,
        "Repetition_n": np.NaN,
        "Stimulus_n": np.NaN,
        "Event_type": np.NaN,
        "Mouse_X": np.NaN,
        "Mouse_Y": np.NaN,
        "Mouse_X'": np.NaN,
        "Mouse_Y'": np.NaN,
        "Mouse_Confirm_Trial": False,
        "Mouse_in_Image": False,
        "Mouse_in_Mask": False,
        "Response_time": np.NaN,
        "Fixation_N": np.NaN,
        "Fixation_X": np.NaN,
        "Fixation_Y": np.NaN,
        "Fixation_X'": np.NaN,
        "Fixation_Y'": np.NaN,
        "Fixation_in_Image": False,
        "Fixation_in_Mask": False,
        "Fixation_duration": np.NaN,
        "Fixation_dist_to_mask_center": np.NaN,
        "Center_mass_X": np.NaN,
        "Center_mass_Y": np.NaN
    }
    
    dict_start_end = get_dict_start_end(recording_name, data)
    dict_fix = get_dict_fixation(dict_start_end, data)
    dict_mouse =  get_dict_mouse_clicks(dict_start_end, data)
    
    list_for_df = []
    last_repetion_n = -1
    mouse_trial = set()
    dict_masks = {}
    for stimulus, (start, end) in dict_start_end.items():
        repetition_n = int(stimulus[-1])
        new_row = data_dict_format.copy()
        if last_repetion_n > repetition_n:
            new_row["Mouse_Confirm_Trial"] = True
            mouse_trial.add(stimulus)
        name = get_img_name_from_trial(stimulus)
        img_n = int(name.split("_")[0])
        new_row["Computer_time"] = data.loc[start,"Computer timestamp"]
        new_row["Image_n"] = img_n
        new_row["Event_time"] = data.loc[start,"Recording timestamp"]
        new_row["Stimulus_name"] = name
        new_row["Repetition_n"] = repetition_n
        new_row["Event_type"] = "TrialStart"
        list_for_df.append(new_row)
        new_row = data_dict_format.copy()
        if stimulus in mouse_trial:
            new_row["Mouse_Confirm_Trial"] = True
        name = get_img_name_from_trial(stimulus)
        img_n = int(name.split("_")[0])
        new_row["Computer_time"] = data.loc[end,"Computer timestamp"]
        new_row["Image_n"] = img_n
        new_row["Event_time"] = data.loc[end,"Recording timestamp"]
        new_row["Stimulus_name"] = name
        new_row["Repetition_n"] = repetition_n
        new_row["Event_type"] = "TrialEnd"
        last_repetion_n = repetition_n
        list_for_df.append(new_row)
        
    center_of_mass_cache = {}
    for stimulus, fixations in dict_fix.items():
        for fi, fixation in enumerate(fixations):
            c_time, time, duration, fX, fY = fixation
            repetition_n = int(stimulus[-1])
            name = get_img_name_from_trial(stimulus)
            img_n = int(name.split("_")[0])

            new_row = data_dict_format.copy()
            new_row["Computer_time"] = c_time
            new_row["Event_time"] = time
            new_row["Stimulus_name"] = name
            new_row["Image_n"] = img_n
            new_row["Repetition_n"] = repetition_n
            new_row["Fixation_duration"] = duration
            new_row["Fixation_N"] = fi
            new_row["Fixation_X"] = fX
            new_row["Fixation_Y"] = fY
            new_row["Fixation_X'"] = fX - 448
            new_row["Fixation_Y'"] = fY - 156
            if "practice" not in name:
                if not (name in center_of_mass_cache):
                    center = getCenterOfMass(name)
                    center_of_mass_cache[name] = center
                else:
                    center = center_of_mass_cache[name]
                    center = np.array(center)
                new_row["Center_mass_X"] = center[0]
                new_row["Center_mass_Y"] = center[1]
                new_row["Fixation_dist_to_mask_center"] = np.linalg.norm(np.array([new_row["Fixation_X'"],new_row["Fixation_Y'"]]) - center)
            
            if "practice" not in name:
                mask_name = get_mask_name(name)
                if mask_name not in dict_masks:
                    dict_masks[mask_name] = load_mask(os.path.join(DATASET_DIR, "Masks", mask_name))
                img_mask =  dict_masks[mask_name]
                int_x, int_y = int(new_row["Fixation_X'"]), int(new_row["Fixation_Y'"])
                if point_in_image(int_x, int_y):
                    new_row["Fixation_in_Image"] = True
                    if is_in_mask(int_x, int_y, img_mask):
                        new_row["Fixation_in_Mask"] = True
            new_row["Event_type"] = "Fixation"
            if stimulus in mouse_trial:
                new_row["Mouse_Confirm_Trial"] = True
            list_for_df.append(new_row)
    
    for stimulus, values in dict_mouse.items():
        (c_time, time, mX, mY) = values
        repetition_n = int(stimulus[-1])
        name = get_img_name_from_trial(stimulus)
        img_n = int(name.split("_")[0])
        new_row = data_dict_format.copy()
        new_row["Computer_time"] = c_time
        new_row["Event_time"] = time
        new_row["Image_n"] = img_n
        new_row["Stimulus_name"] = name
        new_row["Repetition_n"] = repetition_n
        new_row["Mouse_X"] = mX
        new_row["Mouse_Y"] = mY
        new_row["Mouse_X'"] = mX - 448
        new_row["Mouse_Y'"] = mY - 156
        if "practice" not in name:
            mask_name = get_mask_name(name)
            img_mask =  dict_masks[mask_name]
            int_x, int_y = int(new_row["Mouse_X'"]), int(new_row["Mouse_Y'"])
            if point_in_image(int_x, int_y):
                new_row["Mouse_in_Image"] = True
                if is_in_mask(int_x, int_y, img_mask):
                    new_row["Mouse_in_Mask"] = True
            new_row["Event_type"] = "MouseClick"
        if stimulus in mouse_trial:
            new_row["Mouse_Confirm_Trial"] = True
        list_for_df.append(new_row)
    
    df = pd.DataFrame(list_for_df)
    for stimulus_n, img_n in enumerate(df.Image_n.unique()):
        df.loc[df.Image_n == img_n,"Stimulus_n"] = stimulus_n
    df["Stimulus_n"] = df["Stimulus_n"].apply(int)
    df = df.sort_values("Computer_time")
    for stimulus_n, img_n in enumerate(df.Image_n.unique()):
        stimulus_data_mask = df.Image_n == img_n
        trial_start_mask = df.Event_type == "TrialStart"
        r_mask = df.Repetition_n == 0
        not_confirm_trial_mask = df.Mouse_Confirm_Trial == False
        is_mouse_click_mask = df["Event_type"] == "MouseClick"
        # To find the start mask we want:
        # The stimulus data AND is TrialStart event AND is repetition 0 AND is not the confirm trial
        start_time_mask = stimulus_data_mask & trial_start_mask & r_mask & not_confirm_trial_mask
        start_time = df[start_time_mask].iloc[0]["Event_time"]
        
        # To find the response time mask we want:
        # The stimulus data AND is not the confirm trial AND the event is a mouse_click
        combine_mask = stimulus_data_mask & not_confirm_trial_mask & is_mouse_click_mask
        if (combine_mask).any():
            df.loc[combine_mask,"Response_time"] = df.loc[combine_mask,"Event_time"] - start_time
    df = df.reset_index()
    df = df.drop("index",axis=1)
    return df

def plot_data_from_recording_df(recording_df, trial_name, repetition_n):
    stimulus_mask = (recording_df["Stimulus_name"] == trial_name).to_numpy(dtype=bool)
    repetition_mask = (recording_df["Repetition_n"] == repetition_n).to_numpy(dtype=bool)
    select_data = stimulus_mask & repetition_mask
    trial_data = recording_df.iloc[select_data,:]
    fixation_points = trial_data[trial_data.Event_type == "Fixation"].loc[:,["Fixation_X","Fixation_Y"]].to_numpy(dtype=int)
    mouse_points = trial_data[trial_data.Event_type == "MouseClick"].loc[:,["Mouse_X","Mouse_Y"]].to_numpy(dtype=int)

    # These values are always the same, they are taken by checking the
    # presented media positions
    fixation_points[:,0] = fixation_points[:,0] - 448
    fixation_points[:,1] = fixation_points[:,1] - 156
    img_name = trial_name
    if "practice" in img_name:
        print("Error: Practice images have no mask.")
        return
    x_outside_image = fixation_points[:,0] <= 1024
    y_outside_image = fixation_points[:,1] <= 728
    # Remove points outside the image
    fixation_points = fixation_points[x_outside_image & y_outside_image]
    
    mask_name = get_mask_name(trial_name)
    img = plt.imread(os.path.join(DATASET_DIR, "Images", f"{img_name}.jpg"))
    img_mask =  load_mask(os.path.join(DATASET_DIR, "Masks", mask_name))
    fix_in_mask = np.zeros(len(fixation_points), dtype=bool)
    
    for i, point in enumerate(fixation_points):
        if pd.isnull(point[0]):
            continue
        else:
            if is_in_mask(int(point[0]), int(point[1]), img_mask):
                fix_in_mask[i] = True
    
    fig, ax = plt.subplots()
    ax.imshow(img)
    ax.imshow(img_mask, alpha=0.3)
    ax.scatter(fixation_points[fix_in_mask,0], fixation_points[fix_in_mask,1], c="green", s=20, label="Fixation Points in mask")
    ax.scatter(fixation_points[~fix_in_mask,0], fixation_points[~fix_in_mask,1], c="orange", s=20, label="Fixation Points not in mask")
    if len(mouse_points > 0):
        mouse_points[:,0] = mouse_points[:,0] - 448
        mouse_points[:,1] = mouse_points[:,1] - 156
        mouse_click_in_mask = np.zeros(len(mouse_points), dtype=bool)
        for i, point in enumerate(mouse_points):
            if pd.isnull(point[0]):
                continue
            else:
                try:
                    if is_in_mask(int(point[0]), int(point[1]), img_mask):
                        mouse_click_in_mask[i] = True
                    ax.scatter(mouse_points[mouse_click_in_mask,0], mouse_points[mouse_click_in_mask,1], c="red", s=50, marker="x", label="Mouse click in mask")
                    ax.scatter(mouse_points[~mouse_click_in_mask,0], mouse_points[~mouse_click_in_mask,1], c="blue", s=50, marker="x", label="Mouse click outside mask")
                except:
                    print("WARNING: Mouse was clicked outside image area.", mouse_points)

    ax.legend()
    plt.plot()
    

def plot_data_from_recording_df_img_n(recording_df, participant_name, image_n, repetition="ALL", trials_to_plot="BOTH"):
    assert participant_name in recording_df["Participant"].unique()
    assert trials_to_plot in ["BOTH","CONFIRM", "REPETITION"], "trials_to_plot needs to be in [BOTH, CONFIRM, REPETITION]."
    if trials_to_plot == "BOTH":
        assert repetition == "ALL", "trials_to_plot==BOTH does not suport filtering on repetitions."
    if trials_to_plot == "REPETITION":
        assert type(repetition) == int, "repetition needs to be an integer >= 0 AND <= 2 ."
        assert repetition >= 0 and repetition <= 2, "repetition needs to be an integer >= 0 AND <= 2 ."
    if trials_to_plot == "CONFIRM":
        assert repetition == 0 or repetition == "ALL", "If using trials_to_plt==CONFIRM; repetition needs to be set to ALL or 0"
    if trials_to_plot == "BOTH":
        print("INFO: When plotting BOTH, the fixations are only from the repetition trials.")
        stimulus_mask = (recording_df["Image_n"] == image_n).to_numpy(dtype=bool)
    elif trials_to_plot == "CONFIRM":
        stimulus_mask = ((recording_df["Image_n"] == image_n) & (recording_df["Mouse_Confirm_Trial"] == True)).to_numpy(dtype=bool)
    else:
        stimulus_mask = ((recording_df["Image_n"] == image_n) & (recording_df["Repetition_n"] == repetition) & (recording_df["Mouse_Confirm_Trial"] == False)).to_numpy(dtype=bool)
        
    stimulus_mask = (stimulus_mask & (recording_df["Participant"] == participant_name)).to_numpy(dtype=bool)
    trial_data = recording_df.iloc[stimulus_mask,:]
    if trials_to_plot == "BOTH":
        fixation_data = trial_data[(trial_data.Event_type == "Fixation") & (trial_data["Mouse_Confirm_Trial"] == False)].loc[:,["Fixation_X","Fixation_Y", "Fixation_duration"]].to_numpy(dtype=int)
    else:
        fixation_data = trial_data[(trial_data.Event_type == "Fixation")].loc[:,["Fixation_X","Fixation_Y", "Fixation_duration"]].to_numpy(dtype=int)
    mouse_points = trial_data[trial_data.Event_type == "MouseClick"].loc[:,["Mouse_X","Mouse_Y"]].to_numpy(dtype=int)
    fixation_points = fixation_data[:, :-1]
    fix_duration_size = fixation_data[:, -1]
    # These values are always the same, they are taken by checking the
    # presented media positions
    fixation_points[:,0] = fixation_points[:,0] - 448
    fixation_points[:,1] = fixation_points[:,1] - 156
    trial_names = np.unique(trial_data["Stimulus_name"])
    trial_name = trial_names[0] if ("_present_" in trial_names[0] or "_in_" in trial_names[0]) else trial_names[1]
    centerMass_x, centerMass_y = trial_data[trial_data.Event_type == "Fixation"][['Center_mass_X','Center_mass_Y']].iloc[0]
    img_name = trial_name
    if "practice" in img_name:
        print("Error: Practice images have no mask.")
        return
    x_inside_image = fixation_points[:,0] <= IMAGE_X_SIZE
    y_inside_image = fixation_points[:,1] <= IMAGE_Y_SIZE
    
    # Remove points outside the image
    fixation_points = fixation_points[x_inside_image & y_inside_image]
    fix_duration_size = fix_duration_size[x_inside_image & y_inside_image]
    
    mask_name = get_mask_name(trial_name)
    img = plt.imread(os.path.join(DATASET_DIR, "Images", f"{img_name}.jpg"))
    img_mask =  load_mask(os.path.join(DATASET_DIR, "Masks", mask_name))
    fix_in_mask = np.zeros(len(fixation_points), dtype=bool)
    
    for i, point in enumerate(fixation_points):
        if pd.isnull(point[0]):
            continue
        else:
            if is_in_mask(int(point[0]), int(point[1]), img_mask):
                fix_in_mask[i] = True
    
    fix_duration_size = (fix_duration_size/max(fix_duration_size) * 100) * 3
    
    fig, ax = plt.subplots(figsize=(16, 14), dpi=200)
    ax.imshow(img)
    ax.imshow(img_mask, alpha=0.3)
    ax.scatter(fixation_points[fix_in_mask,0], fixation_points[fix_in_mask,1], s=fix_duration_size[fix_in_mask], alpha=0.6, label="Fixation in AoI")
    ax.scatter(fixation_points[~fix_in_mask,0], fixation_points[~fix_in_mask,1], s=fix_duration_size[~fix_in_mask], alpha=0.6, label="Fixation not in AoI")
    ax.scatter(centerMass_x, centerMass_y, alpha=0.5, c="red", s=50, marker="x", label="Center of AoI")
    for i, point in enumerate(fixation_points):
        if i > len(fixation_points) - 5:
            txt = ax.text(point[0], point[1], i, fontsize=12, color="black")
            txt.set_path_effects([PathEffects.withStroke(linewidth=1, foreground='w')])
    if len(mouse_points > 0):
        mouse_points[:,0] = mouse_points[:,0] - 448
        mouse_points[:,1] = mouse_points[:,1] - 156
        mouse_click_in_mask = np.zeros(len(mouse_points), dtype=bool)
        for i, point in enumerate(mouse_points):
            if pd.isnull(point[0]):
                continue
            else:
                try:
                    if is_in_mask(int(point[0]), int(point[1]), img_mask):
                        mouse_click_in_mask[i] = True
                except:
                    print("WARNING: Mouse was clicked outside image area.", mouse_points)
        ax.scatter(mouse_points[mouse_click_in_mask,0], mouse_points[mouse_click_in_mask,1], c="red", s=50, alpha=0.8, marker="X", label="Mouse click in AoI")
        #ax.scatter(mouse_points[~mouse_click_in_mask,0], mouse_points[~mouse_click_in_mask,1], c="blue", s=50, alpha=0.8, marker="X", label="Mouse click outside mask")
    # From: https://stackoverflow.com/questions/4700614/how-to-put-the-legend-outside-the-plot
    # Shrink current axis by 20%
    box = ax.get_position()
    ax.set_position([box.x0, box.y0, box.width * 0.3, box.height])

    # Put a legend to the right of the current axis
    ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.05),
          fancybox=True, shadow=False, ncol=2)
    plt.axis('off')
    plt.plot()

def plot_average_heatmap(recording_df, image_n, gaussianwh=100):
    # Gazeheatplot is based on https://github.com/TobiasRoeddiger/GazePointHeatMap
    stimulus_mask = ((recording_df["Image_n"] == image_n)).to_numpy(dtype=bool)
    trial_data = recording_df.iloc[stimulus_mask,:]
    trial_names = np.unique(trial_data["Stimulus_name"])
    print(trial_names)
    trial_name = trial_names[0] if ("_present_" in trial_names[0] or "_in_" in trial_names[0]) else trial_names[1]
    img_name = trial_name
    mask_name = get_mask_name(trial_name)
    to_plot_dataset = pd.DataFrame(columns=recording_df.columns)
    for participant in recording_df["Participant"].unique():
        trial_rep = recording_df[((recording_df.Participant == participant) 
                        & (recording_df['Image_n'] == image_n) 
                        & (recording_df['Mouse_Confirm_Trial'] == False))]

        fix_time_filter = trial_rep['Event_time'].max() 
        response_time = trial_rep['Response_time'].min()
        if not pd.isnull(response_time):
            when_spot = trial_rep['Response_time'].argmin()
            fix_time_filter = trial_rep.iloc[when_spot]["Event_time"]
        fixation_data = trial_rep[(trial_rep['Event_type'] == "Fixation") & (trial_rep['Event_time'] <= fix_time_filter)]
        to_plot_dataset = pd.concat([to_plot_dataset,fixation_data])
    fixation_data = to_plot_dataset[(to_plot_dataset.Event_type == "Fixation") & (to_plot_dataset["Mouse_Confirm_Trial"] == False)].loc[:,["Fixation_X'","Fixation_Y'", "Fixation_duration"]].to_numpy(dtype=int)
    draw_heatmap(fixation_data, (IMAGE_X_SIZE, IMAGE_Y_SIZE), imagefile=os.path.join(DATASET_DIR, "Images", f"{img_name}.jpg"), maskfile=mask_name, alpha=0.5,
                                 savefilename=None, gaussianwh=gaussianwh, gaussiansd=None)
    plt.plot()

def plot_data_for_trial(trial_name, start, end, data):
    # Get the points nomalized (0-1)
    fixation_points = data.loc[start:end,['Fixation point X (MCSnorm)', 'Fixation point Y (MCSnorm)']].to_numpy(dtype=float)
    fixation_points = fixation_points[~np.isnan(fixation_points[:,0]),:]
    plot_points = data.loc[start:end,['Gaze point X (MCSnorm)', 'Gaze point Y (MCSnorm)']].to_numpy(dtype=float)
    plot_points = plot_points[~np.isnan(plot_points[:,0]),:]
    # Scale to image size:
    plot_points[:,0] = plot_points[:,0] * 1024 # Image X
    plot_points[:,1] = plot_points[:,1] * 768 # Image Y
    fixation_points[:,0] = fixation_points[:,0] * 1024
    fixation_points[:,1] = fixation_points[:,1] * 768
    img_name = get_img_name_from_trial(trial_name)
    mask_name = get_mask_name(trial_name)
    img = plt.imread(os.path.join(DATASET_DIR, "Images", f"{img_name}.jpg"))
    img_mask =  plt.imread(os.path.join(DATASET_DIR, "Masks", mask_name), format="png")
    mask_array = np.zeros(img_mask.shape[:2],dtype=bool)
    mask_array[img_mask.mean(axis=2) < 0.5] = True
    gaze_in_mask = np.zeros(len(plot_points), dtype=bool)
    fix_in_mask = np.zeros(len(fixation_points), dtype=bool)
    for i, point in enumerate(plot_points):
        if pd.isnull(point[0]):
            continue
        else:
            if mask_array[int(point[1]),int(point[0])]:
                gaze_in_mask[i] = True
    for i, point in enumerate(fixation_points):
        if pd.isnull(point[0]):
            continue
        else:
            if mask_array[int(point[1]),int(point[0])]:
                fix_in_mask[i] = True
    fig, ax = plt.subplots()
    ax.imshow(img)
    ax.imshow(mask_array, alpha=0.3)
    ax.scatter(plot_points[gaze_in_mask,0], plot_points[gaze_in_mask,1], c="r", alpha=0.5, label="Gaze Points in mask")
    ax.scatter(plot_points[~gaze_in_mask,0], plot_points[~gaze_in_mask,1], c="b", alpha=0.5, label="Gaze Points not in mask")
    ax.scatter(fixation_points[fix_in_mask,0], fixation_points[fix_in_mask,1], c="green", s=20, label="Fixation Points in mask")
    ax.scatter(fixation_points[~fix_in_mask,0], fixation_points[~fix_in_mask,1], c="orange", s=20, label="Fixation Points not in mask")
    ax.legend()
    plt.plot()



def calculate_features_from_df(df, remove_example_trials = True, remove_data_after_spot_c = False):
    feature_dict = {
        "Participant": np.NaN,
        "Image_n": np.NaN,
        "Seq_Trial_N": np.NaN,
        "Trial_Start": "", # Can be Present / Absent
        "Response_time": np.NaN,
        "Total_Fixation_count": np.NaN,
        "Count_Fixations_in_mask": np.NaN,
        "Total_Fixation_time_in_mask": np.NaN,
        "Total_Fixation_time_out_mask": np.NaN,
        "Time_of_first_Fixation_in_mask": np.NaN,
        "Fixate_in_both_conditions": np.NaN,
        "Fixate_in_any_conditions": np.NaN,
        "Fixated_once_in_mask": np.NaN,
        "Min_Dist_to_mask_center": np.NaN,
        "Max_Dist_to_mask_center": np.NaN,
        "Avg_Dist_to_mask_center":np.NaN,
        "Clicked_on_mask": np.NaN

    }
    data_list = list()
    for participant in df["Participant"].unique():
        for trial_n in df["Image_n"].unique():
            if remove_example_trials and trial_n in [1,2]:
                continue
            new_dict = feature_dict.copy()
            trial_rep = df[((df.Participant == participant) 
                            & (df['Image_n'] == trial_n) 
                            & (df['Mouse_Confirm_Trial'] == False))]

            trial_mouse_click = df[((df.Participant == participant) 
                            & (df['Image_n'] == trial_n) 
                            & (df['Mouse_Confirm_Trial'] == True))]
            try:
                new_dict["Response_time"] = trial_rep['Response_time'].min()
                fix_time_filter = trial_rep['Event_time'].max() 
                if remove_data_after_spot_c:
                    if not pd.isnull(new_dict["Response_time"]):
                        when_spot = trial_rep['Response_time'].argmin()
                        fix_time_filter = trial_rep.iloc[when_spot]["Event_time"]
                fixation_data = trial_rep[(trial_rep['Event_type'] == "Fixation") & (trial_rep['Event_time'] <= fix_time_filter)]
                fixate_both_absent_present = trial_rep.groupby("Stimulus_name").max()["Fixation_in_Mask"].values.all()
                # fixate_in_one_condition = trial_rep.groupby("Stimulus_name").max()["Fixation_in_Mask"].values.any()
                boolean_list = trial_rep.groupby("Stimulus_name").max()["Fixation_in_Mask"].values
                fixate_in_one_condition = boolean_list[0] ^ boolean_list[1] #takes exclusive or
                fixate_once_in_mask =  trial_rep["Fixation_in_Mask"].max()
                new_dict["Fixate_in_both_conditions"] = int(fixate_both_absent_present)
                new_dict["Fixate_in_any_conditions"] = int(fixate_in_one_condition)
                new_dict["Fixated_once_in_mask"] = int(fixate_once_in_mask)
                new_dict["Participant"] = participant
                new_dict["Image_n"] = trial_n
                new_dict["Seq_Trial_N"] = trial_rep['Stimulus_n'].iloc[0]
                new_dict["Trial_Start"] = "Present" if ("_present_" in trial_rep['Stimulus_name'].iloc[0] or "_in_" in trial_rep['Stimulus_name'].iloc[0])  else "Absent"
                new_dict["Total_Fixation_count"] = len(fixation_data)
                new_dict["Count_Fixations_in_mask"] = len(fixation_data[(fixation_data["Fixation_in_Mask"] == True)])
                new_dict["Total_Fixation_time_in_mask"] = fixation_data[(fixation_data["Fixation_in_Mask"] == True)]["Fixation_duration"].sum()/1000
                new_dict["Total_Fixation_time_out_mask"] = fixation_data[(fixation_data["Fixation_in_Mask"] == False)]["Fixation_duration"].sum()/1000
                new_dict["Clicked_on_mask"] = 1 if trial_mouse_click["Mouse_in_Mask"].max() else 0
                new_dict["Time_of_first_Fixation_in_mask"] = fixation_data[fixation_data["Fixation_in_Mask"] == True]['Event_time'].min() - trial_rep['Event_time'].min()
                new_dict["Min_Dist_to_mask_center"] = fixation_data["Fixation_dist_to_mask_center"].min()
                new_dict["Max_Dist_to_mask_center"] = fixation_data["Fixation_dist_to_mask_center"].max()
                new_dict["Avg_Dist_to_mask_center"] = fixation_data["Fixation_dist_to_mask_center"].mean()
                data_list.append(new_dict)
            except Exception as e:
                print(e)
                print(f"WARNING, failed to caluclate for Participant: {participant} on Image: {trial_n}")
    return pd.DataFrame(data_list)


def load_dataset_data():
    DATA_DIR = "data"
    DATASET_DIR = os.path.join("ChangeBlindnessTTM", "ChangeBlindnessTTM-ExpComplete", "Data", "CBDatabase")
    dataset_data = pd.read_excel(os.path.join(DATASET_DIR, "CBDatabase_Size_Eccentricity_RT.xlsx"))
    new_names = dataset_data.columns.to_numpy()
    for i, name in enumerate(dataset_data.iloc[0]):
        if not pd.isnull(name):
            new_names[i] = name[:20].strip()
    dataset_data.columns = new_names
    return dataset_data  

def getCenterOfMass(trial_name):
    img_name = get_img_name_from_trial(trial_name)
    mask_name = get_mask_name(img_name)
    img = plt.imread(os.path.join(DATASET_DIR, "Images", f"{img_name}.jpg"))
    img_mask =  np.array(load_mask(os.path.join(DATASET_DIR, "Masks", mask_name)),dtype=int)
    properties = regionprops(img_mask, img_mask)
    center_of_mass = properties[0].centroid
    return center_of_mass[::-1] # Make sure it's x,y

def saveGeneratedCenterOfMass(trial_list):
    results = {}
    for trial in trial_list:
        center = getCenterOfMass(trial)
        results[trial] = center
    return center


def getFixationFeaturesDf(combined_all_dataframes,remove_data_after_spot_c = False):

    # for odd repetition no
    df_odd = combined_all_dataframes[combined_all_dataframes.Repetition_n % 2== 1.0]
    df_feat_odd = df_odd[['Participant','Stimulus_n','Mouse_in_Mask','Fixation_in_Mask','Image_n','Response_time']].groupby(by = ['Participant','Image_n']).max()
    df_feat_odd.reset_index(inplace = True) 
    # df_feat_odd = df_feat_odd[(df_feat_odd.Mouse_in_Mask == False) & (df_feat_odd.Fixation_in_Mask == True)]

    # df_even.shape,df_odd.shape
    # for even repetition no
    df_even = combined_all_dataframes[combined_all_dataframes.Repetition_n % 2== 0.0]
    df_feat_even = df_even[['Participant','Stimulus_n','Mouse_in_Mask','Fixation_in_Mask','Image_n','Response_time']].groupby(by = ['Participant','Image_n']).max()
    df_feat_even.reset_index(inplace = True) 
    # df_feat_even = df_feat_even[(df_feat_even.Mouse_in_Mask == False) & (df_feat_even.Fixation_in_Mask == True)]

    # make df feat
    df_feat = combined_all_dataframes[['Participant','Stimulus_n','Mouse_in_Mask','Fixation_in_Mask','Image_n','Response_time']].groupby(by = ['Participant','Image_n']).max()
    df_feat.reset_index(inplace = True) 
    # add fixation in mask columns
    df_feat['fixation_in_mask_even'] = df_feat_even.Fixation_in_Mask
    df_feat['fixation_in_mask_odd']  = df_feat_odd.Fixation_in_Mask
    df_feat['fixation_in_mask_both'] = (df_feat_even.Fixation_in_Mask == True) & (df_feat_odd.Fixation_in_Mask == True)
    
    #get tiagos df feat
    df_feat2 = calculate_features_from_df(combined_all_dataframes, remove_example_trials = True,remove_data_after_spot_c  = remove_data_after_spot_c )
    #merge df feat and tiagos df_feat
    df_new = df_feat2.merge(df_feat[['Participant','Image_n','fixation_in_mask_both','Fixation_in_Mask','Mouse_in_Mask']], how = 'left',on =['Image_n','Participant'])
    df_new = df_new.drop_duplicates()
    df_new["Fixation_in_Mask"] = df_new["Fixation_in_Mask"].astype(int)
    df_new["fixation_in_mask_both"] = df_new["fixation_in_mask_both"].astype(int)    
    return df_new