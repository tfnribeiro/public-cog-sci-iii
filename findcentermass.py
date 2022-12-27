# https://stackoverflow.com/questions/48888239/finding-the-center-of-mass-in-an-image
from skimage.measure import regionprops
from utils import *

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
