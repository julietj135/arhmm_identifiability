from .augment import *


def get_raw_data(body, id, flipright=False):
    if body == "r":
        sub = pd.read_pickle(r'data/IndexFingertapping_3DPoses_Labels/Grab_3DPoses_Mediapipe/Righthand_fingertapping/Filtered_medsav_rhand_Subject_{}.pkl'.format(id))
    elif body == "l" or body == "lr":
        sub = pd.read_pickle(r'data/IndexFingertapping_3DPoses_Labels/Grab_3DPoses_Mediapipe/Lefthand_fingertapping/Filtered_medsav_lhand_Subject_{}.pkl'.format(id))
    elif body == "g":
        sub = pd.read_pickle(r'data/kpts_and_labels/all_sub_median_picker_3d_coordinates.pkl')[str(id)]
    return sub


def augment_data(sub, id, num_windows, body):
    if body in "lr":
        subs=augment_fingers(sub, num_windows, True)
        np.random.seed(id)
        indices = np.random.choice(len(subs),num_windows,replace=False)
        subs = [subs[ind] for ind in indices]
    else:
        subs=clean_gaits(sub, True)
        subs=subs[:num_windows]
    return subs