"""This is an example file for using the track anything tool to export data that can be used as input for analysis. See https://arxiv.org/abs/2304.11968 for more details."""

import numpy as np
import pandas as pd
import os
from pathlib import Path
import cv2
import typing


def morph_filter_mask(mask: np.ndarray, kernel_size: int = 15, count: int = 1) -> np.ndarray:
    """Applies morphological filtering to a mask.

    Args:
        mask: boolean image
        kernel_size: kernel size for filters
        count: number of dilation-erosion chains to run

    Returns:
        mask after applying the morphological filtering
    """
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size)).astype(np.uint8)
    new_mask = cv2.erode(mask.astype(np.uint8), kernel, iterations=count)
    new_mask = cv2.dilate(new_mask, kernel, iterations=count)
    return new_mask


def annotate_video(video: Path, animals: typing.Dict = {}, mask_folder: typing.Optional[Path] = None) -> pd.DataFrame:
    """Renders a video interactively to annotate the frames where masks are valid.

    Args:
        video: filename of the video to render
        animals: dictionary of starts and ends for each identity
        mask_folder: optional folder for the mask files to render id numbers on top of the frame

    Returns:
        dataframe containing the annotations
    """
    vid_reader = cv2.VideoCapture(str(video))
    cur_frame = 0
    max_frames = vid_reader.get(cv2.CAP_PROP_FRAME_COUNT) - 1
    while True:
        # Read in the frame and display it
        succeed = vid_reader.set(cv2.CAP_PROP_POS_FRAMES, cur_frame)
        if not succeed:
            break
        succeed, frame = vid_reader.read(1)
        if not succeed:
            break
        # render id on frame if masks are available
        if mask_folder:
            frame_file = mask_folder / Path(os.path.splitext(os.path.basename(video))[0]) / Path(f"{int(cur_frame):05d}.npy")
            if os.path.exists(frame_file):
                mask = np.load(frame_file)
                available_ids = np.unique(mask)
                # remove background
                available_ids = available_ids[available_ids != 0]
                for cur_id in available_ids:
                    centroid = np.mean(np.argwhere(mask == cur_id), axis=0).astype(np.int64)[::-1]
                    # plot the morphologically filtered contour
                    new_mask = morph_filter_mask(mask == cur_id)
                    contours, _ = cv2.findContours(new_mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
                    frame = cv2.drawContours(frame, contours, -1, (0, 255, 0), 3)
                    # plot the identity over the centroid
                    frame = cv2.putText(frame, str(cur_id), centroid, cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2, cv2.LINE_AA)
        cv2.imshow(str(video), frame)
        key = cv2.waitKey(0)
        if key == ord('q'):
            break
        elif key == ord('p'):
            print("Animals annotated:")
            for key, val in animals.items():
                print(f"{key} -> {val}")
            continue
        elif key == ord('j'):
            cur_frame -= 1
            cur_frame = np.clip(cur_frame, 0, max_frames)
        elif key == ord('l'):
            cur_frame += 1
            cur_frame = np.clip(cur_frame, 0, max_frames)
        elif key == ord('i'):
            cur_frame = cur_frame + 10
            cur_frame = np.clip(cur_frame, 0, max_frames)
        elif key == ord('k'):
            cur_frame = cur_frame - 10
            cur_frame = np.clip(cur_frame, 0, max_frames)
        else:
            try:
                toggled_id = int(chr(key))
                print(f"ID {toggled_id} selected.")
                if str(toggled_id) in animals.keys():
                    animal_dict = animals[str(toggled_id)]
                else:
                    animal_dict = {}
                mod_key = cv2.waitKey(0)
                if mod_key == ord('s'):
                    animal_dict['start'] = cur_frame
                    animals[str(toggled_id)] = animal_dict
                elif mod_key == ord('e'):
                    animal_dict['end'] = cur_frame
                    animals[str(toggled_id)] = animal_dict
                elif mod_key == ord('q'):
                    continue
                else:
                    print("Option -> s: start, e: end, q: quit selection.")
            except ValueError:
                try:
                    print(f"Key {chr(key)} not found.\nl,j for seeking +/-1\ni,k for seeking +/-10\np to print annotations\nnumbers to select animal\nq to quit")
                except:
                    print("Error not detecting key character (no special keys allowed)...")
    cv2.destroyAllWindows()
    vid_reader.release()
    # Exporting the data
    df_list = [pd.DataFrame({'experiment': [os.path.splitext(os.path.basename(video))[0]], 'id': [key], 'start': [int(val['start'])], 'end': [int(val['end'])]}) for key, val in animals.items() if 'start' in val.keys() and 'end' in val.keys()]
    if len(df_list) > 0:
        return pd.concat(df_list)
    return pd.DataFrame(columns = ['experiment', 'id', 'start', 'end'])


def get_experiment(df: pd.DataFrame, filename: str) -> typing.Dict:
    """ Converts the long representation to a nested dict for use in annotate_video.

    Args:
        df: input dataframe with columns 'experiment', 'id', 'start', and 'end'
        filename: filter for 'experiment' column

    Return:
        dict of structure
        {
            id1: {
                start: int
                end: int
            },
            ...
        }
    """
    sub_df = df[df['experiment'] == filename]
    if len(df) == 0:
        return {}
    ret_val = {}
    for _, row in sub_df.iterrows():
        ret_val[row['id']] = {'start': row['start'], 'end': row['end']}
    return ret_val


# Variables to change
annotation_file = Path('result/annotations.csv')
results_folder = Path('result/mask/')
render_folder = Path('result/track/')
out_folder = Path('result/weight/')
suffix = 'moments_table1_circrect.csv'
# Do we want to annotate all videos whether or not they already have annotations?
annotate_all_videos = False
# Do we want to annotate videos at all (True) or just skip to exporting (False)?
annotate_any_videos = True


# Routine for generating the starts/ends of masks
meta_columns = ['experiment', 'id', 'start', 'end']
if os.path.exists(annotation_file):
    annotation_meta = pd.read_csv(annotation_file)
else:
    annotation_meta = pd.DataFrame(columns=meta_columns)

experiments = os.listdir(results_folder)

if annotate_any_videos:
    for experiment in experiments:
        cur_animals = get_experiment(annotation_meta, experiment)
        if len(cur_animals) != 0 and not annotate_all_videos:
            continue
        new_rows = annotate_video(str(render_folder / experiment) + '.mp4', cur_animals, results_folder)
        annotation_meta = pd.concat([annotation_meta[annotation_meta['experiment'] != experiment], new_rows])
    annotation_meta.to_csv(annotation_file, index=False)


###########################################################

def calculate_summary_df(video: Path, animal: int, mask_folder: Path, start: int, end: int) -> pd.DataFrame:
    """Extracts frame-wise summaries for an individual animal based on track anything predictions.

    Args:
        video: filename of the video to read
        animal: identity value in the mask file
        mask_folder: folder for the mask files
        start: starting frame to export data
        end: ending frame to export data

    Returns:
        dataframe containing the keyed predictions for all frames in the video
    """
    result_df = []
    for cur_frame in np.arange(start, end + 1):
        frame_file = mask_folder / Path(os.path.splitext(os.path.basename(video))[0]) / Path(f"{int(cur_frame):05d}.npy")
        if os.path.exists(frame_file):
            mask = np.load(str(frame_file))
            mask = morph_filter_mask(mask == animal)
            moments = cv2.moments(mask.astype(np.uint8))
            try:
                contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                moments['perimeter'] = 0
                for contour in contours:
                    moments['perimeter'] = moments['perimeter'] + cv2.arcLength(contour, True)
            except:
                moments['perimeter'] = 0
            result_df.append(pd.DataFrame(moments, index=[cur_frame]))
    if len(result_df) == 0:
        return None
    result_df = pd.concat(result_df)
    # Continue with the other calculations that are preserved
    result_df['frame'] = result_df.index
    result_df['x'] = result_df['m10'] / result_df['m00']
    result_df['y'] = result_df['m01'] / result_df['m00']
    result_df['x_mom'] = result_df['x']
    result_df['y_mom'] = result_df['y']
    result_df['x_pos'] = result_df['x']
    result_df['y_pos'] = result_df['y']
    result_df['seg_area'] = result_df['m00']
    result_df['v_x'] = np.gradient(result_df['x_pos'], 1)
    result_df['v_y'] = np.gradient(result_df['y_pos'], 1)
    result_df['v_mag'] = np.linalg.norm(result_df[['v_x', 'v_y']].values, axis=1)
    result_df['a'] = result_df['m20'] / result_df['m00'] - result_df['x']**2
    result_df['b'] = 2 * (result_df['m11'] / result_df['m00'] - result_df['x'] * result_df['y'])
    result_df['c'] = result_df['m02'] / result_df['m00'] - result_df['y']**2
    result_df['w'] = np.sqrt(8 * (result_df['a'] + result_df['c'] - np.sqrt(result_df['b']**2 + (result_df['a'] - result_df['c'])**2))) / 2
    result_df['l'] = np.sqrt(8 * (result_df['a'] + result_df['c'] + np.sqrt(result_df['b']**2 + (result_df['a'] - result_df['c'])**2))) / 2
    result_df['aspect_w/l'] = result_df['w'] / result_df['l']
    result_df['circularity'] = result_df['m00'] * 4 * np.pi / result_df['perimeter']**2
    result_df['rectangular'] = result_df['m00'] / (result_df['w'] * result_df['l'])
    result_df['eccentricity'] = np.sqrt(result_df['w']**2 + result_df['l']**2) / result_df['l']
    result_df['elongation'] = (result_df['mu20'] + result_df['mu02'] + (4 * result_df['mu11']**2 + (result_df['mu20'] - result_df['mu02'])**2)**0.5) / (result_df['mu20'] + result_df['mu02'] - (4 * result_df['mu11']**2 + (result_df['mu20'] - result_df['mu02'])**2)**0.5)
    result_df['area_x_eccen'] = result_df['m00'] * result_df['eccentricity']
    # Only keep the necessary metrics
    return result_df.reindex(['frame', 'x_pos', 'y_pos', 'v_x', 'v_y', 'v_mag', 'seg_area', 'm00', 'x_mom', 'y_mom', 'aspect_w/l', 'eccentricity', 'elongation', 'circularity', 'rectangular'], axis='columns')


# Routine for exporting segmentation data in the same format as master_pixel_analysis
os.makedirs(out_folder, exist_ok=True)
for _, row in annotation_meta.iterrows():
    cur_df = calculate_summary_df(Path(row['experiment']), row['id'], results_folder, row['start'], row['end'])
    filename = out_folder / Path(f"{row['id']}_{row['experiment']}_{suffix}")
    cur_df.to_csv(filename, index=False)
