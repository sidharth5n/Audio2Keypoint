import matplotlib
matplotlib.use("Agg")
import subprocess
import os
import numpy as np
from matplotlib import cm, pyplot as plt
from PIL import Image
from common.consts import BASE_KEYPOINT, CHIN_KEYPOINTS, LEFT_BROW_KEYPOINTS, RIGHT_BROW_KEYPOINTS, NOSE_KEYPOINTS, LEFT_EYE_KEYPOINTS, RIGHT_EYE_KEYPOINTS, OUTER_LIP_KEYPOINTS, INNER_LIP_KEYPOINTS, LINE_WIDTH_CONST

def plot_chin_keypoints(keypoints, alpha=None, line_width=LINE_WIDTH_CONST):
    _keypoints = np.array(CHIN_KEYPOINTS)
    plt.plot(keypoints[0][_keypoints], keypoints[1][_keypoints], linewidth=line_width, alpha=alpha,
                     color='darkgreen')

def plot_right_brow_keypoints(keypoints, alpha=None, line_width=LINE_WIDTH_CONST):
    _keypoints = np.array(RIGHT_BROW_KEYPOINTS)
    plt.plot(keypoints[0][_keypoints], keypoints[1][_keypoints], linewidth=line_width, alpha=alpha,
                     color='darkorange')

def plot_left_brow_keypoints(keypoints, alpha=None, line_width=LINE_WIDTH_CONST):
    _keypoints = np.array(LEFT_BROW_KEYPOINTS)
    plt.plot(keypoints[0][_keypoints], keypoints[1][_keypoints], linewidth=line_width, alpha=alpha,
                             color='darkorange')


def plot_nose_keypoints(keypoints, alpha=None, line_width=LINE_WIDTH_CONST):
    _keypoints = np.array(BASE_KEYPOINT + NOSE_KEYPOINTS)
    plt.plot(keypoints[0][_keypoints], keypoints[1][_keypoints], linewidth=line_width, alpha=alpha,
                     color='blue')

def plot_left_eye_keypoints(keypoints, alpha=None, line_width=LINE_WIDTH_CONST):
    _keypoints = np.array(LEFT_EYE_KEYPOINTS)
    plt.plot(keypoints[0][_keypoints], keypoints[1][_keypoints], linewidth=line_width, alpha=alpha,
                             color='red')


def plot_right_eye_keypoints(keypoints, alpha=None, line_width=LINE_WIDTH_CONST):
    _keypoints = np.array(RIGHT_EYE_KEYPOINTS)
    plt.plot(keypoints[0][_keypoints], keypoints[1][_keypoints], linewidth=line_width, alpha=alpha,
                     color='red')

def plot_lip_keypoints(keypoints, alpha=None, line_width=LINE_WIDTH_CONST):
    _keypoints = np.array(OUTER_LIP_KEYPOINTS + INNER_LIP_KEYPOINTS)
    plt.plot(keypoints[0][_keypoints], keypoints[1][_keypoints], linewidth=line_width, alpha=alpha,
             color='darkmagenta')

def draw_pose(img, keypoints, img_width=64, img_height=64, output=None, title=None, title_x=1, cm=cm.rainbow,
              alpha_img=0.5, alpha_keypoints=None, fig=None, line_width=LINE_WIDTH_CONST):
    '''
    Note: calling functions must call plt.close() to avoid a memory blowup.
    '''
    if fig is None:
        plt.close("all")
        fig = plt.figure(figsize=(6,4))

    plt.axis('off')

    if img != None:
        img = Image.open(img)
        img_width, img_height = img.size
    else:
        img = Image.new(mode='RGB', size=(img_width, img_height), color='white')

    plt.imshow(img, alpha=alpha_img)
    plot_chin_keypoints(keypoints, alpha_keypoints, line_width)
    plot_left_brow_keypoints(keypoints, alpha_keypoints, line_width)
    plot_right_brow_keypoints(keypoints, alpha_keypoints, line_width)
    plot_nose_keypoints(keypoints, alpha_keypoints, line_width)
    plot_left_eye_keypoints(keypoints, alpha_keypoints, line_width)
    plot_right_eye_keypoints(keypoints, alpha_keypoints, line_width)
    plot_lip_keypoints(keypoints, alpha_keypoints, line_width)
    ax = fig.get_axes()[0]
    ax.set_xlim(0, img_width)
    ax.set_ylim(img_height, 0)
    if title:
        plt.title(title, x=title_x)

    if output:
        plt.savefig(output)
        plt.close()


def draw_side_by_side_poses(img, keypoints1, keypoints2, output=None, show=True,
                            title="Prediction %s Ground Truth" % (7 * ' '), img_size=(64,64)):
    plt.close("all")
    fig = plt.figure(figsize=(6,4), dpi=400)
    plt.axis('off')
    if title:
        plt.title(title)
    if img != None:
        img = Image.open(img)
    else:
        img = Image.new(mode='RGB', size=img_size, color='white')

    plt.imshow(img, alpha=0.5)

    for keypoints in [keypoints1, keypoints2]:
        plot_chin_keypoints(keypoints)
        plot_left_brow_keypoints(keypoints)
        plot_right_brow_keypoints(keypoints)
        plot_nose_keypoints(keypoints)
        plot_right_eye_keypoints(keypoints) 
        plot_left_eye_keypoints(keypoints)
        plot_lip_keypoints(keypoints)

    if show:
        plt.show()
    if output is not None:
        plt.savefig(output)
    return fig


def save_side_by_side_video(temp_folder, keypoints1, keypoints2, output_fn, delete_tmp=True):
    if not (os.path.exists(temp_folder)):
        os.makedirs(temp_folder)

    if not (os.path.exists(os.path.dirname(output_fn))):
        os.makedirs(temp_folder)

    output_fn_pattern = os.path.join(temp_folder, '%04d.jpg')

    #diff = len(keypoints2) - len(keypoints1)
    #if diff > 0:
    #    conditioned_keypoints = keypoints2[:diff]
    #    keypoints2 = keypoints2[diff:]
    #    for i in range(len(conditioned_keypoints)):
    #        draw_pose(img=None, keypoints=conditioned_keypoints[i], img_width=1200, img_height=1200,
    #                  output=output_fn_pattern % i, title="Input", title_x=0.63)

    for j in range(len(keypoints1)):
        draw_side_by_side_poses(None, keypoints1[j], keypoints2[j], output=output_fn_pattern % (j), show=False)
        plt.close()
    #print(j)
    #import pdb
    #pdb.set_trace()

    create_mute_video_from_images(output_fn, temp_folder)
    if delete_tmp:
        subprocess.call('rm -R "%s"' % (temp_folder), shell=True)


def create_mute_video_from_images(output_fn, temp_folder):
    '''
    :param output_fn: output video file name
    :param temp_folder: contains images in the format 0001.jpg, 0002.jpg....
    :return:
    '''
    subprocess.call('ffmpeg -loglevel panic -r 25 -f image2 -i "%s" -r 25 "%s" -y' % (
        os.path.join(temp_folder, '%04d.jpg'), output_fn), shell=True)


def save_video_from_audio_video(audio_input_path, input_video_path, output_video_path):
    subprocess.call(
        'ffmpeg -loglevel panic -i "%s" -i "%s" -strict -2 "%s" -y' % (
        audio_input_path, input_video_path, output_video_path),
        shell=True)
