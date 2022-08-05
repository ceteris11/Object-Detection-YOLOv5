import xml.etree.ElementTree as ET
import numpy as np
import pandas as pd
import cv2
import os


def calc_iou(box1, box2):
    # box = (x1, y1, x2, y2)
    box1_area = (box1[2] - box1[0] + 1) * (box1[3] - box1[1] + 1)
    box2_area = (box2[2] - box2[0] + 1) * (box2[3] - box2[1] + 1)

    # obtain x1, y1, x2, y2 of the intersection
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])

    # compute the width and height of the intersection
    w = max(0, x2 - x1 + 1)
    h = max(0, y2 - y1 + 1)

    inter = w * h
    iou = inter / (box1_area + box2_area - inter)
    return iou


def load_annotation_df(img_name, ann_folder):
    # get path
    ann_name = img_name.split('.')[0] + '.xml'
    ann_path = f'{ann_folder}/{ann_name}'

    # get tree
    tree = ET.parse(ann_path)
    root = tree.getroot()

    # parse tree
    filename = root.find('filename').text
    obj_list = []
    for obj in root.findall('object'):
        obj_dict = {'filename': filename,
                    'name': obj.find('name').text,
                    'xmin': float(obj.find('bndbox').find('xmin').text),
                    'ymin': float(obj.find('bndbox').find('ymin').text),
                    'xmax': float(obj.find('bndbox').find('xmax').text),
                    'ymax': float(obj.find('bndbox').find('ymax').text)}
        obj_list.append(obj_dict)

    # make df
    ann_df = pd.DataFrame(obj_list)

    # filter without 'person' object
    ann_df = ann_df.loc[ann_df['name'] == 'person', :].reset_index(drop=True)

    return ann_df


def save_output(output_df, img_folder, output_folder, thickness=2):
    # set output folder
    f_list = ['FP', 'FN', 'all']
    for f in f_list:
        f_path = f'{output_folder}/{f}'
        if not os.path.isdir(f_path):
            os.makedirs(f_path)

    # get img list
    img_list = output_df['filename'].unique()

    # save output images
    for i in range(len(img_list)):
        img = img_list[i]
        _image = cv2.imread(f'{img_folder}/{img}')
        _output_df = output_df.loc[output_df['filename'] == img, :]

        for j in range(len(_output_df)):
            cv2.rectangle(_image,
                          (int(_output_df.iloc[j, 4]), int(_output_df.iloc[j, 7])),
                          (int(_output_df.iloc[j, 6]), int(_output_df.iloc[j, 5])),
                          color=(255, 0, 0),
                          thickness=thickness)
            cv2.rectangle(_image,
                          (int(_output_df.iloc[j, 8]), int(_output_df.iloc[j, 11])),
                          (int(_output_df.iloc[j, 10]), int(_output_df.iloc[j, 9])),
                          color=(0, 0, 255),
                          thickness=thickness)

        if 'FP' in _output_df['eval'].values:
            cv2.imwrite(f'{output_folder}/FP/{img}', _image)

        if 'FN' in _output_df['eval'].values:
            cv2.imwrite(f'{output_folder}/FN/{img}', _image)

        cv2.imwrite(f'{output_folder}/all/{img}', _image)
        print(f'save progress: {i+1}/{len(img_list)}', end='\r')
    print('')


def video_to_image(video_path):
    output_folder = video_path + '.FrameImages'
    if not os.path.isdir(output_folder):
        os.makedirs(output_folder)

    vidcap = cv2.VideoCapture(video_path)
    frame_count = 0
    while True:
        success, image = vidcap.read()
        if success:
            frame_count = frame_count + 1
            cv2.imwrite(f'{output_folder}/{str(frame_count).zfill(6)}.jpg', image)
            print(f'processed frame: {frame_count}', end='\r')
        else:
            print('')
            break


def save_video_output(img_folder, detected_img_folder, output_folder, fps):
    if not os.path.isdir(output_folder):
        os.makedirs(output_folder)

    # get img list
    img_list = os.listdir(img_folder)
    detected_img_list = os.listdir(detected_img_folder)

    # get video size
    sample_img = cv2.imread(img_folder + '/' + img_list[0])
    size = (sample_img.shape[1], sample_img.shape[0])

    # save video
    out = cv2.VideoWriter(output_folder + '/' + 'output.mp4', cv2.VideoWriter_fourcc(*'DIVX'), fps, size)
    detect_flag_list = np.isin(img_list, detected_img_list)
    for i in range(len(img_list)):
        img_name = img_list[i]
        if detect_flag_list[i]:
            img = cv2.imread(detected_img_folder + '/' + img_name)
        else:
            img = cv2.imread(img_folder + '/' + img_name)
        out.write(img)
        print(f'progress: {i + 1}/{len(img_list)}', end='\r')
    print('')
    out.release()
