import torch
import pandas as pd
import os
from main_utils import save_output, video_to_image, save_video_output

# Check Gpu
print(f'gpu: {torch.cuda.get_device_name(0)}')

# Model
model = torch.hub.load('ultralytics/yolov5', 'yolov5x')  # or yolov5n - yolov5x6, custom

# param setting
video_path = './data/BOSS/NoEvent2/No_Event2.Cam1.avi'
output_img_folder = './output/BOSS/NoEvent2'
output_video_folder = './output/BOSS/NoEvent2/video'
fps = 25

# video to image
video_to_image(video_path)

# img list
img_folder = video_path+'.FrameImages'
img_list = os.listdir(img_folder)

# inference
threshold = 0.5
batch_size = 100
num_iter = (len(img_list) // batch_size) + (1 if len(img_list) % batch_size > 0 else 0)

output_df = pd.DataFrame(columns=['filename', 'pred_no', 'object_no', 'eval', 'gt_xmin', 'gt_ymin', 'gt_xmax', 'gt_ymax',
                                  'pred_xmin', 'pred_ymin', 'pred_xmax', 'pred_ymax', 'IoU'])
for ith_iter in range(num_iter):
    _img_list = img_list[(ith_iter * batch_size):((ith_iter + 1) * batch_size)]

    # inference
    preds = model([f'{img_folder}/{img}' for img in _img_list])
    pred_list = preds.pandas().xyxy

    for i in range(len(_img_list)):
        # set data
        pred_df = pred_list[i]
        pred_df = pred_df.loc[(pred_df['name'] == 'person') & (pred_df['confidence'] > threshold), :]

        if len(pred_df) > 0:
            for j in range(len(pred_df)):
                output_df = pd.concat((output_df,
                                       pd.DataFrame([{'filename': _img_list[i],
                                                      'pred_no': j,
                                                      'object_no': -1,
                                                      'eval': None,
                                                      'gt_xmin': -1,
                                                      'gt_ymin': -1,
                                                      'gt_xmax': -1,
                                                      'gt_ymax': -1,
                                                      'pred_xmin': pred_df.iloc[j, 0],
                                                      'pred_ymin': pred_df.iloc[j, 1],
                                                      'pred_xmax': pred_df.iloc[j, 2],
                                                      'pred_ymax': pred_df.iloc[j, 3],
                                                      'IoU': 0}])))
    print(f'inference progress: {ith_iter+1}/{num_iter}', end='\r')
print('')

# save output
save_output(output_df, img_folder=img_folder, output_folder=output_img_folder)

# make video
save_video_output(img_folder=img_folder,
                  detected_img_folder=output_img_folder+'/'+'all',
                  output_folder=output_video_folder,
                  fps=fps)
