import torch
import pandas as pd
import os
from main_utils import calc_iou, load_annotation_df, save_output

# Check Gpu
print(f'gpu: {torch.cuda.get_device_name(0)}')

# Model
model = torch.hub.load('ultralytics/yolov5', 'yolov5x')  # or yolov5n - yolov5x6, custom

# path setting
base_folder = './data/VOCdevkit/VOC2012'
img_folder = f'{base_folder}/JPEGImages'
ann_folder = f'{base_folder}/Annotations'
img_list = os.listdir(img_folder)

# evaluation
threshold = 0.5
iou_threshold = 0.5
iou_df = pd.DataFrame(columns=['filename', 'pred_no', 'object_no', 'eval', 'gt_xmin', 'gt_ymin', 'gt_xmax', 'gt_ymax',
                               'pred_xmin', 'pred_ymin', 'pred_xmax', 'pred_ymax', 'IoU'])

batch_size = 100
num_iter = (len(img_list) // batch_size) + (1 if len(img_list) % batch_size > 0 else 0)
for ith_iter in range(num_iter):
    _img_list = img_list[(ith_iter * batch_size):((ith_iter + 1) * batch_size)]

    # inference
    preds = model([f'{img_folder}/{img}' for img in _img_list])
    preds.save()
    pred_list = preds.pandas().xyxy

    for i in range(len(_img_list)):
        # set data
        pred_df = pred_list[i]
        pred_df = pred_df.loc[(pred_df['name'] == 'person') & (pred_df['confidence'] > threshold), :]
        label_df = load_annotation_df(_img_list[i], ann_folder)

        # calculate IoU
        tmp_iou_df = pd.DataFrame(columns=iou_df.columns)
        for j in range(len(pred_df)):
            tmp_df = pd.DataFrame(columns=iou_df.columns)
            for k in range(len(label_df)):
                # label과 pred 값을 비교. 모든 경우에 대해 IoU 구함.
                _iou = calc_iou(pred_df.iloc[j, 0:4].values, label_df.iloc[k, 2:6].values)
                tmp_df = pd.concat((tmp_df, pd.DataFrame([{'filename': _img_list[i],
                                                           'pred_no': j,
                                                           'object_no': k,
                                                           'eval': None,
                                                           'gt_xmin': label_df.iloc[k, 2],
                                                           'gt_ymin': label_df.iloc[k, 3],
                                                           'gt_xmax': label_df.iloc[k, 4],
                                                           'gt_ymax': label_df.iloc[k, 5],
                                                           'pred_xmin': pred_df.iloc[j, 0],
                                                           'pred_ymin': pred_df.iloc[j, 1],
                                                           'pred_xmax': pred_df.iloc[j, 2],
                                                           'pred_ymax': pred_df.iloc[j, 3],
                                                           'IoU': _iou}])))

            if len(tmp_df) == 0 or max(tmp_df['IoU']) < iou_threshold:
                # FP: label이 존재하지 않거나, IoU가 iou_threshold 이하인 label만 존재할 경우, FP 처리
                tmp_iou_df = pd.concat((tmp_iou_df,
                                        pd.DataFrame([{'filename': _img_list[i],
                                                       'pred_no': j,
                                                       'object_no': -1,
                                                       'eval': 'FP',
                                                       'gt_xmin': -1,
                                                       'gt_ymin': -1,
                                                       'gt_xmax': -1,
                                                       'gt_ymax': -1,
                                                       'pred_xmin': pred_df.iloc[j, 0],
                                                       'pred_ymin': pred_df.iloc[j, 1],
                                                       'pred_xmax': pred_df.iloc[j, 2],
                                                       'pred_ymax': pred_df.iloc[j, 3],
                                                       'IoU': 0}])))
            else:  # TP: IoU가 가장 높은 label에 대해서 TP 처리
                tmp_df = tmp_df.loc[tmp_df['IoU'] == max(tmp_df['IoU']), :]
                tmp_df['eval'] = 'TP'
                tmp_iou_df = pd.concat((tmp_iou_df, tmp_df))

        for k in range(len(label_df)):
            if sum(tmp_iou_df['object_no'] == k) == 0:  # FN: IoU가 iou_threshold 이상인 prediction 값이 없을 경우 FN 처리.
                tmp_iou_df = pd.concat((tmp_iou_df,
                                        pd.DataFrame([{'filename': _img_list[i],
                                                       'pred_no': -1,
                                                       'object_no': k,
                                                       'eval': 'FN',
                                                       'gt_xmin': label_df.iloc[k, 2],
                                                       'gt_ymin': label_df.iloc[k, 3],
                                                       'gt_xmax': label_df.iloc[k, 4],
                                                       'gt_ymax': label_df.iloc[k, 5],
                                                       'pred_xmin': -1,
                                                       'pred_ymin': -1,
                                                       'pred_xmax': -1,
                                                       'pred_ymax': -1,
                                                       'IoU': 0}])))

        iou_df = pd.concat((iou_df, tmp_iou_df))

# print eval result
eval_result = iou_df['eval'].value_counts()
print(eval_result)
print(f'precision: {eval_result["TP"] / (eval_result["TP"] + eval_result["FP"])}')
print(f'recall: {eval_result["TP"] / (eval_result["TP"] + eval_result["FN"])}')

# save output
save_output(iou_df, img_folder='./data/VOCdevkit/VOC2012/JPEGImages', output_folder='./output/VOC2012')
