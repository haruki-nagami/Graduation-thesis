import argparse
from tqdm import tqdm
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from PIL import Image, ImageDraw
import numpy as np
import cv2
import os
import torch
import torch.nn as nn
import os
import random
from scipy.ndimage import binary_dilation
import pandas as pd

from robust_segment_anything import SamPredictor, sam_model_registry
from robust_segment_anything import SamAutomaticMaskGenerator, sam_model_registry
from robust_segment_anything.utils.transforms import ResizeLongestSide

checkpoint_path = './checkpoints/alpha_beta_gamma_conv_MSRA10k_best.pth'
model_type = "vit_h"
device = "cuda"

model = sam_model_registry[model_type](opt=None, checkpoint=checkpoint_path)

model.to(device=device)

print('Succesfully loading model from {}'.format(checkpoint_path))

propose_predictor= ResizeLongestSide(model.image_encoder.img_size)

def propose_predict(image_path, prompt, visualaize = False):
    prompt = np.expand_dims(prompt, axis=0) #batchを2で学習してるため
    prompt = np.repeat(prompt, repeats=2, axis=0)#batchを2で学習してるため
    label = torch.ones((prompt.shape[0],prompt.shape[1]))

    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image_t = torch.tensor(image, dtype=torch.uint8).unsqueeze(0).to(device)
    image_t = torch.permute(image_t, (0, 3, 1, 2))
    image_t = torch.cat([image_t, image_t], dim=0)#batchを2で学習してるため
    image_t_transformed = propose_predictor.apply_image_torch(image_t.float())

    batched_input = []
    
    for i in range(image_t_transformed.shape[0]):
        if len(prompt[0])<=3:
              input_label = torch.Tensor(np.ones(prompt.shape[1])).to(device)
              point_t = torch.Tensor(prompt).to(device)
        
              data_dict = {}
              data_dict['image'] = image_t_transformed[i]
              data_dict['point_coords'] = propose_predictor.apply_coords_torch(point_t[i], image_t.shape[-2:]).unsqueeze(0)
              data_dict['point_labels'] = input_label.unsqueeze(0)
              data_dict['original_size'] = image_t.shape[-2:]
              batched_input.append(data_dict)
        else:
               box_t = torch.Tensor(prompt).to(device)
               data_dict = {}
               data_dict['image'] = image_t_transformed[i]
               data_dict['boxes'] = propose_predictor.apply_boxes_torch(box_t[i], image_t.shape[-2:]).unsqueeze(0)
               data_dict['original_size'] = image_t.shape[-2:]
               batched_input.append(data_dict)
        
    with torch.no_grad():
                batched_output = model(None, batched_input, multimask_output=True, return_logits=False)
    
    alpha = batched_output[0]['alpha']
    beta = batched_output[0]['beta']
    gamma = batched_output[0]['gamma']
    mask = batched_output[0]['masks']
    mask = mask > 0.0
    
    return mask, alpha, beta, gamma

def clear_image_path_to_mask_path(clear_image_path:str) -> str:
    destination_dir='/workdir/data/all_data/test/masks'
    file_name=os.path.basename(clear_image_path)
    # ファイル名の拡張子を変更する
    new_file_name = os.path.splitext(file_name)[0] + '.npy'
    masks_path = os.path.join(destination_dir, new_file_name)

    return masks_path

def clear_image_path_to_degration_image_path(clear_image_path:str, degration_type:str) -> str:
    file_name = os.path.basename(clear_image_path)
    # 新しいパスを作成
    degration_image_path = os.path.join('./data/all_data/test', degration_type, file_name)
    return degration_image_path

def degration_image_path_to_clear_image_path(degration_image_path: str) -> str:
    file_name = os.path.basename(degration_image_path)  # ファイル名を取得
    # 新しいパスを作成
    clear_image_path = os.path.join('./data/all_data/test', 'clear', file_name)
    return clear_image_path

def get_point_prompt(clear_image_path, num_points):
        masks_path = clear_image_path_to_mask_path(clear_image_path)
        mask = np.load(masks_path)
        input_point = []
        index = np.where(mask == True)
        y_coord_np = index[1]
        x_coord_np = index[2]
        index_list = range(0, len(x_coord_np))
        first_index_list = range(0, int(len(x_coord_np)/2))
        second_index_list = range(int(len(x_coord_np)/2), len(x_coord_np))

        if num_points == 2:
            first_index = random.sample(first_index_list, 1)
            second_index = random.sample(second_index_list, 1)
            coord = [int(x_coord_np[first_index]), int(y_coord_np[first_index])]
            input_point.append(coord)
            coord = [int(x_coord_np[second_index]), int(y_coord_np[second_index])]
            input_point.append(coord)

        else:
            index = random.sample(index_list, num_points)
            for i in index:
                coord = [x_coord_np[i], y_coord_np[i]]
                input_point.append(coord)

        input_point = np.array(input_point)

        return input_point

def get_box_prompt(clear_image_path):
        masks_path = clear_image_path_to_mask_path(clear_image_path)
        mask = np.load(masks_path)
        index = np.where(mask == 1)
        y_coord_np = index[1]
        x_coord_np = index[2]
        x_upper_left = np.min(x_coord_np)-10 - random.randint(0, 10)
        y_upper_left = np.min(y_coord_np)-10 - random.randint(0, 10)
        x_lower_right = np.max(x_coord_np)+10 + random.randint(0, 10)
        y_lower_right = np.max(y_coord_np)+10 + random.randint(0, 10)
        if x_upper_left < 0:
            x_upper_left = 0
        if y_upper_left < 0:
            y_upper_left = 0
        if x_lower_right > mask.shape[2]:
            x_lower_right = mask.shape[2]
        if y_lower_right > mask.shape[1]:
            y_lower_right = mask.shape[1]
        prompt = np.array([x_upper_left, y_upper_left , x_lower_right , y_lower_right])
        return prompt

def compute_iou(preds, mask):
    intersection = np.logical_and(preds.cpu().numpy(), mask).sum()
    union = np.logical_or(preds.cpu().numpy(), mask).sum()
    iou = intersection / union if union > 0 else 0
    
    return iou

def overflow_score(preds, mask):
    #masks_path = clear_image_path_to_mask_path(clear_image_path)
    #mask = np.load(masks_path)
    target = torch.tensor(mask).unsqueeze(1).to('cuda')
    mask_tensor = torch.from_numpy(mask).int().to('cuda')
    # 膨張処理のためのカーネルを定義 (5x5の構造要素)
    structure = np.ones((5, 5), dtype=np.uint8)
    # 膨張処理を行う
    dilated_mask = binary_dilation(mask[0], structure=structure).astype(np.uint8)
    # 元の形状 (1, H, W) に戻す
    dilated_mask = dilated_mask[np.newaxis, ...]
    edge_mask= dilated_mask - mask
    edge_mask_tensor = torch.from_numpy(edge_mask).int().to('cuda')

    preds = preds.int().to('cuda')
    
    predict_edge_mask = preds - mask_tensor[0]

    predict_outside_mask = predict_edge_mask * edge_mask_tensor[0]
    
    non_zero_values = predict_outside_mask[predict_outside_mask != 0]
      
    edge_mask_tensor_non_zero_values = edge_mask_tensor[edge_mask_tensor != 0]
    
    protrud_sum=non_zero_values.sum()
    
    edge_sum=edge_mask_tensor_non_zero_values.sum()
    
    return float(protrud_sum/edge_sum)




if __name__ == "__main__":
    folders = [
        "/workdir/data/all_data/test/snow",
        "/workdir/data/all_data/test/fog",
        "/workdir/data/all_data/test/rain"
    ]
    print('------------------------------------------------------------')
    # 結果を格納するリスト
    selected_files = []

    # 2000回繰り返す
    for _ in range(2000):
        while True:
            # ランダムにフォルダーを選択
            selected_folder = random.choice(folders)
            
            # フォルダー内のファイル一覧を取得
            files = os.listdir(selected_folder)
            
            # `#MSRA` を含むファイルをフィルタリング
            msra_files = [file for file in files if "#MSRA" in file]
            
            # 該当するファイルが存在する場合、ランダムに選択
            if msra_files:
                selected_file = random.choice(msra_files)
                selected_files.append(os.path.join(selected_folder, selected_file))
                break  # 次のループに進む



    #プロンプトの点の数
    num_points = 2

    # ファイル名をアルファベット順にソート

    for k in tqdm(range(5)):
        #IoUとoverflow_scoreを保存するリスト
        overflow_score_and_IoU_list = []
    # 画像を順番に読み込み
        for image_file in selected_files:
            #prompt = get_point_prompt(image_file, num_points)
            prompt = get_box_prompt(image_file)
        
            propose_mask = propose_predict(image_file, prompt)
            # RobustSAM_mask = RobustSAM_predict(image_file, prompt)
            # SAM_mask = SAM_predict(image_file,  prompt)
        
            IoU_score_propose_best, IoU_score_RobustSAM_best, IoU_score_SAM_best = 0, 0, 0

            clear_image_path = degration_image_path_to_clear_image_path(image_file)

            masks_path = clear_image_path_to_mask_path(clear_image_path)
            mask = np.load(masks_path)
            target = torch.tensor(mask).unsqueeze(1).to('cuda')
            
            #iou_propose =  compute_iou(propose_mask, target)
            #iou_RobustSAM =  compute_iou(RobustSAM_mask, target)
            #iou_SAM =  compute_iou(SAM_mask, target)
            
            #overflow_score_propose, overflow_score_RobustSAM, overflow_score_SAM = 0, 0, 0
            
            for i in range(0, 15):
                iou_propose = compute_iou(propose_mask[0][i], mask) 
                # iou_RobustSAM = compute_iou(RobustSAM_mask[0][i], mask)
                if iou_propose > IoU_score_propose_best:
                    IoU_score_propose_best = iou_propose
                    index_propose = i
                # if  iou_RobustSAM > IoU_score_RobustSAM_best:
                #     IoU_score_RobustSAM_best = iou_RobustSAM
                #     index_RobustSAM = i
        
            # for j in range(0, 3):
            #     iou_SAM = compute_iou(SAM_mask[0][j], mask)
            #     if iou_SAM > IoU_score_SAM_best:
            #         IoU_score_SAM_best = iou_SAM
            #         index_SAM = j
            
            overflow_score_propose = overflow_score(propose_mask[0][index_propose], mask)
            # overflow_score_RobustSAM = overflow_score(RobustSAM_mask[0][index_RobustSAM], mask)
            # overflow_score_SAM = overflow_score(SAM_mask[0][index_SAM], mask)
            
            overflow_score_and_IoU_list.append([image_file,IoU_score_propose_best, IoU_score_RobustSAM_best, IoU_score_SAM_best, overflow_score_propose, overflow_score_RobustSAM, overflow_score_SAM])

        csv_filename = f"degrade_score{k}_box.csv"
        os.chdir('/workdir/alpha_analyze/degraded')
        # CSVファイルとして保存
        if os.path.exists(csv_filename):
            # ファイルが存在する場合、追記モードでデータを追加
            df = pd.DataFrame(overflow_score_and_IoU_list)
            df.to_csv(csv_filename, mode='a', header=False, index=False)
        else:
            # ファイルが存在しない場合、新しく作成してデータを保存
            df = pd.DataFrame(overflow_score_and_IoU_list, columns=['clear_image_path','IoU_score_propose', 'IoU_score_RobustSAM', 'IoU_score_SAM','overflow_score_propose', 'overflow_score_RobustSAM','overflow_score_SAM'])
            df.to_csv(csv_filename, index=False)