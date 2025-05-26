import numpy as np
import torch
from torch.utils.data import Dataset
import os
import random
from glob import glob
import cv2

class TrainDataset(Dataset):
    def __init__(self, opt, mode=None, training_size=512):
        super(TrainDataset).__init__()

        self.root_dir = os.path.join(opt.data_dir, mode)
        self.mode = mode #add

        self.clear_image_path = os.path.join(self.root_dir, 'clear')
        self.clear_image_list = sorted(glob(os.path.join(self.clear_image_path, '*jpg'))) 

        self.clear_image_files_MSRA = []

        for image_file in self.clear_image_list:
            # ファイル名に「#MSRA」が含まれているかを確認
            if '#MSRA' in image_file:
                self.clear_image_files_MSRA.append(image_file)
        
        self.clear_image_files_MSRA = sorted(self.clear_image_files_MSRA)
        """""
        print("------------------------------------------------------------------")
        print(self.clear_image_files_MSRA[:2])
        print(len(self.clear_image_files_MSRA))
        print(self.clear_image_list[:2])
        print(len(self.clear_image_list))
        """""
        #add
        np.random.seed(opt.seed)
        #if mode == 'train':
            #data_num = opt.train_num
        #elif mode == 'val':
            #data_num = opt.val_num
        #sample_indices = np.random.choice(len(self.clear_image_list), data_num, replace=False)
        #self.sample_indices = sample_indices
        #
               
        self.mask_path = os.path.join(self.root_dir, 'masks')
        self.mask_list = sorted(glob(os.path.join(self.mask_path, '*npy')))

        self.mask_list_MSRA = []

        for mask_file in self.mask_list:
            # ファイル名に「#MSRA」が含まれているかを確認
            if '#MSRA' in mask_file:
                self.mask_list_MSRA.append(mask_file)
        
        self.mask_list_MSRA = sorted(self.mask_list_MSRA)

        """""
        print("-----------------------------mask---------------------------")
        print(self.mask_list_MSRA[:2])
        print(len(self.mask_list_MSRA))
        print(self.mask_list[:2])
        print(len(self.mask_list))
        """
        #self.mask_list = np.array(self.mask_list)[sample_indices].tolist() #add
        #self.clear_image_list = np.array(self.clear_image_list)[sample_indices].tolist() #add
        #add
        if mode == 'val':
            degraded_image_path_val = os.path.join(self.root_dir, opt.val_degraded_type)
            degraded_image_list_val = sorted(glob(os.path.join(degraded_image_path_val, '*jpg'))) 

            self.degraded_image_files_MSRA_val = []

            for image_file in degraded_image_list_val:
                # ファイル名に「#MSRA」が含まれているかを確認
                if '#MSRA' in image_file:
                    self.degraded_image_files_MSRA_val.append(image_file)
            self.degraded_image_files_MSRA_val = sorted(self.degraded_image_files_MSRA_val)
            """
            print("-----------------------------val---------------------------")
            print(self.degraded_image_files_MSRA_val[:2])
            print(len(self.degraded_image_files_MSRA_val))
            print(degraded_image_list_val[:2])
            print(len(degraded_image_list_val))
            exit()
            #self.degraded_image_files_MSRA_val = sorted(self.degraded_image_files_MSRA_val)
            #self.degraded_image_list_val = np.array(degraded_image_list_val)[self.sample_indices].tolist()
            """

        #self.transform_list = ['clear', 'snow', 'fog', 'rain', 'gauss_noise', 'ISO_noise', 'impulse_noise', 'resampling_blur', 
                               #'motion_blur', 'zoom_blur', 'color_jitter', 'compression', 'elastic_transform',
                               #'frosted_glass_blur', 'brightness', 'contrast' ]
        #self.transform_list = ['clear', 'snow', 'fog', 'rain', 'gauss_noise', 'ISO_noise', 'brightness']
        self.transform_list = ['clear', 'snow', 'fog', 'rain']
        self.training_size = (training_size, training_size)
        self.num_points = opt.num_points
        self.num_points_test = opt.num_points_test #add
        
        # making sure there is same number of clear and degraded images
        self.snow_image_path = os.path.join(self.root_dir, 'snow')
        self.snow_image_list = sorted(glob(os.path.join(self.snow_image_path, '*jpg'))) 
        #self.snow_image_list = np.array(self.snow_image_list)[sample_indices].tolist() #add
        print('Number of training images: {}'.format(min(len(self.clear_image_list), len(self.snow_image_list))))
                
    def get_im(self, im_path):
        image = cv2.imread(im_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  
        image = cv2.resize(image, self.training_size)
        image = torch.tensor(image, dtype=torch.uint8)
        
        return image

    def get_largest_mask(self, all_mask, output_size=None):        
        max_area = 0  
        mask_index = [0]   
        
        for i, mask in enumerate(all_mask):
            index = np.where(mask == True)
            y_coord_np = index[0]
            x_coord_np = index[1]
            
            if len(y_coord_np) > max_area:
                max_area = len(y_coord_np)
                mask_index = [i]    
    
        output_mask = all_mask[mask_index]
        output_mask = np.transpose(output_mask, (1, 2, 0))       

        if output_size is not None:
            output_mask = cv2.resize(output_mask.astype(np.uint8), output_size)

        return output_mask

    def get_prompt(self, mask, num_points):                
        input_point, input_label = [], []
        index = np.where(mask == True)
        y_coord_np = index[0]
        x_coord_np = index[1]
        index_list = range(0, len(x_coord_np))        
        
        if len(x_coord_np) < num_points:
            for i in range(len(x_coord_np)):
                coord = [x_coord_np[i], y_coord_np[i]]
                input_point.append(coord)
                input_label.append(1)
                
            while(len(input_point) < num_points):
                if len(x_coord_np) != 0:
                    input_point.append(coord)
                
                else:
                    input_point.append([256, 256])
                               
                input_label.append(1)
            
        else:
            index = random.sample(index_list, num_points)                           
            for i in index:
                coord = [x_coord_np[i], y_coord_np[i]]
                input_point.append(coord)
                input_label.append(1)   
        
        input_point = np.array(input_point, dtype=np.float32)
        input_label = np.array(input_label, dtype=np.float32)   
        
        return input_point, input_label
    
    def __len__(self):
        #return min(len(self.clear_image_files_MSRA), len(self.snow_image_list))
        return len(self.clear_image_files_MSRA)
    
    def __getitem__(self, idx):     
        #clear_path = self.clear_image_list[idx] 
        clear_path = self.clear_image_files_MSRA[idx]
        clear_im = self.get_im(clear_path)                 
        
        #all_mask = np.load(self.mask_list[idx])
        all_mask = np.load(self.mask_list_MSRA[idx]) # npy file with all masks inside specific image      
        mask = self.get_largest_mask(all_mask, output_size=self.training_size) 
        
        #add
        if self.mode == 'train':
            input_point, input_label = self.get_prompt(mask, self.num_points)   
            degraded_type = random.choice(self.transform_list)
            self.degraded_image_path = os.path.join(self.root_dir, degraded_type)
            self.degraded_image_list = sorted(glob(os.path.join(self.degraded_image_path, '*jpg'))) 

            self.degraded_image_files_MSRA = []

            for image_file in self.degraded_image_list:
                # ファイル名に「#MSRA」が含まれているかを確認
                if '#MSRA' in image_file:
                    self.degraded_image_files_MSRA.append(image_file)
            
            self.degraded_image_files_MSRA = sorted(self.degraded_image_files_MSRA)
            #self.degraded_image_list = np.array(self.degraded_image_list)[self.sample_indices].tolist() #add
        elif self.mode == 'val':
            input_point, input_label = self.get_prompt(mask, self.num_points_test)
            self.degraded_image_list = self.degraded_image_files_MSRA_val 
        
        degraded_path = self.degraded_image_list[idx]  
        degraded_im = self.get_im(degraded_path)   
        #

        return clear_im, degraded_im, clear_path, mask, input_point, input_label



    
