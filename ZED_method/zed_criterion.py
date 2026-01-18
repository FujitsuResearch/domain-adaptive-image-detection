"""
the main() function in this file is based on src.eval.py from the official SReC github page - 
https://github.com/caoscott/SReC/tree/master
"""

import torch
import numpy as np
import os
import imageio
import argparse
from torch.utils import data
from tqdm import tqdm

import os
import sys

# Get the absolute path of the current script
script_path = os.path.abspath(__file__)

# Get the directory of the current script
script_dir = os.path.dirname(script_path)

# Add the script directory to the sys.path if not already included
if script_dir not in sys.path:
    sys.path.append(script_dir)
from zed_utils import *


def to_tensor_not_normalized(pic: Image) -> torch.Tensor:
    
    if isinstance(pic, np.ndarray):
        return torch.from_numpy(pic.transpose((2, 0, 1)))

    mode_to_dtype = {
        'I': np.int32,
        'I;16': np.int16,
        'F': np.float32,
        '1': np.uint8,
        'RGB': np.uint8
    }
    
    if pic.mode in mode_to_dtype:
        img = torch.from_numpy(np.array(pic, mode_to_dtype[pic.mode], copy=True))
        if pic.mode == '1':
            img *= 255

    nchannel = 3 if pic.mode in ['YCbCr', 'RGB'] else (1 if pic.mode == 'I;16' else len(pic.mode))
    img = img.view(pic.size[1], pic.size[0], nchannel).transpose(0, 1).transpose(0, 2).contiguous()
    return img.float()


def load_video_for_zed(video_path, device, max_frames, downsample_frames):
    
    reader = imageio.get_reader(video_path, format='ffmpeg')
    
    frame_tensors_list = []
    for idx, frame in enumerate(reader):
        # Downsample: Skip frames based on downsample_frames
        if idx % downsample_frames != 0:
            continue
        # convert to tensor
        frame_t = to_tensor_not_normalized(frame).unsqueeze(0).to(device)
        # Append to batch
        frame_tensors_list.append(frame_t)
        # Stop if max_frames limit is reached
        if max_frames is not None and len(frame_tensors_list) >= max_frames:
            break

    reader.close()

    if len(frame_tensors_list) == 0:
        raise ValueError(f"No frames found in {video_path}")

    return frame_tensors_list


def load_images_for_zed(image_paths, device):

    image_list = []
    for image_path in image_paths:
        img = Image.open(image_path)
        image_list.append(to_tensor_not_normalized(img).unsqueeze(0))

    return image_list


def factory_zed_criterion(device, compressor):
    def zed_criterion(images_raw):

        compressor.eval()
        ret_dicts = []
        bad_ims = []
        print(len(bad_ims))
        with torch.no_grad():
            
            for i, x in enumerate(tqdm(images_raw, desc="Processing images")):
                
                if i in bad_ims:
                    continue
                
                x = x.permute(2, 0, 1).unsqueeze(0).to("cuda")
                bits = compressor(x)
                x.detach().cpu()
                
                level_nll_sums = {}
                level_entropy_sums = {}
                level_counts = {}
                for key in bits.get_keys():
                    
                    level = int(key.split('_')[0].split('/')[1])
                    nll_full = bits.get_nll(key)
                    entropy_full = bits.get_entropy(key)
                    entropy = torch.mean(entropy_full) 
                    nll = torch.mean(nll_full)
                    
                    if level not in level_nll_sums:
                        level_nll_sums[level] = 0.0
                        level_entropy_sums[level] = 0.0
                        level_counts[level] = 0

                    level_nll_sums[level] += nll.item()
                    level_entropy_sums[level] += entropy.item()
                    level_counts[level] += 1

                level_nll_avgs = {}
                level_entropy_avgs = {}
                for level in level_nll_sums:
                    level_nll_avgs[level] = level_nll_sums[level] / level_counts[level]
                    level_entropy_avgs[level] = level_entropy_sums[level] / level_counts[level]
                
                D_0 = level_nll_avgs[2] - level_entropy_avgs[2]
                D_0_abs = np.abs(D_0)
                D_1 = level_nll_avgs[1] - level_entropy_avgs[1]
                Delta_0_1 = D_0 - D_1
                Delta_0_1_abs = np.abs(Delta_0_1)
                
                cur_dict = {
                        "criterion": float(Delta_0_1),
                        "criterion D_0": float(D_0),
                        "criterion |D_0|": float(D_0_abs),                  
                        "criterion Delta_0_1": float(Delta_0_1),           
                        "criterion |Delta_0_1|": float(Delta_0_1_abs),
                    }  
                ret_dicts.append(cur_dict)

        return ret_dicts
    return zed_criterion


def main(images_dir, output_path)-> None:
    
    if not os.path.isdir(images_dir):
        raise ValueError(f"images_dir does not exist or is not a directory: {images_dir}")

    model_path = "openimages.pth"
    image_paths = [os.path.join(images_dir, file) for file in os.listdir(images_dir) if os.path.isfile(os.path.join(images_dir, file))]
    
    checkpoint = torch.load(model_path, weights_only=True)
    print(f"Loaded model from {model_path}.")

    compressor = Compressor()
    compressor.nets.load_state_dict(checkpoint["nets"])
    compressor = compressor.cuda()

    dataset = ImageFolder(image_paths)
    
    loader = data.DataLoader(
        dataset, batch_size=1, shuffle=False,
        num_workers=2, drop_last=False,
    )
    print(f"Loaded dataset with {len(dataset)} images")

    compressor.eval()
    D_0_criterion = []
    D_0_abs_criterion = []
    Delta_0_1_criterion = []
    Delta_0_1_abs_criterion = []
    with torch.no_grad():
        
        for x in tqdm(loader):
            print(x.size())
            inp_size = np.prod(x.size()[1:])
            x = x.cuda()
            bits = compressor(x)
            
            level_nll_sums = {}
            level_entropy_sums = {}
            level_counts = {}
            for key in bits.get_keys():
                
                level = int(key.split('_')[0].split('/')[1])
                nll = bits.get_scaled_bpsp(key, inp_size)
                p = torch.exp(-nll) 
                entropy = -torch.sum(p * torch.log(p))  
                
                if level not in level_nll_sums:
                    level_nll_sums[level] = 0.0
                    level_entropy_sums[level] = 0.0
                    level_counts[level] = 0

                level_nll_sums[level] += nll.item()
                level_entropy_sums[level] += entropy.item()
                level_counts[level] += 1

            level_nll_avgs = {}
            level_entropy_avgs = {}
            for level in level_nll_sums:
                level_nll_avgs[level] = level_nll_sums[level] / level_counts[level]
                level_entropy_avgs[level] = level_entropy_sums[level] / level_counts[level]
            
            D_0 = level_nll_avgs[0] - level_entropy_avgs[0]
            D_0_abs = np.abs(D_0)
            D_1 = level_nll_avgs[1] - level_entropy_avgs[1]
            Delta_0_1 = D_0 - D_1
            Delta_0_1_abs = np.abs(Delta_0_1)
            
            D_0_criterion.append(D_0)
            D_0_abs_criterion.append(D_0_abs)
            Delta_0_1_criterion.append(Delta_0_1)
            Delta_0_1_abs_criterion.append(Delta_0_1_abs)
            print(f'D_0: {D_0}, |D_0|: {D_0_abs}, Delta_0_1: {Delta_0_1}, |Delta_0_1|: {Delta_0_1_abs}')

    if output_path:
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        torch.save((torch.tensor(D_0_criterion), torch.tensor(D_0_abs_criterion), torch.tensor(Delta_0_1_criterion), torch.tensor(Delta_0_1_abs_criterion)), output_path)  

if __name__ == "__main__":
    
    parser = argparse.ArgumentParser(
        description="Run ZED criterion on a directory of images"
    )
    parser.add_argument(
        "--images_dir",
        type=str,
        default="data/images",
        help="Path to directory containing input images"
    )
    parser.add_argument(
        "--output_path",
        type=str,
        default="zed_criterions.pt",
        help="Path to output file containing criterions."
    )
    args = parser.parse_args()

    main(images_dir=args.images_dir, output_path=args.output_path)
