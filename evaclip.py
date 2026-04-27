import cv2
import numpy as np
from PIL import Image, ImageOps
import torch
import argparse
import os
import json
import torch
import onnx
import onnxruntime as ort
import csv
import glob
import re
import torchvision.transforms as T
from eva_clip.factory import create_model_and_transforms
from eva_clip.constants import OPENAI_DATASET_MEAN, OPENAI_DATASET_STD

def image_to_tensor(image):
    tensor = torch.clamp(torch.from_numpy(image).float() / 255., 0, 1)
    return tensor

class EVACLIP:
    def __init__(self):
        self.device = 'cuda'
        self.dtype = torch.float32
        model, _, _ = create_model_and_transforms('EVA02-CLIP-L-14-336', 'eva_clip', cache_dir='checkpoints/evaclip',force_custom_clip=True)
        model = model.visual

        eva_transform_mean = getattr(model, 'image_mean', OPENAI_DATASET_MEAN)
        eva_transform_std = getattr(model, 'image_std', OPENAI_DATASET_STD)
        if not isinstance(eva_transform_mean, (list, tuple)):
            model["image_mean"] = (eva_transform_mean,) * 3
        if not isinstance(eva_transform_std, (list, tuple)):
            model["image_std"] = (eva_transform_std,) * 3
        self.model = model.to(self.device, dtype=self.dtype)

    def predict(self, image):
        image = np.array(image)
        face_features_image = image_to_tensor(image).unsqueeze(0).permute(0,3,1,2).to(self.device)
        face_features_image = T.functional.resize(face_features_image, self.model.image_size, T.InterpolationMode.BICUBIC).to(dtype=self.dtype)
        face_features_image = T.functional.center_crop(face_features_image, self.model.image_size)
        face_features_image = T.functional.normalize(face_features_image, self.model.image_mean, self.model.image_std)
        with torch.no_grad():
            id_cond_vit, id_vit_hidden = self.model(face_features_image, return_all_features=False, return_hidden=True, shuffle=False)
        id_vit_hidden = torch.cat(id_vit_hidden, dim=0)
        return {'clip_emb': id_cond_vit.cpu(), 'clip_hidden': id_vit_hidden.cpu()}

def load_image_face(image_path, bbox):
    image = Image.open(image_path).convert('RGB')
    image = ImageOps.exif_transpose(image)
    image_face = image.crop(bbox)
    return image_face

def get_part_lines(file_list, part_index, part_total):
    total = len(file_list)
    part_len = (total - 1) // part_total + 1
    part_file_list = file_list[part_index * part_len:(part_index + 1) * part_len]
    print(total, len(part_file_list))
    return part_file_list

def glob_all_file_part(file_path):
    file_path = re.sub(r'-\d+:\d+', '*', file_path)
    if '*' not in file_path:
        file_path = file_path.replace('.txt', '*.txt')
    file_list = glob.glob(file_path)
    return file_list

def load_exist_file(file_path):
    accept_dict = {}
    accept_files = glob_all_file_part(file_path)
    if len(accept_files) == 0: return accept_dict
    afile_list = []
    for accept_file in accept_files:
        with open(accept_file, 'r') as af:
            alist = af.readlines()
        afile_list.append(alist)
    for alist in afile_list:
        for file in alist:
            file = file.strip()
            file_json = json.loads(file)
            name = file_json['file_name']
            accept_dict[name] = file_json
    return accept_dict

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--image_path_files",
        type=str,
        default=None,
        help="图像文件索引路径",
    )
    parser.add_argument(
        "--part_index",
        type=int,
        default=None,
        help="图像文件索引编号",
    )
    parser.add_argument(
        "--part_total",
        type=int,
        default=None,
        help="图像文件索引总编号",
    )
    parser.add_argument(
        "--save_root",
        type=str,
        default='./portrait_train_evaclip',
        help="如果需要保存的路径",
    )
    args = parser.parse_args()
    evaclip = EVACLIP()
    file_list = args.image_path_files.split(',')
    for file in file_list:
        with open(file, 'r') as f:
            image_json_list = f.readlines()
        
        file_name = file.split('/')[-1].split('.')[0]
        if args.part_index is not None and args.part_total is not None:
            image_json_list = get_part_lines(image_json_list, args.part_index, args.part_total)
            file_name += f'-{args.part_index}:{args.part_total}'
        save_path = os.path.join(args.save_root, file_name)
        os.makedirs(save_path, exist_ok=True)
        save_file = os.path.join(args.save_root, file_name + '.txt')
        exist_dict = load_exist_file(save_file)
        with open(save_file, 'w') as f:
            for image_json in image_json_list:
                image_json = json.loads(image_json)
                image_path = image_json['file_name']
                if image_path in exist_dict:
                    print('exist', image_path)
                    content = exist_dict[image_path]
                    origin_save_path = content['evaclip_feature_file']
                    save_name = origin_save_path.split('/')[-1]
                    current_save_path = os.path.join(save_path, save_name)
                    if os.path.exists(origin_save_path):
                        os.rename(origin_save_path, current_save_path)
                    elif not os.path.exists(current_save_path):
                        print('feature file lost, regenerate', image_path)
                        face_image = load_image_face(image_path, image_json['bbox'])
                        clip_emb = evaclip.predict(face_image)
                        torch.save(clip_emb, current_save_path)
                    content['evaclip_feature_file'] = current_save_path
                    f.write(json.dumps(content) + '\n')
                    f.flush()
                    continue
                face_image = load_image_face(image_path, image_json['bbox'])
                clip_emb = evaclip.predict(face_image)
                feature_save_path = os.path.join(save_path, os.path.basename(image_json['insightface_feature_file']))
                torch.save(clip_emb, feature_save_path)
                image_json['evaclip_feature_file'] = feature_save_path
                f.write(json.dumps(image_json) + '\n')
                f.flush()
                print('success', image_path)
        print("finish", file_name)

def check_evaclip_feature():
    evaclip = EVACLIP()

    path = './portrait_train_evaclip/accept_*.txt'
    file_list = glob.glob(path)
    file_list = sorted(file_list)
    print(len(file_list))
    lost_num = 0
    n = 0
    for file in file_list:
        print(file)
        file_name = file.split('/')[-1]
        with open(file, 'r') as f:
            lines = f.readlines()
        with open('./lost.txt', 'w') as f:
            for i, line in enumerate(lines):
                line = line.strip()
                line = json.loads(line)
                feature = line['evaclip_feature_file']
                if not os.path.exists(feature):
                    lost_num += 1
                    image_path = line['file_name']
                    face_image = load_image_face(image_path, line['bbox'])
                    clip_emb = evaclip.predict(face_image)
                    torch.save(clip_emb, feature)
                    f.write('ok\t' + json.dumps(line) + '\n')
                    f.flush()
                    print('lost', lost_num, feature, 'regenerate')
                n += 1
                if n % 10000 == 0:
                    print('check', n, 'lost', lost_num)

if __name__ == '__main__':
    main()
    check_evaclip_feature()
    