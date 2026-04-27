import cv2
import numpy as np
from PIL import Image, ImageOps
import torch
import os
import json
import argparse
from insightface.app import FaceAnalysis
import hashlib
import glob
import re

class InsightFace:
    def __init__(self):
        root = "./checkpoints/"
        self.app = FaceAnalysis(name='antelopev2', root=root, providers=['CUDAExecutionProvider', 'CPUExecutionProvider'])
        self.app.det_model.input_size = (640, 640)

    def detect(self, image):
        image = ImageOps.exif_transpose(image)
        image = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
        for size in [(size, size) for size in range(640, 256, -64)]:
            self.app.det_model.input_size = size
            face_info = self.app.get(image)
            if face_info:
                face_info = sorted(face_info, key=lambda x: (x.bbox[2] - x.bbox[0]) * (x.bbox[3] - x.bbox[1]))[-1]
                return face_info
            else:
                print(f'InsightFace detection resolution lowered to {size}')

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
        "--save_root",
        type=str,
        default='/apdcephfs_cq8/share_1367250/francofhzhu/data/portrait_cond',
        help="如果需要保存的路径",
    )
    args = parser.parse_args()
    insightface_model = InsightFace()
    file_list = args.image_path_files.split(',')
    for file in file_list:
        with open(file, 'r') as f:
            image_path_list = f.readlines()
        file_name = file.split('/')[-1]
        save_path = os.path.join(args.save_root, file_name.split('.')[0])
        os.makedirs(save_path, exist_ok=True)
        save_file = os.path.join(args.save_root, file_name)
        exist_dict = load_exist_file(save_file)
        with open(save_file, 'w') as f:
            for image_path in image_path_list:
                image_path = image_path.strip()
                if image_path in exist_dict:
                    print('exist', image_path)
                    content = exist_dict[image_path]
                    origin_save_path = content['insightface_feature_file']
                    save_name = origin_save_path.split('/')[-1]
                    current_save_path = os.path.join(save_path, save_name)
                    if os.path.exists(origin_save_path):
                        os.rename(origin_save_path, current_save_path)
                    elif not os.path.exists(current_save_path):
                        print('feature file lost, regenerate', image_path)
                        image = Image.open(image_path).convert('RGB')
                        face_info = insightface_model.detect(image)
                        if face_info is None:
                            print('face detect fail', image_path)
                            continue
                        torch.save(face_info['embedding'], current_save_path)
                    content['insightface_feature_file'] = current_save_path
                    f.write(json.dumps(content) + '\n')
                    f.flush()
                    continue
                image = Image.open(image_path).convert('RGB')
                face_info = insightface_model.detect(image)
                if face_info is None:
                    print('face detect fail', image_path)
                    continue
                image_md5 = hashlib.md5(image.tobytes()).hexdigest()
                feature_save_path = os.path.join(save_path, f'{image_md5}.bin')
                torch.save(face_info['embedding'], feature_save_path)

                face_dict = {'file_name': image_path, 'insightface_feature_file': feature_save_path}
                face_dict['bbox'] = face_info['bbox'].tolist()
                face_dict['landmarks'] = face_info['kps'].tolist()

                f.write(json.dumps(face_dict) + '\n')
                f.flush()
                print('success', image_path)

        print("finish", file)

def check_insightface_feature():
    insightface_model = InsightFace()

    path = '/apdcephfs_cq8/share_1367250/francofhzhu/data/portrait_train/accept_*.txt'
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
                feature = line['insightface_feature_file']
                if not os.path.exists(feature):
                    lost_num += 1
                    image_path = line['file_name']
                    image = Image.open(image_path).convert('RGB')
                    face_info = insightface_model.detect(image)
                    if face_info is None:
                        f.write('lost\t' + json.dumps(line) + '\n')
                        f.flush()
                        print('lost', lost_num, feature, 'face detect fail')
                    else:
                        torch.save(face_info['embedding'], feature)
                        f.write('ok\t' + json.dumps(line) + '\n')
                        f.flush()
                        print('lost', lost_num, feature, 'regenerate')
                n += 1
                if n % 10000 == 0:
                    print('check', n, 'lost', lost_num)

if __name__ == '__main__':
    # main()
    check_insightface_feature()
    