import os
import re
import random
import argparse
from pathlib import Path
import json
import itertools
import time
from datetime import datetime
import shutil
import torch
import torch.nn.functional as F
import numpy as np
import math
import cv2
from torchvision import transforms
from PIL import Image, ImageOps
import PIL
from transformers import CLIPImageProcessor
from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.utils import ProjectConfiguration
from diffusers import AutoencoderKL, DDPMScheduler, DPMSolverMultistepScheduler, DPMSolverSinglestepScheduler, UNet2DConditionModel, ControlNetModel
from transformers import CLIPTextModel, CLIPTokenizer, CLIPVisionModelWithProjection, CLIPTextModelWithProjection
from safetensors.torch import load_file
from skimage import transform as trans
from kornia.geometry import warp_affine

from ip_adapter.resampler import Resampler
from ip_adapter.utils import is_torch2_available
from ip_adapter.arcface import get_arcface
import deepspeed_utils
from datetime import timedelta
from accelerate import InitProcessGroupKwargs

if is_torch2_available():
    from ip_adapter.attention_processor import IPAttnProcessor2_0 as IPAttnProcessor, AttnProcessor2_0 as AttnProcessor
else:
    from ip_adapter.attention_processor import IPAttnProcessor, AttnProcessor


def load_file_parallel(path, n_parallel=48):
    from multiprocessing import Process, Manager
    def get_part_lines(total, part_index, part_total):
        part_len = (total - 1) // part_total + 1
        start, end = part_index * part_len, (part_index + 1) * part_len
        end = min(total, end)
        return start, end

    def parse(mlist, lines):
        part_lines = []
        for line in lines:
            part_lines.append(json.loads(line))
        mlist.extend(part_lines)

    with open(path, 'r') as f:
        lines = f.readlines()
    total = len(lines)
    with Manager() as manager:
        mlist = manager.list()
        processes = []
        for part_index in range(n_parallel):
            start, end = get_part_lines(total, part_index, n_parallel)
            p = Process(target=parse, args=[mlist, lines[start:end]]) 
            p.start()
            processes.append(p)
        for p in processes:
            p.join()
        mlist = list(mlist)
    return mlist

# Draw the input image for controlnet based on facial keypoints.
def draw_kps(w, h, kps, color_list=[(255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0), (255, 0, 255)]):
    stickwidth = 4
    limbSeq = np.array([[0, 2], [1, 2], [3, 2], [4, 2]])
    kps = np.array(kps)
    out_img = np.zeros([h, w, 3])

    for i in range(len(limbSeq)):
        index = limbSeq[i]
        color = color_list[index[0]]

        x = kps[index][:, 0]
        y = kps[index][:, 1]
        length = ((x[0] - x[1]) ** 2 + (y[0] - y[1]) ** 2) ** 0.5
        angle = math.degrees(math.atan2(y[0] - y[1], x[0] - x[1]))
        polygon = cv2.ellipse2Poly((int(np.mean(x)), int(np.mean(y))), (int(length / 2), stickwidth), int(angle), 0,
                                   360, 1)
        out_img = cv2.fillConvexPoly(out_img.copy(), polygon, color)
    out_img = (out_img * 0.6).astype(np.uint8)

    for idx_kp, kp in enumerate(kps):
        color = color_list[idx_kp]
        x, y = kp
        out_img = cv2.circle(out_img.copy(), (int(x), int(y)), 10, color, -1)

    out_img_pil = PIL.Image.fromarray(out_img.astype(np.uint8))
    return out_img_pil

# Process the dataset by loading info from a JSON file, which includes image files, image labels, feature files, keypoint coordinates.
class MyDataset(torch.utils.data.Dataset):

    def __init__(self, json_file, tokenizer, tokenizer_2, size=1024, center_crop=True,
                 t_drop_rate=0.025, i_drop_rate=0.025, c_drop_rate=0.025, 
                 ti_drop_rate=0.025, tc_drop_rate=0.025, ic_drop_rate=0.025,
                 tic_drop_rate=0.025, image_root_path="", do_cfg=True):
        super().__init__()

        self.tokenizer = tokenizer
        self.tokenizer_2 = tokenizer_2
        self.size = size
        self.center_crop = center_crop
        self.i_drop_rate = i_drop_rate
        self.t_drop_rate = t_drop_rate
        self.c_drop_rate = c_drop_rate
        self.ti_drop_rate = ti_drop_rate
        self.tc_drop_rate = tc_drop_rate
        self.ic_drop_rate = ic_drop_rate
        self.tic_drop_rate = tic_drop_rate
        self.image_root_path = image_root_path
        self.do_cfg = do_cfg

        self.data = load_file_parallel(json_file)
        # with open(json_file, 'r') as f:
        #     for line in f:
        #         self.data.append(json.loads(line))

        self.image_transforms = transforms.Compose(
            [
                transforms.Resize(self.size, interpolation=transforms.InterpolationMode.BILINEAR),
                transforms.ToTensor(),
                transforms.Normalize([0.5], [0.5]),
            ]
        )

        self.conditioning_image_transforms = transforms.Compose(
            [
                transforms.Resize(self.size, interpolation=transforms.InterpolationMode.BILINEAR),
                transforms.ToTensor(),
            ]
        )

        self.clip_image_processor = CLIPImageProcessor()
        self.sim_trans = trans.SimilarityTransform()
        self.arcface_dst = np.array(
            [[38.2946, 51.6963], [73.5318, 51.5014], [56.0252, 71.7366], [41.5493, 92.3655], [70.7299, 92.2041]],
        dtype=np.float32)  # for image_size = 112
        self.target_kps = np.array([
            [[308.78045654296875, 456.2644958496094], [687.5463256835938, 457.37286376953125], [496.63861083984375, 711.7979125976562], [375.9710388183594, 856.9695434570312], [636.3991088867188, 859.5838623046875]], 
            [[245.42750549316406, 590.9263916015625], [609.3950805664062, 396.0459899902344], [510.4403381347656, 723.5193481445312], [533.3201293945312, 901.0200805664062], [790.1721801757812, 738.7066040039062]], 
            [[339.3576965332031, 342.3412780761719], [705.9505004882812, 373.58367919921875], [512.0148315429688, 471.19329833984375], [348.88421630859375, 726.4502563476562], [634.3196411132812, 752.3500366210938]], 
            [[350.4601135253906, 471.5392150878906], [726.3257446289062, 497.94427490234375], [533.7391967773438, 735.3667602539062], [373.4756164550781, 834.7154541015625], [637.2051391601562, 857.34375]], 
            [[391.7713623046875, 406.4329528808594], [780.064453125, 532.4291381835938], [551.5917358398438, 750.3025512695312], [278.9065856933594, 776.3635864257812], [569.379150390625, 887.6179809570312]], 
            [[118.7446517944336, 33.22216796875], [135.39498901367188, 30.794979095458984], [125.89217376708984, 43.697566986083984], [123.57523345947266, 55.4010009765625], [136.13916015625, 53.288978576660156]], 
            [[327.3335266113281, 370.80401611328125], [680.6755981445312, 367.013671875], [503.62322998046875, 484.5788879394531], [374.3053283691406, 742.8248291015625], [649.6185302734375, 738.2794189453125]], 
            [[319.66815185546875, 458.8902893066406], [700.0777587890625, 461.4566955566406], [508.23944091796875, 711.6011352539062], [380.20208740234375, 852.5963745117188], [639.3641357421875, 854.7448120117188]], 
            [[261.2552795410156, 476.7975158691406], [608.4534301757812, 449.2742004394531], [413.0910339355469, 704.1737670898438], [366.84027099609375, 845.9500122070312], [603.4259643554688, 822.4013061523438]], 
            [[462.6876525878906, 464.9235534667969], [745.2998046875, 427.7240295410156], [716.51416015625, 625.8187866210938], [569.18994140625, 841.5875854492188], [763.3973388671875, 810.2045288085938]], 
            [[353.1114196777344, 451.3506164550781], [720.353515625, 460.84906005859375], [546.4733276367188, 704.2888793945312], [392.6568908691406, 820.5996704101562], [653.7382202148438, 827.1228637695312]], 
            [[369.35809326171875, 468.475830078125], [749.390625, 481.80194091796875], [581.0047607421875, 736.4447631835938], [395.7215881347656, 851.8864135742188], [665.763671875, 862.9447021484375]], 
            [[437.7513122558594, 341.6368713378906], [752.8917846679688, 467.8391418457031], [580.3821411132812, 566.1687622070312], [348.9738464355469, 714.5936889648438], [584.5840454101562, 816.6094970703125]]
        ])  # for 1024

    def __getitem__(self, idx):
        item = self.data[idx]
        image_file = item["file_name"]
        text = item["additional_feature"]
        bbox = item['bbox']
        landmarks = item['landmarks']
        target_landmarks = self.target_kps[np.random.randint(len(self.target_kps))]
        feature_file = item["insightface_feature_file"]
        evaclip_feature_file = item["evaclip_feature_file"]

        # read image
        raw_image = Image.open(os.path.join(self.image_root_path, image_file))
        raw_image = ImageOps.exif_transpose(raw_image)
        raw_image = raw_image.convert("RGB")
        # draw keypoints
        kps_image = draw_kps(raw_image.width, raw_image.height, landmarks)
        target_kps_image = draw_kps(1024, 1024, target_landmarks)

        # original size
        original_width, original_height = raw_image.size
        original_size = torch.tensor([original_height, original_width])

        # transform raw_image and kps_image
        image_tensor = self.image_transforms(raw_image)
        kps_image_tensor = self.conditioning_image_transforms(kps_image)
        target_kps_image_tensor = self.conditioning_image_transforms(target_kps_image)

        # random crop
        delta_h = image_tensor.shape[1] - self.size
        delta_w = image_tensor.shape[2] - self.size
        assert not all([delta_h, delta_w])

        if self.center_crop:
            top = delta_h // 2
            left = delta_w // 2
        else:
            top = np.random.randint(0, delta_h // 2 + 1)  # random top crop
            # top = np.random.randint(0, delta_h + 1)  # random crop
            left = np.random.randint(0, delta_w + 1)  # random crop

        # The image and kps_image must follow the same cropping to ensure that the facial coordinates correspond correctly.
        image = transforms.functional.crop(
            image_tensor, top=top, left=left, height=self.size, width=self.size
        )
        kps_image = transforms.functional.crop(
            kps_image_tensor, top=top, left=left, height=self.size, width=self.size
        )
        target_kps_image = transforms.functional.crop(
            target_kps_image_tensor, top=top, left=left, height=self.size, width=self.size
        )

        crop_coords_top_left = torch.tensor([top, left])

        # load face feature
        face_id_embed = torch.load(os.path.join(self.image_root_path, feature_file), map_location="cpu")
        face_id_embed = torch.from_numpy(face_id_embed)
        face_id_embed = face_id_embed.reshape(1, -1)

        # load evaclip feature
        evaclip_feature = torch.load(os.path.join(self.image_root_path, evaclip_feature_file), map_location="cpu")
        evaclip_embed, evaclip_hidden = evaclip_feature['clip_emb'], evaclip_feature['clip_hidden']
        # evaclip_embed = torch.from_numpy(evaclip_embed)
        evaclip_embed = evaclip_embed.reshape(1, -1)
        # evaclip_hidden = torch.from_numpy(evaclip_hidden)
        evaclip_hidden = evaclip_hidden.reshape(4, -1)

        # set cfg drop rate
        drop_feature_embed = 0
        drop_text_embed = 0
        drop_clip_embed = 0
        rand_num = random.random()
        if rand_num < self.i_drop_rate:
            drop_feature_embed = 1
        elif rand_num < (self.i_drop_rate + self.t_drop_rate):
            drop_text_embed = 1
        elif rand_num < (self.i_drop_rate + self.t_drop_rate + self.c_drop_rate):
            drop_clip_embed = 1
        elif rand_num < (self.i_drop_rate + self.t_drop_rate + self.c_drop_rate + self.ti_drop_rate):
            drop_text_embed = 1
            drop_feature_embed = 1
        elif rand_num < (self.i_drop_rate + self.t_drop_rate + self.c_drop_rate + self.ti_drop_rate + self.tc_drop_rate):
            drop_text_embed = 1
            drop_clip_embed = 1
        elif rand_num < (self.i_drop_rate + self.t_drop_rate + self.c_drop_rate + self.ti_drop_rate + self.tc_drop_rate + self.ic_drop_rate):
            drop_feature_embed = 1
            drop_clip_embed = 1
        elif rand_num < (self.i_drop_rate + self.t_drop_rate + self.ti_drop_rate + self.ti_drop_rate + self.tc_drop_rate + self.ic_drop_rate + self.tic_drop_rate):
            drop_text_embed = 1
            drop_feature_embed = 1
            drop_clip_embed = 1

        # CFG process
        if self.do_cfg:
            if drop_text_embed:
                text = ""
            if drop_feature_embed:
                face_id_embed = torch.zeros_like(face_id_embed)
            if drop_clip_embed:
                evaclip_embed = torch.zeros_like(evaclip_embed)
                evaclip_hidden = torch.zeros_like(evaclip_hidden)

        # get text and tokenize
        text_input_ids = self.tokenizer(
            text,
            max_length=self.tokenizer.model_max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt"
        ).input_ids

        text_input_ids_2 = self.tokenizer_2(
            text,
            max_length=self.tokenizer_2.model_max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt"
        ).input_ids

        self.sim_trans.estimate(np.array(landmarks), self.arcface_dst)
        align_matrix = self.sim_trans.params[:2]
        align_matrix = torch.tensor(align_matrix, dtype=torch.float32)
        self.sim_trans.estimate(np.array(target_landmarks), self.arcface_dst)
        target_align_matrix = self.sim_trans.params[:2]
        target_align_matrix = torch.tensor(target_align_matrix, dtype=torch.float32)

        return {
            "image": image,
            "kps_image": kps_image,
            "target_kps_image": target_kps_image,
            "align_matrix": align_matrix,
            "target_align_matrix": target_align_matrix,
            "text_input_ids": text_input_ids,
            "text_input_ids_2": text_input_ids_2,
            "face_id_embed": face_id_embed,
            "evaclip_embed": evaclip_embed,
            "evaclip_hidden": evaclip_hidden,
            "original_size": original_size,
            "crop_coords_top_left": crop_coords_top_left,
            "target_size": torch.tensor([self.size, self.size]),
        }

    def __len__(self):
        return len(self.data)


def collate_fn(data):
    images = torch.stack([example["image"] for example in data])
    kps_images = torch.stack([example["kps_image"] for example in data])
    target_kps_images = torch.stack([example["target_kps_image"] for example in data])
    align_matrices = torch.stack([example["align_matrix"] for example in data])
    target_align_matrices = torch.stack([example["target_align_matrix"] for example in data])
    text_input_ids = torch.cat([example["text_input_ids"] for example in data], dim=0)
    text_input_ids_2 = torch.cat([example["text_input_ids_2"] for example in data], dim=0)
    face_id_embed = torch.stack([example["face_id_embed"] for example in data])
    evaclip_embed = torch.stack([example["evaclip_embed"] for example in data])
    evaclip_hidden = torch.stack([example["evaclip_hidden"] for example in data])
    original_size = torch.stack([example["original_size"] for example in data])
    crop_coords_top_left = torch.stack([example["crop_coords_top_left"] for example in data])
    target_size = torch.stack([example["target_size"] for example in data])

    return {
        "images": images,
        "kps_images": kps_images,
        "target_kps_images": target_kps_images,
        "align_matrices": align_matrices,
        "target_align_matrices": target_align_matrices,
        "text_input_ids": text_input_ids,
        "text_input_ids_2": text_input_ids_2,
        "face_id_embed": face_id_embed,
        "evaclip_embed": evaclip_embed,
        "evaclip_hidden": evaclip_hidden,
        "original_size": original_size,
        "crop_coords_top_left": crop_coords_top_left,
        "target_size": target_size,
    }


class InstantIDAdapter(torch.nn.Module):
    """InstantIDAdapter"""
    def __init__(self, unet, controlnet, feature_proj_model, adapter_modules, ckpt_path=None, rerandom_latents=False):
        super().__init__()
        self.unet = unet
        self.controlnet = controlnet
        self.feature_proj_model = feature_proj_model
        self.adapter_modules = adapter_modules
        if ckpt_path is not None:
            self.load_from_checkpoint(ckpt_path, rerandom_latents)

    def forward(self,noisy_latents, timesteps, encoder_hidden_states, unet_added_cond_kwargs, feature_embeds, evaclip_embeds, evaclip_hiddens, controlnet_image):
        face_embedding = self.feature_proj_model(feature_embeds, evaclip_embeds, evaclip_hiddens, timesteps)
        encoder_hidden_states = torch.cat([encoder_hidden_states, face_embedding], dim=1)
        # ControlNet conditioning.
        down_block_res_samples, mid_block_res_sample = self.controlnet(
            noisy_latents,
            timesteps,
            encoder_hidden_states=face_embedding,  # Insightface feature
            added_cond_kwargs=unet_added_cond_kwargs,
            controlnet_cond=controlnet_image,  # keypoints image
            return_dict=False,
        )
        # Predict the noise residual.
        noise_pred = self.unet(
            noisy_latents,
            timesteps,
            encoder_hidden_states=encoder_hidden_states,
            added_cond_kwargs=unet_added_cond_kwargs,
            down_block_additional_residuals=[sample for sample in down_block_res_samples],
            mid_block_additional_residual=mid_block_res_sample,
        ).sample

        return noise_pred

    def load_from_checkpoint(self, ckpt_path, rerandom_latents):
        # Calculate original checksums
        orig_ip_proj_sum = torch.sum(torch.stack([torch.sum(p) for p in self.feature_proj_model.parameters()]))
        orig_adapter_sum = torch.sum(torch.stack([torch.sum(p) for p in self.adapter_modules.parameters()]))

        state_dict = torch.load(ckpt_path, map_location="cpu")

        if rerandom_latents:
            latents = state_dict["image_proj"]["latents"]
            latents_std = torch.std(latents)
            rand_latents = torch.randn_like(latents) * latents_std
            state_dict["image_proj"]["latents"] += rand_latents
            print("rerandom the pretrained latents")

        # Check if 'latents' exists in both the saved state_dict and the current model's state_dict
        strict_load_feature_proj_model = True
        if "latents" in state_dict["image_proj"] and "latents" in self.feature_proj_model.state_dict():
            # Check if the shapes are mismatched
            if state_dict["image_proj"]["latents"].shape != self.feature_proj_model.state_dict()["latents"].shape:
                print(f"Shapes of 'image_proj.latents' in checkpoint {ckpt_path} and current model do not match.")
                print("Removing 'latents' from checkpoint and loading the rest of the weights.")
                del state_dict["image_proj"]["latents"]
                strict_load_feature_proj_model = False

        # Load state dict for feature_proj_model and adapter_modules
        self.feature_proj_model.load_state_dict(state_dict["image_proj"], strict=False)
        self.adapter_modules.load_state_dict(state_dict["ip_adapter"], strict=True)

        # Calculate new checksums
        new_ip_proj_sum = torch.sum(torch.stack([torch.sum(p) for p in self.feature_proj_model.parameters()]))
        new_adapter_sum = torch.sum(torch.stack([torch.sum(p) for p in self.adapter_modules.parameters()]))

        # Verify if the weights have changed
        assert orig_ip_proj_sum != new_ip_proj_sum, "Weights of feature_proj_model did not change!"
        assert orig_adapter_sum != new_adapter_sum, "Weights of adapter_modules did not change!"

        print(f"Successfully loaded weights from checkpoint {ckpt_path}")


def inference(model, vae, latents, scheduler, text_embeds, unet_added_cond_kwargs, feat_embeds, evaclip_embeds, evaclip_hiddens, kps_images, guidance_scale=1):
    do_classifier_free_guidance = guidance_scale > 1
    if do_classifier_free_guidance:
        text_embeds = torch.cat([torch.zeros_like(text_embeds), text_embeds], dim=0)
        unet_added_cond_kwargs = {
            "text_embeds": torch.cat([torch.zeros_like(unet_added_cond_kwargs['text_embeds']), unet_added_cond_kwargs['text_embeds']], dim=0), 
            "time_ids": torch.cat([unet_added_cond_kwargs['time_ids'], unet_added_cond_kwargs['time_ids']], dim=0),
        }
        feat_embeds = torch.cat([torch.zeros_like(feat_embeds), feat_embeds], dim=0)
        evaclip_embeds = torch.cat([torch.zeros_like(evaclip_embeds), evaclip_embeds], dim=0)
        evaclip_hiddens = torch.cat([torch.zeros_like(evaclip_hiddens), evaclip_hiddens], dim=0)
        kps_images = torch.cat([kps_images] * 2, dim=0)
    latents = torch.randn_like(latents)
    # scale the initial noise by the standard deviation required by the scheduler
    latents = latents * scheduler.init_noise_sigma
    timesteps = scheduler.timesteps
    for t in timesteps:
        latent_model_input = torch.cat([latents] * 2) if do_classifier_free_guidance else latents
        latent_model_input = scheduler.scale_model_input(latent_model_input, t)
        noise_pred = model(latent_model_input, t, text_embeds, unet_added_cond_kwargs, feat_embeds, evaclip_embeds, evaclip_hiddens, kps_images)
        if do_classifier_free_guidance:
            noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
            noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond)
        latents = scheduler.step(noise_pred, t, latents, return_dict=False)[0]
    latents = latents.to(next(iter(vae.post_quant_conv.parameters())).dtype)
    with torch.no_grad():
        images = vae.decode(latents / vae.config.scaling_factor, return_dict=False)[0]
    return images
    

def parse_args():
    parser = argparse.ArgumentParser(description="Simple example of a training script.")
    parser.add_argument(
        "--pretrained_model_name_or_path",
        type=str,
        default=None,
        required=True,
        help="Path to pretrained model or model identifier from huggingface.co/models.",
    )
    parser.add_argument(
        "--pretrained_ip_adapter_path",
        type=str,
        default=None,
        help="Path to pretrained ip adapter model. If not specified weights are initialized randomly.",
    )
    parser.add_argument(
        "--controlnet_model_name_or_path",
        type=str,
        default=None,
        help="Path to pretrained controlnet model. If not specified weights are initialized from unet.",
    )
    parser.add_argument(
        "--pretrained_unet_path",
        type=str,
        default=None,
        help="Path to pretrained unet model. i.e. SDXL-lighting",
    )
    parser.add_argument(
        "--num_inference_steps",
        type=int,
        default=4,
        help="The number of denoising steps for SDXL-lighting",
    )
    parser.add_argument(
        "--id_loss_weight",
        type=float,
        default=1.0,
        help="id loss added weight",
    )
    parser.add_argument(
        "--id_loss_warmup_steps",
        type=int,
        default=0,
        help="id loss weight warmup steps",
    )
    parser.add_argument(
        "--inference_guidance_scale",
        type=float,
        default=1.0,
        help="classifier free guidance scale",
    )
    parser.add_argument(
        "--num_tokens",
        type=int,
        default=16,
        help="Number of tokens to query from the CLIP image encoding.",
    )
    parser.add_argument(
        "--checkpoints_total_limit",
        type=int,
        default=1,
        help=(
            "Save a checkpoint of the training state every X updates"
        ),
    )
    parser.add_argument(
        "--data_json_file",
        type=str,
        default=None,
        required=True,
        help="Training data",
    )
    parser.add_argument(
        "--data_root_path",
        type=str,
        default="",
        required=True,
        help="Training data root path",
    )
    parser.add_argument('--clip_proc_mode',
                        choices=["seg_align", "seg_crop", "orig_align", "orig_crop", "seg_align_pad",
                                 "orig_align_pad"],
                        default="orig_crop",
                        help='The mode to preprocess clip image encoder input.')

    parser.add_argument(
        "--image_encoder_path",
        type=str,
        default=None,
        required=True,
        help="Path to CLIP image encoder",
    )
    parser.add_argument(
        "--center_crop",
        default=False,
        action="store_true",
        help=(
            "Whether to center crop the input images to the resolution. If not set, the images will be randomly"
            " cropped. The images will be resized to the resolution first before cropping."
        ),
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="sd-ip_adapter",
        help="The output directory where the model predictions and checkpoints will be written.",
    )
    parser.add_argument(
        "--logging_dir",
        type=str,
        default="logs",
        help=(
            "[TensorBoard](https://www.tensorflow.org/tensorboard) log directory. Will default to"
            " *output_dir/runs/**CURRENT_DATETIME_HOSTNAME***."
        ),
    )
    parser.add_argument(
        "--resolution",
        type=int,
        default=512,
        help=(
            "The resolution for input images"
        ),
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=1e-4,
        help="Learning rate to use.",
    )
    parser.add_argument("--weight_decay", type=float, default=1e-2, help="Weight decay to use.")
    parser.add_argument("--num_train_epochs", type=int, default=100)
    parser.add_argument(
        "--train_batch_size", type=int, default=8, help="Batch size (per device) for the training dataloader."
    )
    parser.add_argument(
        "--max_data_loader_n_workers",
        type=int,
        default=8,
        help="max num workers for DataLoader (lower is less main RAM usage, faster epoch start and slower data loading)",
    )
    parser.add_argument(
        "--save_steps",
        type=int,
        default=2000,
        help=(
            "Save a checkpoint of the training state every X updates"
        ),
    )
    parser.add_argument(
        "--mixed_precision",
        type=str,
        default=None,
        choices=["no", "fp16", "bf16"],
        help=(
            "Whether to use mixed precision. Choose between fp16 and bf16 (bfloat16). Bf16 requires PyTorch >="
            " 1.10.and an Nvidia Ampere GPU.  Default to the value of accelerate config of the current system or the"
            " flag passed with the `accelerate.launch` command. Use this argument to override the accelerate config."
        ),
    )
    parser.add_argument(
        "--report_to",
        type=str,
        default="tensorboard",
        help=(
            'The integration to report the results and logs to. Supported platforms are `"tensorboard"`'
            ' (default), `"wandb"` and `"comet_ml"`. Use `"all"` to report to all integrations.'
        ),
    )
    parser.add_argument("--local_rank", type=int, default=-1, help="For distributed training: local_rank")
    parser.add_argument("--noise_offset", type=float, default=None, help="noise offset")
    parser.add_argument("--rerandom_latents", default=False, action="store_true",
        help="rerandom latents for learning new embedding"
    )
    deepspeed_utils.add_deepspeed_arguments(parser)

    args = parser.parse_args()
    env_local_rank = int(os.environ.get("LOCAL_RANK", -1))
    if env_local_rank != -1 and env_local_rank != args.local_rank:
        args.local_rank = env_local_rank

    return args


def save_model(unwrapped_model, global_step, output_dir, checkpoints_total_limit):
    # before saving state, check if this save would set us over the `checkpoints_total_limit`
    if checkpoints_total_limit is not None:
        checkpoints = os.listdir(output_dir)
        checkpoints = [d for d in checkpoints if d.startswith("checkpoint")]
        checkpoints = sorted(checkpoints, key=lambda x: int(x.split("-")[1]))
        # before we save the new checkpoint, we need to have at _most_ `checkpoints_total_limit - 1` checkpoints
        if len(checkpoints) >= checkpoints_total_limit:
            num_to_remove = len(checkpoints) - checkpoints_total_limit + 1
            removing_checkpoints = checkpoints[0:num_to_remove]
            print(
                f"{len(checkpoints)} checkpoints already exist, removing {len(removing_checkpoints)} checkpoints")
            print(f"removing checkpoints: {', '.join(removing_checkpoints)}")

            for removing_checkpoint in removing_checkpoints:
                removing_checkpoint = os.path.join(output_dir, removing_checkpoint)
                shutil.rmtree(removing_checkpoint)

    save_path = os.path.join(output_dir, f"checkpoint-{global_step}")
    unwrapped_model.save_weights(save_path)


def to_image(image):
    image = image.detach()
    image = (image / 2 + 0.5).clamp(0, 1)
    image = image.permute(0, 2, 3, 1).float().cpu().numpy()
    image = (image * 255).round().astype("uint8")
    return image


def save_images(src_images, dst_images, path, name):
    os.makedirs(path, exist_ok=True)
    src_images = to_image(src_images)
    dst_images = to_image(dst_images)
    images = np.concatenate([src_images, dst_images], axis=2)
    for i, image in enumerate(images):
        Image.fromarray(image).save(os.path.join(path, name + f'-b{i}.png'))


def get_id_loss(arcface_model, target_align_matrices, align_matrix, src_images, dst_images, size=112):
    with torch.no_grad():
        src_images_align = warp_affine(src_images, target_align_matrices, dsize=(size, size))
        dst_images_align = warp_affine(dst_images, align_matrix, dsize=(size, size))
        src_embeds = arcface_model(src_images_align)
        dst_embeds = arcface_model(dst_images_align)
    loss = 1 - F.cosine_similarity(src_embeds.float(), dst_embeds.float(), dim=1, eps=1e-6)
    loss = loss.mean()
    return loss
    

def main():
    args = parse_args()
    logging_dir = Path(args.output_dir, args.logging_dir)

    accelerator_project_config = ProjectConfiguration(project_dir=args.output_dir, logging_dir=logging_dir)
    deepspeed_plugin = deepspeed_utils.prepare_deepspeed_plugin(args)
    process_group_kwargs = [InitProcessGroupKwargs(timeout=timedelta(seconds=1800))]

    accelerator = Accelerator(
        mixed_precision=args.mixed_precision,
        log_with=args.report_to,
        project_config=accelerator_project_config,
        deepspeed_plugin=deepspeed_plugin,
        kwargs_handlers=process_group_kwargs
    )

    num_devices = accelerator.num_processes

    if accelerator.is_main_process:
        if args.output_dir is not None:
            os.makedirs(args.output_dir, exist_ok=True)

    # Load scheduler, tokenizer and models.
    noise_scheduler = DDPMScheduler.from_pretrained(args.pretrained_model_name_or_path, subfolder="scheduler")
    dpm_scheduler = DPMSolverSinglestepScheduler.from_pretrained(args.pretrained_model_name_or_path, subfolder="scheduler", timestep_spacing="trailing")
    tokenizer = CLIPTokenizer.from_pretrained(args.pretrained_model_name_or_path, subfolder="tokenizer")
    text_encoder = CLIPTextModel.from_pretrained(args.pretrained_model_name_or_path, subfolder="text_encoder")
    tokenizer_2 = CLIPTokenizer.from_pretrained(args.pretrained_model_name_or_path, subfolder="tokenizer_2")
    text_encoder_2 = CLIPTextModelWithProjection.from_pretrained(args.pretrained_model_name_or_path, subfolder="text_encoder_2")
    vae = AutoencoderKL.from_pretrained(args.pretrained_model_name_or_path, subfolder="vae")
    if args.pretrained_unet_path:
        unet = UNet2DConditionModel.from_config(args.pretrained_model_name_or_path, subfolder='unet')
        unet.load_state_dict(load_file(args.pretrained_unet_path))
    else:
        unet = UNet2DConditionModel.from_pretrained(args.pretrained_model_name_or_path, subfolder="unet")
    # image_encoder = CLIPVisionModelWithProjection.from_pretrained(args.image_encoder_path)
    if args.controlnet_model_name_or_path:
        print(f"Loading existing controlnet weights from {args.controlnet_model_name_or_path}")
        controlnet = ControlNetModel.from_pretrained(args.controlnet_model_name_or_path)
    else:
        print("Initializing controlnet weights from unet")
        controlnet = ControlNetModel.from_unet(unet)

    # load arcface torch model
    arcface_model = get_arcface(model_path='./checkpoints/backbone.pth')

    # freeze parameters of models to save more memory
    unet.requires_grad_(False)
    vae.requires_grad_(False)
    text_encoder.requires_grad_(False)
    text_encoder_2.requires_grad_(False)
    # image_encoder.requires_grad_(False)
    arcface_model.requires_grad_(False)
    controlnet.requires_grad_(True)
    controlnet.train()

    # ip-adapter: insightface feature
    num_tokens = 16

    feature_proj_model = Resampler(
        dim=1280,
        time_channel=320,
        time_embed_dim=1280,
        depth=4,
        dim_head=64,
        heads=20,
        num_queries=num_tokens,
        embedding_dim=512,
        embedding_dim2=768,
        embedding_hidden=1024,
        output_dim=unet.config.cross_attention_dim,
        ff_mult=4,
    )

    # init adapter modules
    attn_procs = {}
    unet_sd = unet.state_dict()
    for name in unet.attn_processors.keys():
        cross_attention_dim = None if name.endswith("attn1.processor") else unet.config.cross_attention_dim
        if name.startswith("mid_block"):
            hidden_size = unet.config.block_out_channels[-1]
        elif name.startswith("up_blocks"):
            block_id = int(name[len("up_blocks.")])
            hidden_size = list(reversed(unet.config.block_out_channels))[block_id]
        elif name.startswith("down_blocks"):
            block_id = int(name[len("down_blocks.")])
            hidden_size = unet.config.block_out_channels[block_id]
        if cross_attention_dim is None:
            attn_procs[name] = AttnProcessor()
        else:
            layer_name = name.split(".processor")[0]
            weights = {
                "to_k_ip.weight": unet_sd[layer_name + ".to_k.weight"],
                "to_v_ip.weight": unet_sd[layer_name + ".to_v.weight"],
            }
            attn_procs[name] = IPAttnProcessor(hidden_size=hidden_size, cross_attention_dim=cross_attention_dim, num_tokens=num_tokens)
            attn_procs[name].load_state_dict(weights)
    unet.set_attn_processor(attn_procs)
    adapter_modules = torch.nn.ModuleList(unet.attn_processors.values())
    # Instantiate InstantIDAdapter from pretrained model or from scratch.
    ip_adapter = InstantIDAdapter(unet, controlnet, feature_proj_model, adapter_modules, args.pretrained_ip_adapter_path, args.rerandom_latents)

    # # before resuming make hook for saving/loading to save/load the ip_adapter weights only
    # def save_model_hook(models, weights, output_dir):
    #     # pop weights of other models than ip_adapter to save only ip_adapter weights
    #     if accelerator.is_main_process:
    #         remove_indices = []
    #         for i, model in enumerate(models):
    #             if not isinstance(model, type(accelerator.unwrap_model(ip_adapter))):
    #                 remove_indices.append(i)
    #         for i in reversed(remove_indices):
    #             weights.pop(i)
    #         # print(f"save model hook: {len(weights)} weights will be saved")

    # def load_model_hook(models, input_dir):
    #     # remove models except ip_adapter
    #     remove_indices = []
    #     for i, model in enumerate(models):
    #         if not isinstance(model, type(accelerator.unwrap_model(ip_adapter))):
    #             remove_indices.append(i)
    #     for i in reversed(remove_indices):
    #         models.pop(i)
    #     # print(f"load model hook: {len(models)} models will be loaded")
        
    # Register a hook function to process the state of a specific module before saving.
    def save_model_hook(models, weights, output_dir):
        if accelerator.is_main_process:
            # find instance of InstantIDAdapter Model.
            for i, model_instance in enumerate(models):
                model_instance = accelerator.unwrap_model(model_instance)
                if isinstance(model_instance, InstantIDAdapter):
                    # When saving a checkpoint, only save the ip-adapter and image_proj, do not save the unet.
                    ip_adapter_state = {
                        'image_proj': model_instance.feature_proj_model.state_dict(),
                        'ip_adapter': model_instance.adapter_modules.state_dict(),
                    }
                    torch.save(ip_adapter_state, os.path.join(output_dir, 'ip-adapter.bin'))
                    print(f"IP-Adapter Model weights saved in {os.path.join(output_dir, 'ip-adapter.bin')}")
                    # Save controlnet separately.
                    sub_dir = "ControlNetModel"
                    model_instance.controlnet.save_pretrained(os.path.join(output_dir, sub_dir))
                    print(f"Controlnet weights saved in {os.path.join(output_dir, sub_dir)}")
                    # Remove the corresponding weights from the weights list because they have been saved separately.
                    # Remember not to delete the corresponding model, otherwise, you will not be able to save the model
                    # starting from the second epoch.
                    if len(weights) > i:
                        weights.pop(i)
                    break

    def load_model_hook(models, input_dir):
        # find instance of InstantIDAdapter Model.
        while len(models) > 0:
            model_instance = models.pop()
            if isinstance(model_instance, InstantIDAdapter):
                ip_adapter_path = os.path.join(input_dir, 'ip-adapter.bin')
                if os.path.exists(ip_adapter_path):
                    ip_adapter_state = torch.load(ip_adapter_path)
                    model_instance.feature_proj_model.load_state_dict(ip_adapter_state['image_proj'])
                    model_instance.adapter_modules.load_state_dict(ip_adapter_state['ip_adapter'])
                    sub_dir = "ControlNetModel"
                    model_instance.controlnet.from_pretrained(os.path.join(input_dir, sub_dir))
                    print(f"Model weights loaded from {ip_adapter_path}")
                else:
                    print(f"No saved weights found at {ip_adapter_path}")


    # Register hook functions for saving  and loading.
    accelerator.register_save_state_pre_hook(save_model_hook)
    accelerator.register_load_state_pre_hook(load_model_hook)

    weight_dtype = torch.float32
    if accelerator.mixed_precision == "fp16":
        weight_dtype = torch.float16
    elif accelerator.mixed_precision == "bf16":
        weight_dtype = torch.bfloat16
    # unet.to(accelerator.device, dtype=weight_dtype)  # error
    vae.to(accelerator.device)  # use fp32
    text_encoder.to(accelerator.device, dtype=weight_dtype)
    text_encoder_2.to(accelerator.device, dtype=weight_dtype)
    # image_encoder.to(accelerator.device, dtype=weight_dtype)
    # controlnet.to(accelerator.device, dtype=weight_dtype)  # error
    controlnet.to(accelerator.device)
    arcface_model.to(accelerator.device)

    # trainable params
    params_to_opt = itertools.chain(ip_adapter.feature_proj_model.parameters(),
                                    ip_adapter.adapter_modules.parameters(),
                                    ip_adapter.controlnet.parameters())
    # pytorch_total_params = sum(p.numel() for p in params_to_opt if p.requires_grad) # 1673779984 params
    optimizer = torch.optim.AdamW(params_to_opt, lr=args.learning_rate, weight_decay=args.weight_decay)

    # dataloader
    train_dataset = MyDataset(args.data_json_file, tokenizer=tokenizer, tokenizer_2=tokenizer_2, size=args.resolution,
                              center_crop=args.center_crop, image_root_path=args.data_root_path, do_cfg=False)
    total_data_size = len(train_dataset)
    n_workers = min(args.max_data_loader_n_workers, os.cpu_count())  # cpu_count or max_data_loader_n_workers

    train_dataloader = torch.utils.data.DataLoader(
        train_dataset,
        shuffle=True,
        collate_fn=collate_fn,
        batch_size=args.train_batch_size,
        num_workers=n_workers,
    )

    # Prepare everything with our `accelerator`.
    ip_adapter, optimizer, train_dataloader = accelerator.prepare(ip_adapter, optimizer, train_dataloader)

    # Restore checkpoints
    checkpoint_folders = [folder for folder in os.listdir(args.output_dir) if folder.startswith('checkpoint-')]
    if checkpoint_folders:
        # Extract step numbers from all checkpoints and find the maximum step number
        global_step = max(int(folder.split('-')[-1]) for folder in checkpoint_folders if folder.split('-')[-1].isdigit())
        checkpoint_path = os.path.join(args.output_dir, f"checkpoint-{global_step}")
        # Load the checkpoint
        print(f"checkpoint folders found, train from global step {global_step}.")
        accelerator.load_state(checkpoint_path)
    else:
        global_step = 0
        print("No checkpoint folders found, train from scratch.")

    # global_step = 0
    # Calculate steps per epoch and the current epoch and its step number
    # steps_per_epoch = total_data_size // (args.train_batch_size * num_devices)
    # current_epoch = global_step // steps_per_epoch
    # current_step_in_epoch = global_step % steps_per_epoch

    # Training loop
    for epoch in range(0, args.num_train_epochs):
        begin = time.perf_counter()
        for step, batch in enumerate(train_dataloader):
            load_data_time = time.perf_counter() - begin
            with accelerator.accumulate(ip_adapter):
                images = batch["images"].to(accelerator.device, dtype=torch.float32)
                # Convert images to latent space
                with torch.no_grad():
                    # vae of sdxl should use fp32
                    latents = vae.encode(images).latent_dist.sample()
                    latents = latents * vae.config.scaling_factor
                    latents = latents.to(accelerator.device, dtype=weight_dtype)

                # Sample noise that we'll add to the latents
                noise = torch.randn_like(latents)
                if args.noise_offset:
                    # https://www.crosslabs.org//blog/diffusion-with-offset-noise
                    noise += args.noise_offset * torch.randn((latents.shape[0], latents.shape[1], 1, 1)).to(
                        accelerator.device, dtype=weight_dtype)

                bsz = latents.shape[0]
                # Sample a random timestep for each image
                timesteps = torch.randint(0, noise_scheduler.num_train_timesteps, (bsz,), device=latents.device)
                timesteps = timesteps.long()

                # Add noise to the latents according to the noise magnitude at each timestep
                # (this is the forward diffusion process)
                noisy_latents = noise_scheduler.add_noise(latents, noise, timesteps)

                # get feature embeddings, with cfg
                feat_embeds = batch["face_id_embed"].to(accelerator.device, dtype=weight_dtype)
                evaclip_embeds = batch["evaclip_embed"].to(accelerator.device, dtype=weight_dtype)
                evaclip_hiddens = batch["evaclip_hidden"].to(accelerator.device, dtype=weight_dtype)
                kps_images = batch["kps_images"].to(accelerator.device, dtype=weight_dtype)
                target_kps_images = batch["target_kps_images"].to(accelerator.device, dtype=weight_dtype)
                align_matrices = batch["align_matrices"].to(accelerator.device)
                target_align_matrices = batch["target_align_matrices"].to(accelerator.device)

                # for other experiments
                # clip_images = []
                # for clip_image, drop_image_embed in zip(batch["clip_images"], batch["drop_image_embeds"]):
                #     if drop_image_embed == 1:
                #         clip_images.append(torch.zeros_like(clip_image))
                #     else:
                #         clip_images.append(clip_image)
                # clip_images = torch.stack(clip_images, dim=0)
                # with torch.no_grad():
                #     image_embeds = image_encoder(clip_images.to(accelerator.device, dtype=weight_dtype),
                #                                  output_hidden_states=True).hidden_states[-2]

                with torch.no_grad():
                    encoder_output = text_encoder(batch['text_input_ids'].to(accelerator.device), output_hidden_states=True)
                    text_embeds = encoder_output.hidden_states[-2]
                    encoder_output_2 = text_encoder_2(batch['text_input_ids_2'].to(accelerator.device), output_hidden_states=True)
                    pooled_text_embeds = encoder_output_2[0]
                    text_embeds_2 = encoder_output_2.hidden_states[-2]
                    text_embeds = torch.concat([text_embeds, text_embeds_2], dim=-1)  # concat

                # add cond
                add_time_ids = [
                    batch["original_size"].to(accelerator.device),
                    batch["crop_coords_top_left"].to(accelerator.device),
                    batch["target_size"].to(accelerator.device),
                ]
                add_time_ids = torch.cat(add_time_ids, dim=1).to(accelerator.device, dtype=weight_dtype)
                unet_added_cond_kwargs = {"text_embeds": pooled_text_embeds, "time_ids": add_time_ids}

                noise_pred = ip_adapter(noisy_latents, timesteps, text_embeds, unet_added_cond_kwargs, feat_embeds, evaclip_embeds, evaclip_hiddens, kps_images)

                sd_loss = F.mse_loss(noise_pred.float(), noise.float(), reduction="mean")

                dpm_scheduler.set_timesteps(args.num_inference_steps, device=accelerator.device)
                src_images = inference(ip_adapter, vae, latents, dpm_scheduler, text_embeds, unet_added_cond_kwargs, feat_embeds, evaclip_embeds, evaclip_hiddens, target_kps_images, args.inference_guidance_scale)
                id_loss = get_id_loss(arcface_model, target_align_matrices, align_matrices, src_images, images)
                id_loss_weight = args.id_loss_weight * global_step / args.id_loss_warmup_steps if global_step < args.id_loss_warmup_steps else args.id_loss_weight
                total_loss = sd_loss + id_loss * id_loss_weight

                # Gather the losses across all processes for logging (if we use distributed training).
                avg_loss = accelerator.gather(total_loss.repeat(args.train_batch_size)).mean().item()
                avg_sd_loss = accelerator.gather(sd_loss.repeat(args.train_batch_size)).mean().item()
                avg_id_loss = accelerator.gather(id_loss.repeat(args.train_batch_size)).mean().item()

                # Backpropagate
                accelerator.backward(total_loss)
                optimizer.step()
                optimizer.zero_grad()

                now = datetime.now()
                formatted_time = now.strftime('%Y-%m-%d %H:%M:%S')
                if accelerator.is_local_main_process and step % 10 == 0:
                    print("[{}]:{}/{} epoch {}, global_step {}, step {}, data_time: {:.3f}, time: {:.3f}, step_loss: {:.6f}, sd_loss: {:.6f}, id_loss: {:.6f}".format(
                        formatted_time, accelerator.process_index, accelerator.num_processes, 
                        epoch, global_step, step, load_data_time, time.perf_counter() - begin, avg_loss, avg_sd_loss, avg_id_loss)
                    )
                    save_images(src_images, images, os.path.join(args.output_dir, 'id_image'), f'epoch{epoch}_step{global_step}')

            global_step += 1
            if global_step % args.save_steps == 0:
                accelerator.wait_for_everyone()
                # if accelerator.is_main_process:
                # before saving state, check if this save would set us over the `checkpoints_total_limit`
                if accelerator.is_main_process and args.checkpoints_total_limit is not None:
                    checkpoints = os.listdir(args.output_dir)
                    checkpoints = [d for d in checkpoints if d.startswith("checkpoint")]
                    checkpoints = sorted(checkpoints, key=lambda x: int(x.split("-")[1]))
                    # before we save the new checkpoint, we need to have at _most_ `checkpoints_total_limit - 1` checkpoints
                    if len(checkpoints) >= args.checkpoints_total_limit:
                        num_to_remove = len(checkpoints) - args.checkpoints_total_limit + 1
                        removing_checkpoints = checkpoints[0:num_to_remove]
                        print(
                            f"{len(checkpoints)} checkpoints already exist, removing {len(removing_checkpoints)} checkpoints")
                        print(f"removing checkpoints: {', '.join(removing_checkpoints)}")

                        for removing_checkpoint in removing_checkpoints:
                            removing_checkpoint = os.path.join(args.output_dir, removing_checkpoint)
                            shutil.rmtree(removing_checkpoint)

                save_path = os.path.join(args.output_dir, f"checkpoint-{global_step}")
                accelerator.save_state(save_path)

            begin = time.perf_counter()


if __name__ == "__main__":
    main()
