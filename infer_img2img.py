import cv2
import torch
import numpy as np
from PIL import Image, ImageOps
import PIL
from diffusers.models import ControlNetModel, UNet2DConditionModel
from diffusers import DPMSolverMultistepScheduler, DPMSolverSinglestepScheduler
from safetensors.torch import load_file
import math
import os
import glob
from insightface.app import FaceAnalysis
from pipeline_stable_diffusion_xl_instantid_img2img import StableDiffusionXLInstantIDImg2ImgPipeline
from evaclip import EVACLIP

def draw_kps(image_pil, kps, color_list=[(255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0), (255, 0, 255)]):
    stickwidth = 4
    limbSeq = np.array([[0, 2], [1, 2], [3, 2], [4, 2]])
    kps = np.array(kps)

    w, h = image_pil.size
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

def resize_img(input_image, max_side=1280, min_side=1024, size=None, 
               pad_to_max_side=False, mode=Image.BILINEAR, base_pixel_number=64):

    w, h = input_image.size
    if size is not None:
        w_resize_new, h_resize_new = size
    else:
        ratio = min_side / min(h, w)
        w, h = round(ratio*w), round(ratio*h)
        ratio = max_side / max(h, w)
        input_image = input_image.resize([round(ratio*w), round(ratio*h)], mode)
        w_resize_new = (round(ratio * w) // base_pixel_number) * base_pixel_number
        h_resize_new = (round(ratio * h) // base_pixel_number) * base_pixel_number
    input_image = input_image.resize([w_resize_new, h_resize_new], mode)

    if pad_to_max_side:
        res = np.ones([max_side, max_side, 3], dtype=np.uint8) * 255
        offset_x = (max_side - w_resize_new) // 2
        offset_y = (max_side - h_resize_new) // 2
        res[offset_y:offset_y+h_resize_new, offset_x:offset_x+w_resize_new] = np.array(input_image)
        input_image = Image.fromarray(res)
    return input_image

def get_face_info(app, image):
    image = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
    for size in [(size, size) for size in range(640, 256, -64)]:
        app.det_model.input_size = size
        faces = app.get(image)
        if faces:
            break
        else:
            print(f'InsightFace detection resolution lowered to {size}')
    else:
        print('no face detect')
        return
    
    face = sorted(faces, key=lambda x:(x['bbox'][2]-x['bbox'][0])*x['bbox'][3]-x['bbox'][1])[-1] # only use the maximum face
    return face
    

def get_image_list(path):
    image_path_list = glob.glob(os.path.join(path, '*.*'))
    image_list = []
    for image_path in image_path_list:
        image = Image.open(image_path).convert('RGB')
        image = ImageOps.exif_transpose(image)
        image = resize_img(image)
        image_list.append(image)
    return image_list

def get_checkpoint_path(path, index=-1):
    path_list = glob.glob(os.path.join(path, 'checkpoint-*'))
    path_list = sorted(path_list, key=lambda x: int(x.split('checkpoint-')[1]))
    return path_list[index]

def get_model_version(path):
    if './checkpoints' in path: return 'pretrain'
    version = '_'.join(path.split('/')[-2:])
    return version

if __name__ == "__main__":
    hf_home = '~/huggingface'
    pretrained_root = "./checkpoints"
    trained_root = "./InstantID_output/v1"
    checkpoint_path = get_checkpoint_path(trained_root, -1)
    model_root = checkpoint_path
    model_version = get_model_version(model_root)
    
    app = FaceAnalysis(name='antelopev2', root=pretrained_root, providers=['CUDAExecutionProvider', 'CPUExecutionProvider'])
    app.prepare(ctx_id=0, det_size=(640, 640))
    evaclip_model = EVACLIP()

    # Path to InstantID models
    face_adapter = os.path.join(model_root, 'ip-adapter.bin')
    controlnet_path = os.path.join(model_root, 'ControlNetModel')

    # Load pipeline
    controlnet = ControlNetModel.from_pretrained(controlnet_path, torch_dtype=torch.float16)

    # base_model_path = 'stabilityai/stable-diffusion-xl-base-1.0'
    base_model_path = os.path.join(hf_home, "hub/models--stabilityai--stable-diffusion-xl-base-1.0/snapshots/462165984030d82259a11f4367a4eed129e94a7b")

    # unet = UNet2DConditionModel.from_config(base_model_path, subfolder='unet').to(torch.float16)
    # unet.load_state_dict(load_file(os.path.join(pretrained_root, 'sdxl_lightning_8step_unet.safetensors')))

    pipe = StableDiffusionXLInstantIDImg2ImgPipeline.from_pretrained(
        base_model_path,
        # unet=unet,
        controlnet=controlnet,
        torch_dtype=torch.float16,
    )
    pipe.scheduler = DPMSolverSinglestepScheduler.from_config(
        pipe.scheduler.config, timestep_spacing="trailing", use_karras_sigmas=True
    )
    pipe.cuda()
    pipe.load_ip_adapter_instantid(face_adapter)

    print("#" * 30, model_version, "#" * 30)

    prompt = "1man, solo, in the center, (detailed eyes), (detailed face), (looking_at_viewer), high definition, sharp focus, real photo, captivating portrait"
    n_prompt = "(worst quality, low quality, bad_pictures), blurry, lowres, bad anatomy, naked, nude, nipples, vagina, glans, ugly, pregnant, vore, duplicate, morbid,mut ilated, tran nsexual, hermaphrodite, long neck"
    in_path = './examples/images'
    temp_path = './examples/inputs'
    out_path = './examples/outputs_i2i'
    image_list = get_image_list(temp_path)
    seed = 42
    for image_path in glob.glob(os.path.join(in_path, '*')):
        image_name = os.path.basename(image_path).split('.')[0]
        image = Image.open(image_path).convert('RGB')
        image = ImageOps.exif_transpose(image)
        image = resize_img(image)
        face = get_face_info(app, image)
        face_emb, face_bbox = face.embedding, face.bbox
        image_face = image.crop(face_bbox)
        clip_info = evaclip_model.predict(image_face)
        clip_emb, clip_hidden = clip_info['clip_emb'], clip_info['clip_hidden']

        pipe.set_ip_adapter_scale(0.8)
        print(image_name)
        for i, temp_image in enumerate(image_list):
            face_temp = get_face_info(app, temp_image)
            face_kps = draw_kps(temp_image, face_temp.kps)
            image = pipe(
                image=temp_image,
                prompt=prompt,
                negative_prompt=n_prompt,
                image_embeds=face_emb,
                evaclip_embeds=clip_emb,
                evaclip_hiddens=clip_hidden,
                control_image=face_kps,
                controlnet_conditioning_scale=0.4,
                num_inference_steps=30,
                strength=0.6,
                guidance_scale=5,
                generator=torch.Generator(device="cpu").manual_seed(seed),
            ).images[0]
            save_path = f'{model_version}-{image_name}_{i}.png'
            image.save(os.path.join(out_path, save_path))
            print('save to', save_path)