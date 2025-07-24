import os
import glob
import re
import cv2
import torch
import numpy as np
from PIL import Image
from collections import defaultdict
from sam2.build_sam import build_sam2_video_predictor

# ==== 用户参数设置 ====
PROMPT_TYPE = 'points'  # 可选：'points', 'boxes', 'masks'
prompt_dir = '/mnt/data/lyx/sam2-main/videos/dino/points_pt'
video_dir = '/mnt/data/lyx/sam2-main/results/box/dino/flare'
save_dir = '/mnt/data/lyx/sam2-main/results/AAAI26/box/grounding'
save_mask_dir = '/mnt/data/lyx/sam2-main/results/AAAI26/box/grounding/mask'
sam2_checkpoint = 'checkpoints/sam2.1_hiera_large.pt'
model_cfg = 'configs/sam2.1/sam2.1_hiera_l.yaml'
STEP = 5

# ==== 自动精度设置 ====
torch.autocast(device_type="cuda", dtype=torch.bfloat16).__enter__()
if torch.cuda.get_device_properties(0).major >= 8:
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

def extract_frame_index(filename):
    match = re.search(r'_(\d+)\.', filename)
    return int(match.group(1)) if match else -1

def get_color_for_label(label_id):
    label_id = int(label_id)
    np.random.seed(label_id * 13)
    return np.random.randint(0, 256, size=3, dtype=np.uint8)

def add_prompt(predictor, state, frame_idx, obj_id, prompt, prompt_type):
    if prompt_type == 'masks':
        mask_tensor = torch.from_numpy(prompt).float().cuda()
        predictor.add_new_mask(state, frame_idx, obj_id, mask_tensor)
    elif prompt_type == 'points':
        coords, labels = prompt
        predictor.add_new_points(state, frame_idx, obj_id, coords, labels)
    elif prompt_type == 'boxes':
        predictor.add_new_box(state, frame_idx, obj_id, prompt)
    else:
        raise NotImplementedError(f"Unsupported PROMPT_TYPE: {prompt_type}")

# ==== 加载图像帧并排序 ====
os.makedirs(save_dir, exist_ok=True)
os.makedirs(save_mask_dir, exist_ok=True)
frame_names = sorted(
    [f for f in os.listdir(video_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp'))],
    key=extract_frame_index
)

# ==== 一次性生成逆序文件夹 ====
reverse_video_dir = os.path.join(video_dir, '__reversed_temp')
os.makedirs(reverse_video_dir, exist_ok=True)
reverse_frame_names = list(reversed(frame_names))
for idx, fname in enumerate(reverse_frame_names):
    src = os.path.join(video_dir, fname)
    dst = os.path.join(reverse_video_dir, f'{idx}.png')
    if not os.path.exists(dst):
        img = cv2.imread(src)
        cv2.imwrite(dst, img)
print(f"已生成逆序缓存: {reverse_video_dir}")

# ==== 初始化模型 ====
predictor = build_sam2_video_predictor(model_cfg, sam2_checkpoint)
video_segments = {}
total_frames = len(frame_names)
pt_files = {os.path.splitext(os.path.basename(f))[0]: f for f in glob.glob(os.path.join(prompt_dir, '*.pt'))}

# ==== 分段处理 ====
for start in range(0, total_frames, STEP):
    end = min(start + STEP, total_frames)
    start_frame_name = frame_names[start]
    basename = os.path.splitext(start_frame_name)[0]

    if basename not in pt_files:
        continue

    pt_path = pt_files[basename]
    print(f"处理段落帧: {start}-{end - 1}, 提示文件: {os.path.basename(pt_path)}")
    prompt_data = torch.load(pt_path)

    # === 前向传播 ===
    inference_state = predictor.init_state(video_path=video_dir)
    predictor.reset_state(inference_state)
    for obj_id in prompt_data:
        add_prompt(predictor, inference_state, start, obj_id, prompt_data[obj_id], PROMPT_TYPE)

    for out_frame_idx, out_obj_ids, out_mask_logits in predictor.propagate_in_video(
        inference_state,
        start_frame_idx=start,
        max_frame_num_to_track=(end - start),
        reverse=False):

        if out_frame_idx not in video_segments:
            video_segments[out_frame_idx] = {}
        for j, out_obj_id in enumerate(out_obj_ids):
            out_obj_id = int(out_obj_id)
            new_logit = out_mask_logits[j].cpu().numpy()
            existing_logit = video_segments[out_frame_idx].get(out_obj_id)
            if existing_logit is not None:
                fused = np.maximum(existing_logit, new_logit)
                video_segments[out_frame_idx][out_obj_id] = fused
            else:
                video_segments[out_frame_idx][out_obj_id] = new_logit

    # === 反向传播（只用已生成的逆序文件夹） ===
    if start >= STEP:
        rev_idx = total_frames - 1 - start
        inference_state = predictor.init_state(video_path=reverse_video_dir)
        predictor.reset_state(inference_state)
        for obj_id in prompt_data:
            add_prompt(predictor, inference_state, rev_idx, obj_id, prompt_data[obj_id], PROMPT_TYPE)

        for out_rev_idx, out_obj_ids, out_mask_logits in predictor.propagate_in_video(
            inference_state,
            start_frame_idx=rev_idx,
            max_frame_num_to_track=STEP,
            reverse=True):

            orig_idx = total_frames - 1 - out_rev_idx
            if orig_idx not in video_segments:
                video_segments[orig_idx] = {}
            for j, out_obj_id in enumerate(out_obj_ids):
                out_obj_id = int(out_obj_id)
                new_logit = out_mask_logits[j].cpu().numpy()
                existing_logit = video_segments[orig_idx].get(out_obj_id)
                if existing_logit is not None:
                    fused = np.maximum(existing_logit, new_logit)
                    video_segments[orig_idx][out_obj_id] = fused
                else:
                    video_segments[orig_idx][out_obj_id] = new_logit

# ==== 可视化并保存 mask 图 ====
def add_mask2(image, mask, color_id):
    int_mask = mask.astype(np.uint8)
    int_mask_3d = np.dstack((int_mask, int_mask, int_mask))
    mask_color = get_color_for_label(color_id)
    color_mask = np.full_like(image, mask_color)
    color_mask[int_mask == 0] = 0
    res = cv2.addWeighted(image, 0.6, color_mask, 0.4, 0, dtype=cv2.CV_8U)
    res[int_mask_3d == 0] = image[int_mask_3d == 0]

    ys, xs = np.where(int_mask == 1)
    if len(xs) > 0:
        center_x = int(xs.mean())
        center_y = int(ys.mean())
        cv2.putText(res, f"{color_id}", (center_x, center_y),
                    fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                    fontScale=0.6,
                    color=(255, 255, 255),
                    thickness=2,
                    lineType=cv2.LINE_AA)
    return res

for out_frame_idx in range(len(frame_names)):
    frame_name = frame_names[out_frame_idx]
    frame_path = os.path.join(video_dir, frame_name)
    now_img = cv2.imread(frame_path)

    if out_frame_idx in video_segments:
        fused_logit_dict = defaultdict(lambda: None)
        for out_obj_id, logit in video_segments[out_frame_idx].items():
            out_obj_id = int(out_obj_id)
            if fused_logit_dict[out_obj_id] is None:
                fused_logit_dict[out_obj_id] = logit
            else:
                fused_logit_dict[out_obj_id] = np.maximum(fused_logit_dict[out_obj_id], logit)

        for out_obj_id in sorted(fused_logit_dict.keys()):
            out_logit = fused_logit_dict[out_obj_id]
            mask = (out_logit > 0.4).astype(np.uint8)
            mask = np.squeeze(mask)
            if mask.ndim != 2:
                raise ValueError(f"保存前 mask 必须是 2D 图像，但当前 shape 为 {mask.shape}")

            now_img = add_mask2(now_img, mask, out_obj_id)

            mask_save_dir = os.path.join(save_mask_dir, str(out_obj_id))
            os.makedirs(mask_save_dir, exist_ok=True)
            mask_path = os.path.join(mask_save_dir, frame_name)
            cv2.imwrite(mask_path, mask * 255)

    save_path = os.path.join(save_dir, frame_name)
    cv2.imwrite(save_path, now_img)

print(f"所有提示引导的分割结果已保存至: {save_dir}")
print(f"所有 label 分别的 mask 已保存至: {save_mask_dir}")

# ==== 可选：结束后删除逆序文件夹 ====
import shutil
shutil.rmtree(reverse_video_dir)
print(f"已删除逆序缓存: {reverse_video_dir}")
