import argparse
import os
import cv2
import torch
import torchvision.transforms as standard_transforms
from PIL import Image
from models import build_model
import numpy as np
import re
import engine as eng
import shutil

def prepare_log_directory(log_dir):
    # بررسی وجود پوشه log
    if os.path.exists(log_dir):
        # حذف پوشه
        shutil.rmtree(log_dir)
    # ایجاد پوشه جدید
    os.makedirs(log_dir)
    print(f"Log directory prepared: {log_dir}")
    
def get_args_parser():
    parser = argparse.ArgumentParser('Set parameters for P2PNet evaluation', add_help=False)
    
    parser.add_argument('--backbone', default='vgg16_bn', type=str,
                        help="name of the convolutional backbone to use")
    parser.add_argument('--row', default=2, type=int,
                        help="row number of anchor points")
    parser.add_argument('--line', default=2, type=int,
                        help="line number of anchor points")
    parser.add_argument('--output_dir', default='',
                        help='path where to save')
    parser.add_argument('--weight_path', default='',
                        help='path where the trained weights saved')
    parser.add_argument('--gpu_id', default=0, type=int, help='the gpu used for evaluation')

    return parser

def count_lines_in_txt(txt_path):
    """تعداد خطوط غیر خالی فایل txt را شمارش می‌کند."""
    with open(txt_path, 'r') as f:
        lines = f.readlines()
        return len([line for line in lines if line.strip()])

def calculate_nap(predicted_counts, actual_counts):
    """
    محاسبه NAP (Normalized Average Precision)
    """
    precision = []
    for predicted, actual in zip(predicted_counts, actual_counts):
        # اگر تعداد واقعی صفر نباشد، دقت نرمال‌شده محاسبه می‌شود
        precision.append(min(predicted, actual) / float(actual) if actual != 0 else 0)
    
    nap = np.mean(precision)  # میانگین دقت نرمال‌شده
    return nap
import cv2
import numpy as np
from PIL import Image

def apply_denoise_filter(img_raw, filter_type):
    """
    اعمال فیلتر رفع نویز به تصویر.
    :param img_raw: تصویر ورودی از نوع PIL Image
    :param filter_type: نوع فیلتر ('gaussian', 'median', 'bilateral')
    :return: تصویر فیلتر شده از نوع PIL Image
    """
    # تبدیل تصویر از PIL به فرمت numpy
    img_np = np.array(img_raw)

    if filter_type == 'gaussian':
        # اعمال Gaussian Blur
        img_denoised = cv2.GaussianBlur(img_np, (5, 5), 0)
    elif filter_type == 'median':
        # اعمال Median Filter
        img_denoised = cv2.medianBlur(img_np, 5)
    elif filter_type == 'bilateral':
        # اعمال Bilateral Filter
        img_denoised = cv2.bilateralFilter(img_np, 9, 75, 75)
    else:
        raise ValueError("نوع فیلتر نامعتبر است. لطفاً یکی از 'gaussian', 'median', یا 'bilateral' را انتخاب کنید.")

    # تبدیل تصویر از numpy به PIL
    img_denoised = Image.fromarray(img_denoised)
    return img_denoised

# نمونه استفاده در تست یک تصویر
def test__one_img_with_denoise(img_path, transform, device, model, args, txt_path,scene_number, filter_type='gaussian'):
    if not os.path.exists(img_path):
        print(f"File not found: {img_path}")
        return 0, 0, 0, 0  # برگشت 0 برای پیش‌بینی و 0 برای تعداد واقعی
    print(f"^^^^^^^^img_path:{img_path} , txt_path:{txt_path} ")
    # بارگذاری تصویر
    img_raw = Image.open(img_path).convert('RGB')
    # تغییر اندازه
    width, height = img_raw.size
    new_width = width // 128 * 128
    new_height = height // 128 * 128
    img_raw = img_raw.resize((new_width, new_height), Image.Resampling.LANCZOS)

    # اعمال فیلتر رفع نویز
    img_raw = apply_denoise_filter(img_raw, filter_type)

    img = transform(img_raw)

    samples = torch.Tensor(img).unsqueeze(0)
    samples = samples.to(device)
    
    outputs = model(samples)
    outputs_scores = torch.nn.functional.softmax(outputs['pred_logits'], -1)[:, :, 1][0]

    outputs_points = outputs['pred_points'][0]
    num_people = outputs['pred_points'].shape[1]

    threshold = 0.5
    points = outputs_points[outputs_scores > threshold].detach().cpu().numpy().tolist()
    predict_cnt = int((outputs_scores > threshold).sum())

    img_to_draw = cv2.cvtColor(np.array(img_raw), cv2.COLOR_RGB2BGR)
    for p in points:
        img_to_draw = cv2.circle(img_to_draw, (int(p[0]), int(p[1])), 2, (0, 0, 255), -1)

    num_lines = count_lines_in_txt(txt_path)
    
    # output_filename = os.path.join(args.output_dir, f'pred_{num_lines}_{os.path.basename(img_path)}_{predict_cnt}')
    output_filename = os.path.join(args.output_dir, f'img_num:{os.path.basename(img_path)}_gt:{num_lines}_predict:{predict_cnt}.jpg')
    cv2.imwrite(output_filename, img_to_draw)

    print(f'########## predict:{predict_cnt} , real:{num_lines}, real_lines:{num_lines}')
    
    # محاسبه MAE و MSE
    mae = abs(num_lines - predict_cnt)
    mse = (num_lines - predict_cnt) ** 2

    return num_lines, predict_cnt, mae, mse

def main(args):
    os.environ["CUDA_VISIBLE_DEVICES"] = '{}'.format(args.gpu_id)

    print(args)
    device = torch.device('cuda')
    model = build_model(args)
    model.to(device)

    if args.weight_path is not None:
        checkpoint = torch.load(args.weight_path, map_location='cpu')
        model.load_state_dict(checkpoint['model'])
        
    model.eval()
    
    transform = standard_transforms.Compose([
        standard_transforms.ToTensor(), 
        standard_transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    
    prepare_log_directory(args.output_dir)

    scene_dir = "/content/drive/MyDrive/CrowdCounting-P2PNet/DATA_ROOT/test"
    scene_folders = sorted([os.path.join(scene_dir, folder) for folder in os.listdir(scene_dir) if os.path.isdir(os.path.join(scene_dir, folder))])

    all_real_counts = []
    all_pred_counts = []
    all_mae = []
    all_mse = []

    for scene_folder in scene_folders:
        scene_name = os.path.basename(scene_folder)
        match = re.search(r'(\d+)$', scene_name)

        if match:
            scene_number = match.group(1)
            if int(scene_number) < 10:  # بررسی عدد یک‌رقمی
              scene_number = scene_number.zfill(2)  # اضافه کردن صفر در ابتدای عدد

            img_filename = os.path.join(scene_folder, f"img{scene_number}.jpg")
            txt_filename = os.path.join(scene_folder, f"img{scene_number}.txt")
            print(f"*************txt_filename:{txt_filename}")
            real_count, pred_count, mae, mse = test__one_img_with_denoise(img_filename, transform, device, model, args ,txt_filename,scene_number)
                                      
            all_real_counts.append(real_count)
            all_pred_counts.append(pred_count)
            all_mae.append(mae)
            all_mse.append(mse)

    # محاسبه MAE و MSE میانگین
    print(f"len mae:{len(all_mae)}")
    print(f"len mse:{len(all_mse)}")
    avg_mae = np.mean(all_mae)
    avg_mse = np.sqrt(np.mean(all_mse))

    # محاسبه NAP
    nap = calculate_nap(all_pred_counts, all_real_counts)
    #  
    # avg_mae,avg_mse= eng.evaluate_crowd_no_overlap(model, data_loader, device, vis_dir=None)
    print(f'Mean Absolute Error (MAE): {avg_mae}')
    print(f'Mean Squared Error (MSE): {avg_mse}')
    print(f'Normalized Average Precision (NAP): {nap}')

if __name__ == '__main__':
    parser = argparse.ArgumentParser('P2PNet evaluation script', parents=[get_args_parser()])
    args = parser.parse_args()
    main(args)
