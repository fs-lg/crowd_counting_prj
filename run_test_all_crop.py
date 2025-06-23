import os
import torch
import numpy as np
from PIL import Image
import torchvision.transforms as standard_transforms
import matplotlib.pyplot as plt
from models import build_model
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
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

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

def split_image(img_path, output_dir):
    """کراپ کردن تصویر به ۴ بخش، با اطمینان از اینکه ابعاد زوج باشند"""
    if not os.path.exists(img_path):
        print(f"⚠ فایل یافت نشد: {img_path}")
        return None

    img = Image.open(img_path)
    width, height = img.size

    # اطمینان از عدد زوج بودن ابعاد کراپ
    mid_width = width // 2
    mid_height = height // 2

    if width % 2 != 0:
        mid_width += 1  # اگر عرض فرد بود، یک پیکسل اضافه کن

    if height % 2 != 0:
        mid_height += 1  # اگر ارتفاع فرد بود، یک پیکسل اضافه کن

    # انجام کراپ‌ها
    crops = {
        "top_left": img.crop((0, 0, mid_width, mid_height)),
        "top_right": img.crop((mid_width, 0, width, mid_height)),
        "bottom_left": img.crop((0, mid_height, mid_width, height)),
        "bottom_right": img.crop((mid_width, mid_height, width, height))
    }

    os.makedirs(output_dir, exist_ok=True)
    saved_paths = []
    for name, crop in crops.items():
        crop_path = os.path.join(output_dir, f"{os.path.basename(img_path).split('.')[0]}_{name}.jpg")
        crop.save(crop_path)
        saved_paths.append(crop_path)

    return saved_paths

def count_lines_in_txt(txt_path):
    """خواندن تعداد افراد از فایل متنی"""
    if not os.path.exists(txt_path):
        print(f"⚠ فایل TXT یافت نشد: {txt_path}")
        return None
    with open(txt_path, 'r') as f:
        return len([line for line in f.readlines() if line.strip()])

def test_one_img(img_path, transform, device, model, txt_path, output_dir):
    """اجرای مدل روی کراپ‌های یک تصویر (هر تصویر لیست جدا داره)"""
    crop_paths = split_image(img_path, output_dir)
    if crop_paths is None:
        return None

    real_count = count_lines_in_txt(txt_path)
    if real_count is None:
        return None

    predictions = []
    for crop_path in crop_paths:
        img_raw = Image.open(crop_path).convert('RGB')
        img_raw = img_raw.resize((img_raw.width // 128 * 128, img_raw.height // 128 * 128), Image.Resampling.LANCZOS)
        img = transform(img_raw)    
        samples = torch.Tensor(img).unsqueeze(0).to(device)
        outputs = model(samples)
        outputs_scores = torch.nn.functional.softmax(outputs['pred_logits'], -1)[:, :, 1][0]
        outputs_points = outputs['pred_points'][0]
        num_people = outputs['pred_points'].shape[1]
        threshold = 0.5
        points = outputs_points[outputs_scores > threshold].detach().cpu().numpy().tolist()
        predict_cnt = int((outputs_scores > threshold).sum())
        predictions.append(predict_cnt)
    total_pred = sum(predictions)
    mae = abs(real_count - total_pred)
    mse = (real_count - total_pred) ** 2

    return {"image": os.path.basename(img_path), "real": real_count, "predicted": total_pred, "MAE": mae, "MSE": mse}

def main(args):
    os.environ["CUDA_VISIBLE_DEVICES"] = '{}'.format(args.gpu_id)
    device = torch.device('cuda')
    model = build_model(args).to(device)
    if args.weight_path is not None:
        checkpoint = torch.load(args.weight_path, map_location='cpu')
        model.load_state_dict(checkpoint['model'])
    model.eval()
    transform = standard_transforms.Compose([
        standard_transforms.ToTensor(), 
        standard_transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    image_paths = [
        "/content/drive/MyDrive/CrowdCounting-P2PNet/DATA_ROOT/test/scene106/img106.jpg",
        "/content/drive/MyDrive/CrowdCounting-P2PNet/DATA_ROOT/test/scene1/img01.jpg",
        "/content/drive/MyDrive/CrowdCounting-P2PNet/DATA_ROOT/test/scene10/img10.jpg",
        "/content/drive/MyDrive/CrowdCounting-P2PNet/DATA_ROOT/test/scene122/img122.jpg",
        "/content/drive/MyDrive/CrowdCounting-P2PNet/DATA_ROOT/test/scene112/img112.jpg"
    ]

    txt_paths = [
        "/content/drive/MyDrive/CrowdCounting-P2PNet/DATA_ROOT/test/scene106/img106.txt",
        "/content/drive/MyDrive/CrowdCounting-P2PNet/DATA_ROOT/test/scene1/img01.txt",
        "/content/drive/MyDrive/CrowdCounting-P2PNet/DATA_ROOT/test/scene10/img10.txt",
        "/content/drive/MyDrive/CrowdCounting-P2PNet/DATA_ROOT/test/scene122/img122.txt",
        "/content/drive/MyDrive/CrowdCounting-P2PNet/DATA_ROOT/test/scene112/img112.txt"
    ]

    all_results = []
    
    print("\n📊 **نتایج بررسی تأثیر کراپ روی شمارش افراد**")
    print("=" * 70)

    for img_path, txt_path in zip(image_paths, txt_paths):
        result = test_one_img(img_path, transform, device, model, txt_path, args.output_dir)
        if result is None:
            continue
        
        all_results.append(result)

        print(f"\n📌 **تصویر:** {result['image']}")
        print("--------------------------------------------------")
        print(f"{'Real':<10}{'Predicted':<12}{'MAE':<10}{'MSE'}")
        print("-" * 50)
        print(f"{result['real']:<10}{result['predicted']:<12}{result['MAE']:<10}{result['MSE']}")

    if all_results:
        avg_mae = np.mean([r["MAE"] for r in all_results])
        avg_mse = np.mean([r["MSE"] for r in all_results])
        print("\n📢 **میانگین کلی خطاها روی تمام تصاویر:**")
        print(f"🔹 MAE (میانگین خطای مطلق): {avg_mae:.2f}")
        print(f"🔹 MSE (میانگین مربعات خطا): {avg_mse:.2f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser('P2PNet evaluation script', parents=[get_args_parser()])
    args = parser.parse_args()
    main(args)

