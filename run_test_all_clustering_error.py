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

def apply_denoise_filter(img_raw, filter_type):
    img_np = np.array(img_raw)
    if filter_type == 'gaussian':
        img_denoised = cv2.GaussianBlur(img_np, (5, 5), 0)
    elif filter_type == 'median':
        img_denoised = cv2.medianBlur(img_np, 5)
    elif filter_type == 'bilateral':
        img_denoised = cv2.bilateralFilter(img_np, 7, 50, 50)
    else:
        img_denoised = img_np
    img_denoised = Image.fromarray(img_denoised)
    return img_denoised

def count_lines_in_txt(txt_path):
    with open(txt_path, 'r') as f:
        lines = f.readlines()
        return len([line for line in lines if line.strip()])

def test_one_img(img_path, transform, device, model, args, txt_path, scene_number):
    if not os.path.exists(img_path):
        print(f"File not found: {img_path}")
        return 0, 0, 0, 0
    img_raw = Image.open(img_path).convert('RGB')
    img_raw = img_raw.resize((img_raw.width // 128 * 128, img_raw.height // 128 * 128), Image.Resampling.LANCZOS)
    img_raw = apply_denoise_filter(img_raw, '')  # استفاده از فیلتر جدید*************
    img = transform(img_raw)
    
    samples = torch.Tensor(img).unsqueeze(0).to(device)
    outputs = model(samples)
    outputs_scores = torch.nn.functional.softmax(outputs['pred_logits'], -1)[:, :, 1][0]
    outputs_points = outputs['pred_points'][0]
    num_people = outputs['pred_points'].shape[1]
    threshold = 0.5
    points = outputs_points[outputs_scores > threshold].detach().cpu().numpy().tolist()
    predict_cnt = int((outputs_scores > threshold).sum())
    num_lines = count_lines_in_txt(txt_path)
    mae = abs(num_lines - predict_cnt)
    mse = (num_lines - predict_cnt) ** 2
    #######################دید درصد به خطا
    if num_lines != 0:  # جلوگیری از تقسیم بر صفر
        mape = abs((num_lines - predict_cnt) / num_lines) * 100
        mspe = ((num_lines - predict_cnt) / num_lines) ** 2 * 100
    else:
        mape = 0
        mspe = 0

    return num_lines, predict_cnt, mae, mse , mape , mspe

def plot_error_distribution(error_dict, typeIs):
    output_path = f"/content/drive/MyDrive/CrowdCounting-P2PNet/Output distribution/distrib_{typeIs}.png"
    errors = list(error_dict.values())
    plt.figure(figsize=(8, 6))
    plt.hist(errors, bins=np.linspace(min(errors), max(errors), 20), color='blue', alpha=0.7)
    plt.xlabel('Error (MSE)')
    plt.ylabel('Frequency')
    plt.title(f'Distribution of Prediction Errors - {typeIs}')
    plt.grid(True)
    plt.savefig(output_path)
    plt.show()
    
def cluster_images_by_error(error_dict, n_clusters):
    error_values = np.array(list(error_dict.values())).reshape(-1, 1)
    
    # اجرای خوشه‌بندی KMeans
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    labels = kmeans.fit_predict(error_values)

    clusters = {i: [] for i in range(n_clusters)}
    mse_averages = {i: [] for i in range(n_clusters)}

    for img, label, error in zip(error_dict.keys(), labels, error_values.flatten()):
        clusters[label].append(img)
        mse_averages[label].append(error)
 
    # محاسبه میانگین خطاهای MSE و MAE برای هر کلاستر
    mse_averages = {i: np.mean(mse_averages[i]) for i in mse_averages}

    return clusters, mse_averages


# import matplotlib.pyplot as plt
# from adjustText import adjust_text  # وارد کردن کتابخانه adjustText

from adjustText import adjust_text
import matplotlib.pyplot as plt

def plot_clusters_pie_chart(clusters, mse_averages):
    sizes = [len(images) for images in clusters.values()]
    total = sum(sizes)
    percentages = [size / total * 100 for size in sizes]
    labels = [f'Cluster {i+1}\n{percentages[i]:.1f}%' for i in range(len(clusters))]
    
    fig, ax = plt.subplots(figsize=(8, 8))
    
    # 📌 رسم نمودار دایره‌ای
    wedges, texts, autotexts = ax.pie(
        sizes, labels=labels, autopct='%1.1f%%', startangle=90, 
        colors=['lightblue', 'lightgreen', 'lightcoral', 'gold', 'purple']
    )
    
    plt.title('Clustered Images by Error')
    
    # 📌 ذخیره اطلاعات مربوط به کلاسترهای زیر ۶۰٪ برای نمایش جداگانه
    cluster_info_texts = []
    
    for i, (cluster_id, percentage) in enumerate(zip(clusters.keys(), percentages)):
        if percentage < 60:
            image_list = ', '.join(clusters[cluster_id])
            mse_val = mse_averages[cluster_id]
            cluster_info_texts.append(f"📌 **Cluster {i+1}** - MSE: {mse_val:.2f}\n   Images: {image_list}")

    # 📌 نمایش اطلاعات کلاسترهای کوچک‌تر در یک باکس جداگانه
    if cluster_info_texts:
        info_text = "\n\n".join(cluster_info_texts)
        plt.figtext(0.5, -0.1, info_text, ha="center", fontsize=10, bbox=dict(boxstyle="round", facecolor="white", alpha=0.5))

    plt.show()

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
    
    scene_dir = "/content/drive/MyDrive/CrowdCounting-P2PNet/DATA_ROOT/test"
    scene_folders = sorted([os.path.join(scene_dir, folder) for folder in os.listdir(scene_dir) if os.path.isdir(os.path.join(scene_dir, folder))])
    
    error_dict = {}
    error_dict_for_plot = {}
    
    # 🔹 اضافه کردن لیست‌ها برای میانگین‌گیری
    mae_list = []
    mse_list = []
    mape_list = []
    mspe_list = []
    
    for scene_folder in scene_folders:
        scene_name = os.path.basename(scene_folder)
        match = re.search(r'(\d+)$', scene_name)
        if match:
            scene_number = match.group(1).zfill(2)
            # if scene_number == '101' or scene_number == '01':
            #     print(f"Skipping scene number {scene_number}")
            #     continue
            
            img_filename = os.path.join(scene_folder, f"img{scene_number}.jpg")
            txt_filename = os.path.join(scene_folder, f"img{scene_number}.txt")
            
            real_count, pred_count, mae, mse, mape, mspe = test_one_img(img_filename, transform, device, model, args, txt_filename, scene_number)
            
            # 🔹 ذخیره مقادیر برای محاسبه میانگین
            error_dict[img_filename] = mse
            error_dict_for_plot[f"img{scene_number}"] = mse
            mae_list.append(mae)
            mse_list.append(mse)
            mape_list.append(mape)
            mspe_list.append(mspe)
            if scene_number=='106':
              print(f"***mse of 106:{mse} ,mae:{mae} ,mape:{mape},mspe:{mspe}")
    
    # 🔹 محاسبه میانگین خطاها
    avg_mae = np.mean(mae_list)
    avg_mse = np.sqrt(np.mean(mse_list))  # محاسبه MSE به‌صورت RMSE
    avg_mape = np.mean(mape_list)
    avg_mspe = np.mean(mspe_list)
    
    # 🔹 نمایش نتایج خلاصه‌شده
    print("\n📢 **میانگین خطاهای محاسبه‌شده:**")
    print(f"🔹 MAE  (میانگین خطای مطلق)   : {avg_mae:.2f}")
    print(f"🔹 MSE  (میانگین مربعات خطا)  : {avg_mse:.2f}")
    print(f"🔹 MAPE (درصد خطای مطلق)      : {avg_mape:.2f}%")
    print(f"🔹 MSPE (درصد خطای مربعات)    : {avg_mspe:.2f}%")

    # 🔹 خوشه‌بندی خطاها و نمایش نمودارها
# خوشه‌بندی خطاها و نمایش نمودارها
#مشخص کردن کلاستر ها ****************************
    clusters, mse_averages= cluster_images_by_error(error_dict_for_plot, n_clusters=5) 

    print("Clustered Images by Error:", clusters)
    plot_error_distribution(error_dict_for_plot, typeIs="bilateral")
    plot_clusters_pie_chart(clusters, mse_averages)  

if __name__ == '__main__':
    parser = argparse.ArgumentParser('P2PNet evaluation script', parents=[get_args_parser()])
    args = parser.parse_args()
    main(args)

