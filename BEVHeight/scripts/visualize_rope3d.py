import json
import cv2
import numpy as np
import os
from pyquaternion import Quaternion
from scripts.gen_info_rope3d import load_calib, get_cam2lidar

def draw_projected_box3d(image, corners, color=(255, 255, 0), thickness=2):
    """ 在图像上绘制3D边界框的投影 """
    corners = corners.astype(np.int32)
    
    # 绘制底部矩形
    for i in range(4):
        j = (i + 1) % 4
        cv2.line(image, (corners[i, 0], corners[i, 1]),
                (corners[j, 0], corners[j, 1]), color, thickness)
    
    # 绘制顶部矩形
    for i in range(4):
        j = (i + 1) % 4
        cv2.line(image, (corners[i + 4, 0], corners[i + 4, 1]),
                (corners[j + 4, 0], corners[j + 4, 1]), color, thickness)
    
    # 绘制连接线
    for i in range(4):
        cv2.line(image, (corners[i, 0], corners[i, 1]),
                (corners[i + 4, 0], corners[i + 4, 1]), color, thickness)
    
    return image

def compute_box_3d(dim, location, rotation):
    """ 计算3D边界框的8个角点 """
    # 创建3D边界框的8个角点
    h, w, l = dim
    x_corners = [l/2, l/2, -l/2, -l/2, l/2, l/2, -l/2, -l/2]
    y_corners = [w/2, -w/2, -w/2, w/2, w/2, -w/2, -w/2, w/2]
    z_corners = [0, 0, 0, 0, h, h, h, h]
    
    # 旋转和平移
    corners_3d = np.vstack([x_corners, y_corners, z_corners])
    rot_mat = rotation.rotation_matrix
    corners_3d = np.dot(rot_mat, corners_3d)
    corners_3d[0, :] += location[0]
    corners_3d[1, :] += location[1]
    corners_3d[2, :] += location[2]
    
    return corners_3d.T

def visualize_results(results_file, rope3d_root, output_dir, num_vis=10):
    """可视化检测结果"""
    # 加载检测结果
    print(f"Loading results from {results_file}")
    with open(results_file, 'r') as f:
        data = json.load(f)
    
    # 获取results部分
    results = data['results']
    print(f"Found {len(results)} samples in results")
    
    os.makedirs(output_dir, exist_ok=True)
    print(f"Created output directory: {output_dir}")
    
    # 只处理前num_vis个样本
    processed_count = 0
    for sample_token, detections in results.items():
        if processed_count >= num_vis:
            break
            
        print(f"\nProcessing sample: {sample_token}")
        print(f"Number of detections: {len(detections)}")
        
        # 找到对应的图像文件
        img_found = False
        for img_dir in ["training-image_2a", "training-image_2b", "training-image_2c", "training-image_2d", "validation-image_2"]:
            # 使用完整的文件名
            img_path = os.path.join(rope3d_root, img_dir, f"{sample_token}.jpg")
            print(f"Checking image path: {img_path}")
            if os.path.exists(img_path):
                img_found = True
                print(f"Found image at: {img_path}")
                break
        
        if not img_found:
            print("Image not found, skipping...")
            continue
            
        # 读取图像和标定信息
        image = cv2.imread(img_path)
        # 使用完整的sample_token作为标定文件名
        calib_file = os.path.join(rope3d_root, "training/calib", f"{sample_token}.txt")
        denorm_file = os.path.join(rope3d_root, "training/denorm", f"{sample_token}.txt")
        
        print(f"Checking calibration files:")
        print(f"Calib file: {calib_file}")
        print(f"Denorm file: {denorm_file}")
        
        if not (os.path.exists(calib_file) and os.path.exists(denorm_file)):
            calib_file = os.path.join(rope3d_root, "validation/calib", f"{sample_token}.txt")
            denorm_file = os.path.join(rope3d_root, "validation/denorm", f"{sample_token}.txt")
            print(f"Trying validation files:")
            print(f"Calib file: {calib_file}")
            print(f"Denorm file: {denorm_file}")
        
        if not (os.path.exists(calib_file) and os.path.exists(denorm_file)):
            print("Calibration files not found, skipping...")
            continue
            
        camera_intrinsic = load_calib(calib_file)
        _, _, Tr_cam2lidar, _ = get_cam2lidar(denorm_file)
        Tr_lidar2cam = np.linalg.inv(Tr_cam2lidar)
        
        # 绘制每个检测结果
        for det in detections:
            translation = det['translation']
            size = det['size']
            rotation = det['rotation']
            if isinstance(rotation, list):  # 如果rotation是四元数列表
                rotation = Quaternion(rotation)
            score = det['detection_score']
            
            # 计算3D边界框角点
            corners_3d = compute_box_3d(size, translation, rotation)
            
            # 将点从雷达坐标系转换到相机坐标系
            corners_3d_hom = np.concatenate([corners_3d, np.ones((8, 1))], axis=1)
            corners_3d_cam = (Tr_lidar2cam @ corners_3d_hom.T).T
            
            # 检查是否在相机前方
            if np.any(corners_3d_cam[:, 2] < 0):
                continue
            
            # 投影到图像平面
            corners_2d = (camera_intrinsic @ corners_3d_cam[:, :3].T).T
            corners_2d = corners_2d[:, :2] / corners_2d[:, 2:3]
            
            # 检查是否在图像范围内
            if np.any(corners_2d[:, 0] < 0) or np.any(corners_2d[:, 0] >= image.shape[1]) or \
               np.any(corners_2d[:, 1] < 0) or np.any(corners_2d[:, 1] >= image.shape[0]):
                continue
            
            # 绘制边界框
            color = (0, 255, 0) if score > 0.5 else (0, 165, 255)
            image = draw_projected_box3d(image, corners_2d, color=color)
            
            # 添加得分标签和类别
            label = f"{det.get('detection_name', '')} {score:.2f}"
            cv2.putText(image, label, (int(corners_2d[0, 0]), int(corners_2d[0, 1])),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
        
        # 保存结果
        output_path = os.path.join(output_dir, f"{sample_token}.jpg")
        cv2.imwrite(output_path, image)
        processed_count += 1
        print(f"Saved visualization to {output_path}")

if __name__ == '__main__':
    rope3d_root = "data/rope3d"
    results_file = "outputs/bev_height_lss_r50_864_1536_128x128/results_nusc.json"
    output_dir = "outputs/visualization"
    
    print(f"\nStarting visualization with parameters:")
    print(f"rope3d_root: {rope3d_root}")
    print(f"results_file: {results_file}")
    print(f"output_dir: {output_dir}")
    
    # 检查文件是否存在
    if not os.path.exists(results_file):
        print(f"Error: Results file not found: {results_file}")
    else:
        visualize_results(results_file, rope3d_root, output_dir, num_vis=10) 