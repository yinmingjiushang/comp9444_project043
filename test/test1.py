# test_stytr2.py
import torch
from PIL import Image
import os

# 假设你在 StyTR-2 的 models/ 中有一个 StyTr2Model 类或类似文件
# 这里的 import 具体以实际文件结构为准
import stytr2
from utils.utils import load_image, save_image, transform_image  # 示例


def main():
    # 1. 准备模型
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = StyTr2Model().to(device)

    # 如果有预训练权重，先加载
    # model.load_state_dict(torch.load("checkpoints/stytr2_pretrained.pth", map_location=device))
    # model.eval()

    # 2. 准备输入图像：1张内容图 + 1张风格图
    content_path = "test_images/content.jpg"  # 你的内容图片路径
    style_path = "test_images/style.jpg"  # 你的风格图片路径
    content_img = load_image(content_path)  # 返回的是一个 PIL.Image 或 Tensor
    style_img = load_image(style_path)

    # 图像预处理（如 resize, to Tensor, normalization）
    content_tensor = transform_image(content_img).unsqueeze(0).to(device)
    style_tensor = transform_image(style_img).unsqueeze(0).to(device)

    # 3. 推理
    with torch.no_grad():
        output_tensor = model(content_tensor, style_tensor)
        # output_tensor 形状一般是 [B, C, H, W]

    # 4. 保存结果
    output_image = output_tensor.squeeze(0).cpu()
    save_image(output_image, "test_images/output_stylized.jpg")
    print("风格迁移完成，结果已保存至 test_images/output_stylized.jpg")


if __name__ == "__main__":
    main()
