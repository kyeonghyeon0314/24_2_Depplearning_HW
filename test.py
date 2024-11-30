import os
import numpy as np
from PIL import Image
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, Sampler
from torchvision import transforms, models
from tqdm import tqdm
import random
from model import ImprovedUNet, CombinedLoss


class Predictor:
    def __init__(self, model_path, device='cuda'):
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        self.model = ImprovedUNet(pretrained=False).to(self.device)
    
        # weights_only=True로 설정하여 안전하게 로드
        state_dict = torch.load(model_path, map_location=self.device, weights_only=True)
    
        # 새로운 state_dict 생성
        new_state_dict = {}
        for key, value in state_dict.items():
            if 'decoder' in key and 'weight' in key:
                if value.size(0) != self.model.state_dict()[key].size(0):
                    continue
            elif key == 'final.1.weight' or key == 'final.1.bias':
                continue
            new_state_dict[key] = value
    
        # 수정된 state_dict로 모델 로드
        self.model.load_state_dict(new_state_dict, strict=False)
        self.model.eval()

    def predict_batch(self, test_dir, output_dir):
        os.makedirs(output_dir, exist_ok=True)
        test_files = [f for f in os.listdir(test_dir) if f.endswith('.tif')]
        
        print(f"총 {len(test_files)}개 이미지 예측 시작...")
        
        with tqdm(test_files) as pbar:
            for file in pbar:
                try:
                    input_path = os.path.join(test_dir, file)
                    output_path = os.path.join(output_dir, file.replace('.tif', '_mask.tif'))
                    
                    # 이미지 로드 및 전처리
                    image = Image.open(input_path).convert('L')
                    image = image.resize((256, 256))
                    image_np = np.array(image)
                    input_tensor = torch.from_numpy(image_np).float().unsqueeze(0).unsqueeze(0) / 255.0
                    input_tensor = input_tensor.to(self.device)
                    
                    # 예측
                    with torch.no_grad():
                        output = self.model(input_tensor)
                    
                    # 후처리 및 저장
                    pred_mask = output.squeeze().cpu().numpy()
                    pred_mask = (pred_mask > 0.5).astype(np.uint8) * 255
                    pred_mask_img = Image.fromarray(pred_mask)
                    pred_mask_img.save(output_path)
                    
                    pbar.set_description(f"처리 중: {file}")
                    
                except Exception as e:
                    print(f"\nError processing {file}: {str(e)}")
                    continue
        
        print(f"\n예측 완료: {output_dir} 디렉토리에 저장되었습니다.")
        
    def preprocess_image(self, image):
        # data_loader.py와 동일한 전처리 적용
        image_np = np.array(image)
        image_tensor = torch.from_numpy(image_np).float().unsqueeze(0) / 255.0
        return image_tensor.unsqueeze(0)
        
    def postprocess_mask(self, mask):
        mask = mask.squeeze().cpu().numpy()
        mask = (mask > 0.5).astype(np.uint8) * 255
        return Image.fromarray(mask)
    
    def dice_coefficient(pred, target):
        smooth = 1e-5
        pred = pred.float().view(-1)
        target = target.float().view(-1)
        intersection = (pred * target).sum()
        return (2. * intersection + smooth) / (pred.sum() + target.sum() + smooth)

    


def main():
    # 설정
    MODEL_PATH = 'models/best_model.pth'
    TEST_DIR = 'data/test'
    OUTPUT_DIR = 'data/predictions'
    
    # 예측기 초기화 및 실행
    try:
        predictor = Predictor(
            model_path=MODEL_PATH,
            device='cuda'
        )
        results = predictor.predict_batch(TEST_DIR, OUTPUT_DIR)
        print("예측이 완료되었습니다!")
        
    except Exception as e:
        print(f"예측 중 오류 발생: {str(e)}")

if __name__ == '__main__':
    main()
