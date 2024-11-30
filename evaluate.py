import torch
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
from model import ImprovedUNet
from data_loader import get_patient_wise_loaders

def calculate_metrics(pred, target):
    pred = (pred > 0.5).float()
    target = (target > 0.5).float()
    
    # IoU 계산
    intersection = (pred * target).sum()
    union = (pred + target).clamp(0, 1).sum()
    iou = (intersection + 1e-7) / (union + 1e-7)
    
    # Dice 계산
    dice = (2 * intersection + 1e-7) / (pred.sum() + target.sum() + 1e-7)
    
    # Precision과 Recall 계산
    true_positives = (pred * target).sum()
    predicted_positives = pred.sum()
    actual_positives = target.sum()
    
    precision = (true_positives + 1e-7) / (predicted_positives + 1e-7)
    recall = (true_positives + 1e-7) / (actual_positives + 1e-7)
    
    return {
        'iou': iou.item(),
        'dice': dice.item(),
        'precision': precision.item(),
        'recall': recall.item()
    }

def evaluate_model(model_path, test_data_path):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # 모델 로드
    model = ImprovedUNet().to(device)
    model.load_state_dict(torch.load(model_path))
    model.eval()
    
    # 테스트 데이터 로더 생성
    test_loaders = get_patient_wise_loaders(test_data_path, batch_size=1)
    
    # 평가 지표 초기화
    total_metrics = {'iou': 0, 'dice': 0, 'precision': 0, 'recall': 0}
    total_samples = 0
    
    # 환자별 메트릭 저장
    patient_metrics = {}
    
    with torch.no_grad():
        for test_loader in test_loaders:
            patient_id = test_loader.dataset.images[0].split('_')[0:4]
            patient_id = '_'.join(patient_id)
            
            patient_metrics[patient_id] = {
                'iou': [], 'dice': [], 'precision': [], 'recall': []
            }
            
            for images, masks in tqdm(test_loader, desc=f'Evaluating {patient_id}'):
                images = images.to(device)
                masks = masks.to(device)
                
                predictions = model(images)
                metrics = calculate_metrics(predictions, masks)
                
                # 환자별 메트릭 저장
                for key in metrics:
                    patient_metrics[patient_id][key].append(metrics[key])
                    total_metrics[key] += metrics[key]
                
                total_samples += 1
    
    # 평균 메트릭 계산
    avg_metrics = {k: v/total_samples for k, v in total_metrics.items()}
    
    # 환자별 평균 계산
    for patient_id in patient_metrics:
        patient_metrics[patient_id] = {
            k: np.mean(v) for k, v in patient_metrics[patient_id].items()
        }
    
    # 결과 출력
    print("\n=== 전체 평가 결과 ===")
    print(f"평균 IoU: {avg_metrics['iou']:.4f}")
    print(f"평균 Dice: {avg_metrics['dice']:.4f}")
    print(f"평균 Precision: {avg_metrics['precision']:.4f}")
    print(f"평균 Recall: {avg_metrics['recall']:.4f}")
    
    print("\n=== 환자별 평가 결과 ===")
    for patient_id, metrics in patient_metrics.items():
        print(f"\n환자 ID: {patient_id}")
        print(f"IoU: {metrics['iou']:.4f}")
        print(f"Dice: {metrics['dice']:.4f}")
        print(f"Precision: {metrics['precision']:.4f}")
        print(f"Recall: {metrics['recall']:.4f}")
    
    return avg_metrics, patient_metrics

if __name__ == '__main__':
    model_path = 'best_model.pth'
    test_data_path = 'data/test'
    evaluate_model(model_path, test_data_path)