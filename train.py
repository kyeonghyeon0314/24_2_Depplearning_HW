import os
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
from model import ImprovedUNet, CombinedLoss
from data_loader import PatientWiseDataset, get_transforms
from config import *

def train_model():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # 데이터 분할
    all_files = [f for f in os.listdir(TRAIN_PATH) if not f.endswith('_mask.tif')]
    train_files = all_files[:int(len(all_files)*0.8)]
    val_files = all_files[int(len(all_files)*0.8):]
    
    # 데이터셋 및 로더 설정
    train_dataset = PatientWiseDataset(
        TRAIN_PATH,
        train_files,
        transform=get_transforms(is_train=True)
    )
    
    val_dataset = PatientWiseDataset(
        TRAIN_PATH,
        val_files,
        transform=get_transforms(is_train=False)
    )
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=NUM_WORKERS,
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=True
    )
    
    # 모델 설정
    model = ImprovedUNet().to(device)
    criterion = CombinedLoss(
        alpha=LOSS_WEIGHTS['dice'],
        beta=LOSS_WEIGHTS['focal']
    )
    
    optimizer = optim.AdamW(
        model.parameters(),
        lr=LEARNING_RATE,
        weight_decay=0.01
    )
    
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode='min',
        factor=0.5,
        patience=5,
        verbose=True
    )
    
    # 학습 루프
    best_val_loss = float('inf')
    patience_counter = 0
    
    for epoch in range(NUM_EPOCHS):
        # 학습
        model.train()
        train_loss = 0
        
        for images, masks in tqdm(train_loader, desc=f'Epoch {epoch+1}/{NUM_EPOCHS}'):
            images = images.to(device)
            masks = masks.to(device)
            
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, masks)
            
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            
            train_loss += loss.item()
        
        avg_train_loss = train_loss / len(train_loader)
        
        # 검증
        model.eval()
        val_loss = 0
        
        with torch.no_grad():
            for val_images, val_masks in tqdm(val_loader, desc='Validation'):
                val_images = val_images.to(device)
                val_masks = val_masks.to(device)
                
                val_outputs = model(val_images)
                val_loss += criterion(val_outputs, val_masks).item()
        
        avg_val_loss = val_loss / len(val_loader)
        
        # 학습률 조정
        scheduler.step(avg_val_loss)
        
        # 현재 epoch의 결과 출력
        print(f'Epoch {epoch+1}/{NUM_EPOCHS}:')
        print(f'Train Loss: {avg_train_loss:.4f}')
        print(f'Val Loss: {avg_val_loss:.4f}')
        print(f'Learning Rate: {optimizer.param_groups[0]["lr"]:.6f}\n')
        
        # 모델 저장
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            patience_counter = 0
            torch.save(model.state_dict(),
                      os.path.join(MODEL_SAVE_PATH, 'best_model.pth'))
        else:
            patience_counter += 1
            
        if patience_counter >= 10:
            print("Early stopping triggered")
            break

if __name__ == '__main__':
    os.makedirs(MODEL_SAVE_PATH, exist_ok=True)
    train_model()