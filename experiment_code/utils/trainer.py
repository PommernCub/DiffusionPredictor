# -*- coding: utf-8 -*-
"""
Created on Wed May 21 16:27:44 2025

@author: 91278
"""
import torch
import torch.nn.functional as F
from tqdm import tqdm


def setup_device():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    return device


def criterion(outputs, targets):
    main_pred, aux_pred = outputs
    seq_target = targets.float().mean(dim=-1)  # 序列级标签
    loss1 = F.cross_entropy(main_pred, seq_target.long())  # 判断序列是否有吸附
    loss2 = F.binary_cross_entropy_with_logits(aux_pred, targets.float())  # 点级别吸附损失
    return 0.7*loss1 + 0.3*loss2  # 主次损失加权


def create_optimizer(model, lr=1e-2):
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=lr,
        weight_decay=1e-4,  # 使用较大的学习率衰减，用于正则化
    )

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=5)
    return optimizer, scheduler


def train_epoch(epochs, model, train_loader, optimizer, criterion, device):
    model.train()
    total_loss = 0.0
    # 进度条
    progress_bar = tqdm(train_loader, desc=f'Epoch {epochs[0]+1}/{epochs[1]}')
    
    for inputs, labels in progress_bar:
        inputs, labels = inputs.to(device), labels.squeeze().to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()

        torch.nn.utils.clip_grad_norm_(
            model.parameters(),1.0,  # 降低到0.5更严格限制梯度爆炸
        )
        optimizer.step()
        total_loss += loss.item() * inputs.size(0)
    return total_loss / len(train_loader.dataset)


def validate(epochs, model, valid_loader, criterion, device):
    model.eval()
    val_loss = 0.0
    correct = 0
    total = 0
    with torch.no_grad():
        progress_bar = tqdm(valid_loader, desc=f'Epoch {epochs[0]+1}/{epochs[1]}')        
        for inputs, labels in progress_bar:
            inputs, labels = inputs.to(device), labels.squeeze().to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            val_loss += loss.item() * inputs.size(0)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.numel()
            correct += (predicted == labels).sum().item()
    return val_loss / len(valid_loader.dataset), 100 * correct / total
