import torch
import os
import random
import numpy as np
import torch.nn.functional as F
import torch.nn as nn

def set_seed(seed: int):
    """
    Thiết lập seed cho random, numpy, và torch để đảm bảo kết quả có thể tái tạo.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed) 
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def train_model(model, train_loader, val_loader, criterion, optimizer, output_name,
                scheduler=None, num_epochs=100, device='cpu', 
                early_stopping_patience=40,
                augmentation=None):
    
    model.to(device)
    best_val_acc = 0.0
    patience_counter = 0
    
    history = {'train_loss': [], 'train_acc': [], 'val_loss': [], 'val_acc': []}
    
    try:
        for epoch in range(num_epochs):
            # Training phase
            model.train()
            running_loss = 0
            correct_train = 0
            total_train = 0  
            
            for batch_idx, (inputs, labels) in enumerate(train_loader):
                try:
                    inputs, labels = inputs.to(device), labels.to(device)
                    if augmentation:
                        inputs = augmentation(inputs)
                    optimizer.zero_grad()
                    outputs = model(inputs)
                    loss = criterion(outputs, labels)
                    loss.backward()
                    optimizer.step()
                    
                    running_loss += loss.item() * inputs.size(0)
                    _, predicted = torch.max(outputs.data, 1)
                    total_train += labels.size(0)
                    correct_train += (predicted == labels).sum().item()
                    
                except RuntimeError as e:
                    if "out of memory" in str(e):
                        print(f"GPU OOM at epoch {epoch+1}, batch {batch_idx}")
                        torch.cuda.empty_cache()
                        continue
                    else:
                        raise e
            
            # Tính metrics training
            if total_train > 0:
                epoch_loss = running_loss / total_train
                epoch_acc_train = 100.0 * correct_train / total_train
            else:
                epoch_loss = 0
                epoch_acc_train = 0
            
            history['train_loss'].append(epoch_loss)
            history['train_acc'].append(epoch_acc_train)
            
            # Validation phase
            model.eval()
            val_loss = 0
            correct_val = 0
            total_val = 0
            
            with torch.no_grad():
                for inputs, labels in val_loader:
                    inputs, labels = inputs.to(device), labels.to(device)
                    outputs = model(inputs)
                    
                    loss = criterion(outputs, labels)
                    val_loss += loss.item() * inputs.size(0) 
                    
                    _, predicted = torch.max(outputs.data, 1)
                    total_val += labels.size(0)
                    correct_val += (predicted == labels).sum().item()
            
            # Tính metrics validation
            if total_val > 0:
                epoch_val_loss = val_loss / total_val
                epoch_acc_val = 100.0 * correct_val / total_val
            else:
                epoch_val_loss = 0
                epoch_acc_val = 0
            
            history['val_loss'].append(epoch_val_loss)
            history['val_acc'].append(epoch_acc_val)
            
            # print(f"Epoch {epoch+1}/{num_epochs} | "
            #       f"Train Loss: {epoch_loss:.4f} Acc: {epoch_acc_train:.2f}% | "
            #       f"Val Loss: {epoch_val_loss:.4f} Acc: {epoch_acc_val:.2f}%")
            
            # Save best model
            if epoch_acc_val > best_val_acc:
                best_val_acc = epoch_acc_val
                patience_counter = 0
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'best_val_acc': best_val_acc,
                    'history': history
                }, os.path.join('./outputs/models', output_name))
            else:
                patience_counter += 1
            
            # Early stopping
            if patience_counter >= early_stopping_patience:
                print(f"Early stopping at epoch {epoch+1}")
                break
            
            # Scheduler step
            if scheduler is not None:
                if hasattr(scheduler, 'step'):
                    # Xử lý các loại scheduler khác nhau
                    if 'ReduceLROnPlateau' in str(type(scheduler)):
                        scheduler.step(epoch_val_loss)
                    else:
                        scheduler.step()
            
            # Memory cleanup
            if device == 'cuda':
                torch.cuda.empty_cache()
    
    except KeyboardInterrupt:
        print("Training interrupted by user")
    except Exception as e:
        print(f"Training failed with error: {e}")
        raise e
    
    return model, history

def distillation_loss(student_outputs, teacher_outputs, labels, temperature, alpha):
    """
    Tính toán hàm mất mát kết hợp cho knowledge distillation.
    
    Args:
        student_outputs (torch.Tensor): Logits từ student model.
        teacher_outputs (torch.Tensor): Logits từ teacher model.
        labels (torch.Tensor): Nhãn thật.
        temperature (float): Nhiệt độ để làm mềm.
        alpha (float): Hệ số cân bằng.

    Returns:
        torch.Tensor: Giá trị loss tổng hợp.
    """
    # 1. Hard Loss (với nhãn thật)
    loss_hard = F.cross_entropy(student_outputs, labels)

    # 2. Soft Loss (với đầu ra của teacher)
    # Sử dụng log_softmax cho đầu ra của student và softmax cho đầu ra của teacher
    # KLDivLoss mong đợi đầu vào là log-probabilities và mục tiêu là probabilities
    loss_soft = nn.KLDivLoss(reduction='batchmean')(
        F.log_softmax(student_outputs / temperature, dim=1),
        F.softmax(teacher_outputs / temperature, dim=1)
    ) * (temperature * temperature) # Nhân lại với T^2 để cân bằng gradient

    # 3. Kết hợp hai loss
    combined_loss = alpha * loss_hard + (1 - alpha) * loss_soft
    return combined_loss

def train_distillation(teacher_model, student_model, train_loader, val_loader, 
                       optimizer, temperature, alpha,
                       scheduler=None, num_epochs=100, device='cpu', 
                       early_stopping_patience=20):
    
    # 1. Chuẩn bị model
    teacher_model.to(device)
    student_model.to(device)
    
    # Rất quan trọng: Đưa teacher model về chế độ eval()
    # Nó chỉ dùng để dự đoán, không cập nhật trọng số và không dùng dropout/batchnorm ở chế độ train
    teacher_model.eval()

    best_val_acc = 0.0
    patience_counter = 0
    
    history = {'train_loss': [], 'train_acc': [], 'val_loss': [], 'val_acc': []}
    
    # Dùng CrossEntropyLoss cho phần validation
    validation_criterion = nn.CrossEntropyLoss()

    try:
        for epoch in range(num_epochs):
            # --- Training phase ---
            student_model.train() # Chỉ student model ở chế độ train
            running_loss = 0
            correct_train = 0
            total_train = 0  
            
            for inputs, labels in train_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                
                # Lấy output từ teacher (không cần tính gradient)
                with torch.no_grad():
                    teacher_outputs = teacher_model(inputs)
                
                # Lấy output từ student và tính loss
                optimizer.zero_grad()
                student_outputs = student_model(inputs)
                
                # Sử dụng hàm loss distillation
                loss = distillation_loss(student_outputs, teacher_outputs, labels, temperature, alpha)
                
                loss.backward()
                optimizer.step()
                
                running_loss += loss.item() * inputs.size(0)
                _, predicted = torch.max(student_outputs.data, 1)
                total_train += labels.size(0)
                correct_train += (predicted == labels).sum().item()
            
            epoch_loss = running_loss / total_train
            epoch_acc_train = 100.0 * correct_train / total_train
            history['train_loss'].append(epoch_loss)
            history['train_acc'].append(epoch_acc_train)
            
            # --- Validation phase (chỉ đánh giá student model) ---
            student_model.eval()
            val_loss = 0
            correct_val = 0
            total_val = 0
            
            with torch.no_grad():
                for inputs, labels in val_loader:
                    inputs, labels = inputs.to(device), labels.to(device)
                    outputs = student_model(inputs)
                    
                    loss = validation_criterion(outputs, labels)
                    val_loss += loss.item() * inputs.size(0) 
                    
                    _, predicted = torch.max(outputs.data, 1)
                    total_val += labels.size(0)
                    correct_val += (predicted == labels).sum().item()
            
            epoch_val_loss = val_loss / total_val
            epoch_acc_val = 100.0 * correct_val / total_val
            history['val_loss'].append(epoch_val_loss)
            history['val_acc'].append(epoch_acc_val)
            
            print(f"Epoch {epoch+1}/{num_epochs} | "
                  f"Train Loss: {epoch_loss:.4f} Acc: {epoch_acc_train:.2f}% | "
                  f"Val Loss: {epoch_val_loss:.4f} Acc: {epoch_acc_val:.2f}%")
            
            # Save best student model
            if epoch_acc_val > best_val_acc:
                best_val_acc = epoch_acc_val
                patience_counter = 0
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': student_model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'best_val_acc': best_val_acc,
                    'history': history
                }, os.path.join('./outputs/models', 'best_student_model.pth'))
            else:
                patience_counter += 1
            
            if patience_counter >= early_stopping_patience:
                print(f"Early stopping at epoch {epoch+1}")
                break
            
            if scheduler is not None:
                scheduler.step()
            
            if device == 'cuda':
                torch.cuda.empty_cache()
    
    except KeyboardInterrupt:
        print("Training interrupted by user")
    except Exception as e:
        print(f"Training failed with error: {e}")
        raise e
    
    # Trả về student model đã được huấn luyện
    return student_model, history
