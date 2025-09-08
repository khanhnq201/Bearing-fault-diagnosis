import models
import torch.optim as optim
import torch.nn as nn 
import torch
import trainer
import data_loader
import cfg
from sklearn.metrics import accuracy_score, f1_score
class OptunaOptimize:
    def __init__(self,
                 study_name='bearing_classification_study',
                 storage_path='optuna_study.db',
                 n_trials=100,
                 timeout=None,
                 seed=42):
        self.study_name = study_name
        self.storage_path = storage_path
        self.n_trials = n_trials
        self.timeout = timeout
        self.seed = seed

        self.storage = f'sqlite:///{storage_path}'

        self.best_params = None 
        self.best_value = None

        self.trial_results = []

    def suggest_hyperparameters(self, trial): #trial???
        hyperparams = {
            'learning_rate': trial.suggest_float('learing_rate', 1e-6, 1e-2, log=True),
            'optimizer_name': trial.suggest_float('optimizer_name', ['Adam', 'AdamW']),
            'weight_decay': trial.suggest_float('weight_decay', 1e-6, 1e-1, log=True),
            'batch_size': trial.suggest_categorical('batch_size', [32,64,128,256]),
            #'model_type'
            'dropout_rate': trial.suggest_float('dropout_rate', 0.1,0.7),
            'scheduler_type': trial.suggest_categorical('scheduler_type',['CosineAnnealingLR', 'ReduceLROnPlateau', 'StepLR']),
            'loss_function': trial.suggest_categorical('loss_function', ['CrossEntropyLoss', 'FocalLoss', 'LabelSmoothingLoss'])
        }
        if hyperparams['scheduler_type'] == 'CosineAnnealingLR':
            hyperparams['eta_min_factor'] = trial.suggest_float('eta_min_factor', 1e-4, 1e-1, log=True)
        elif hyperparams['scheduler_type'] == 'ReduceLROnPlateau':
            hyperparams['patience'] = trial.suggest_int('patience', 5, 20)
            hyperparams['factor'] = trial.suggest_float('factor', 0.1, 0.8)
        elif hyperparams['scheduler_type'] == 'StepLR':
            hyperparams['step_size'] = trial.suggest_int('step_size', 10, 50)
            hyperparams['gamma'] = trial.suggest_float('gamma', 0.1, 0.9)
            
        # Conditional hyperparameters for loss functions
        if hyperparams['loss_function'] == 'FocalLoss':
            hyperparams['focal_alpha'] = trial.suggest_float('focal_alpha', 0.5, 2.0)
            hyperparams['focal_gamma'] = trial.suggest_float('focal_gamma', 1.0, 3.0)
        elif hyperparams['loss_function'] == 'LabelSmoothingLoss':
            hyperparams['label_smoothing'] = trial.suggest_float('label_smoothing', 0.05, 0.3)

        return hyperparams
    
    def create_model(self, model_type, dropout_rate, input_channels=1, num_classes=3):
        if model_type == 'Deeper1DCNN':
            return models.Deeper1DCNN(input_channels, num_classes,dropout_rate)
        
    def create_optimizer(self, model, optimizer_name, learning_rate, weight_decay):
        if optimizer_name == 'Adam':
            return optim.Adam(model.parameters(),
                              lr=learning_rate,
                              weight_decay=weight_decay)
        elif optimizer_name == 'AdamW':
            return optim.AdamW(model.parameters(), 
                             lr=learning_rate, 
                             weight_decay=weight_decay)
        
    def create_scheduler(self, optimizer, scheduler_type, hyperparams):
        if scheduler_type == 'CosineAnneslingLR':
            eta_min = hyperparams['learning_rate']* hyperparams['eta_min_factor']
            return optim.lr_scheduler.CosineAnnealingLR(
                optimizer,
                T_max = hyperparams['num_epochs'],
                eta_min = eta_min
            )
        elif scheduler_type == 'ReduceLROnPlateau':
            return optim.lr_scheduler.ReduceLROnPlateau(
                optimizer, 
                patience=hyperparams['patience'],
                factor=hyperparams['factor'],
                verbose=False
            )
        elif scheduler_type == 'StepLR':
            return optim.lr_scheduler.StepLR(
                optimizer,
                step_size=hyperparams['step_size'],
                gamma=hyperparams['gamma']
            )        
    
    def create_loss_function(self, loss_function, hyperparams, num_classes=3):
        if loss_function == 'CrossEntropyLoss':
            return nn.CrossEntropyLoss()
        elif loss_function == 'FocalLoss':
            return FocalLoss(gamma=hyperparams['focal_gamma'])
        elif loss_function == 'LabelSmoothingLoss':
            return LabelSmoothingLoss(
                num_classes = num_classes,
                smoothing=hyperparams['label_smoothing']
            )
    def objective(self, trial):
        trainer.set_seed(self.seed+trial.number) #trial number?
        hyperparams = self.suggest_hyperparameters(trial)

        X_train, Y_train, X_val, Y_val, X_test, Y_test, test_files, len_processed = \
            data_loader.data_import(cfg, 500)
        
        X_train = X_train * 1e2
        X_val = X_val * 1e2
        X_test = X_test * 1e2

        original_batch_size = cfg.BATCH_SIZE
        cfg.BATCH_SIZE = hyperparams['batch_size']

        train_loader, val_loader, test_loader = data_loader.create_dataloaders(
            X_train, Y_train, X_val, Y_val, X_test, Y_test, cfg, len_processed
        )

        cfg.BATCH_SIZE = original_batch_size

        model = self.create_model(
            hyperparams['model_type'],
            hyperparams['dropout_rate']
        )
        
        optimizer = self.create_optimizer(
            model, 
            hyperparams['optimizer_name'],
            hyperparams['learning_rate'],
            hyperparams['weight_decay']
        )

        scheduler = self.create_scheduler(optimizer, hyperparams['scheduler_type'], hyperparams)
        criterion = self.create_loss_function(hyperparams['loss_function'], hyperparams)
            
        trained_model, history = trainer.train_model(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            criterion=criterion,
            optimizer=optimizer,
            save_path=f'temp_model_trial_{trial.number}.pth',
            scheduler=scheduler,
            num_epochs=hyperparams['num_epochs'],
            device=cfg.DEVICE,
            early_stopping_patience=15,
            trial=trial
        )

        val_accuracy = self.evaluate_model(trained_model, val_loader)

        temp_path = Path(f'temp_model_trial_{trial.number}.pth')
        if temp_path.exists():
            temp_path.unlink()

        trial_result = {
            'trial_number': trial.number, 
            'hyperparams': hyperparams,
            'val_accuracy': val_accuracy, 
            'val_history': history[val_accuracy]
        }

        self.trial_results(trial_result)

        return val_accuracy
    
    def evaluate_model(self, model, data_loader):
        """
        Evaluate model on given data loader
        """
        model.eval()
        all_preds = []
        all_labels = []
        
        with torch.no_grad():
            for data, labels in data_loader:
                data = data.to(cfg.DEVICE)
                labels = labels.to(cfg.DEVICE)
                
                outputs = model(data)
                preds = torch.argmax(outputs, dim=1)
                
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
        
        accuracy = accuracy_score(all_labels, all_preds)
        return accuracy

    def optimize(self):
        stu



class FocalLoss(nn.Module):
    def __init__(self, gamma=2, reduction='mean'):
        super().__init__()
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, inputs, targets):
        ce_loss = nn.functional.cross_entropy(inputs, targets, reduction='none')
        pt = torch.exp(-ce_loss)
        focal_loss = (1-pt)**self.gamma*ce_loss

        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        return focal_loss

class LabelSmoothingLoss(nn.Module):
    def __init__(self, num_classes, smoothing = 0.1):
        super().__init__()
        self.num_classes = num_classes
        self.smoothing = smoothing

    def forward(self, pred, target):
        with torch.no_grad():
            true_dist = torch.zeros_like(pred)
            true_dist.fill_(self.smoothing/ (self.num_classes - 1))
            true_dist.scatter_(1, target.data.unsqueeze(1), 1.0 - self.smoothing)
        
        return torch.mean(torch.sum(-true_dist * nn.functional.log_softmax(pred, dim=1), dim=1))

def run_optimization():
    optimizer = OptunaOptimize()
    study = optimizer.optimize()
    optimizer.save_results(study)
    optimizer.plot_optimization_results(study)

    best_config = optimizer.get_best_config()

    return optimizer, study