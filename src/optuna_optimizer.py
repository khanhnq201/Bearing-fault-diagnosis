import optuna
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
from sklearn.metrics import accuracy_score, f1_score
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import json
import joblib

# Import your modules
import cfg
from src import data_loader, models, trainer, evaluate

class OptunaOptimizer:
    def __init__(self, 
                 study_name="bearing_classification_study",
                 storage_path="optuna_study.db",
                 n_trials=100,
                 timeout=None,
                 seed=42):
        """
        Initialize Optuna optimizer for bearing classification
        
        Args:
            study_name: Name of the study
            storage_path: Path to store study results
            n_trials: Number of trials to run
            timeout: Timeout in seconds (None for no timeout)
            seed: Random seed for reproducibility
        """
        self.study_name = study_name
        self.storage_path = storage_path
        self.n_trials = n_trials
        self.timeout = timeout
        self.seed = seed
        
        # Create storage URL
        self.storage = f"sqlite:///{storage_path}"
        
        # Store best results
        self.best_params = None
        self.best_value = None
        
        # For tracking
        self.trial_results = []
        
    def suggest_hyperparameters(self, trial):
        """
        Define hyperparameter search space
        """
        hyperparams = {
            # Learning rate
            'learning_rate': trial.suggest_float('learning_rate', 1e-6, 1e-2, log=True),
            
            # Optimizer selection
            'optimizer_name': trial.suggest_categorical('optimizer_name', ['Adam', 'AdamW']),
            
            # Weight decay
            'weight_decay': trial.suggest_float('weight_decay', 1e-6, 1e-1, log=True),
            
            # Batch size
            'batch_size': trial.suggest_categorical('batch_size', [32, 64, 128, 256]),
            
            # Model architecture parameters
            'model_type': trial.suggest_categorical('model_type', ['Basic1DCNN', 'Deeper1DCNN', 'Narrow_1DCNN']), #bor 
            'dropout_rate': trial.suggest_float('dropout_rate', 0.1, 0.7),
            
            # Training parameters
            'num_epochs': trial.suggest_int('num_epochs', 50, 400), #bor
            
            # Scheduler parameters
            'scheduler_type': trial.suggest_categorical('scheduler_type', ['CosineAnnealingLR', 'ReduceLROnPlateau', 'StepLR']),
            
            # Data preprocessing parameters
            'f_cutoff': trial.suggest_int('f_cutoff', 200, 1000),
            'preprocessing_scale': trial.suggest_float('preprocessing_scale', 1e1, 1e3, log=True), #bor
            
            # Loss function
            'loss_function': trial.suggest_categorical('loss_function', ['CrossEntropyLoss', 'FocalLoss', 'LabelSmoothingLoss']),
        }
        
        # Conditional hyperparameters based on scheduler type
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
            #hyperparams['focal_alpha'] = trial.suggest_float('focal_alpha', 0.5, 2.0)
            hyperparams['focal_gamma'] = trial.suggest_float('focal_gamma', 1.0, 3.0)
        elif hyperparams['loss_function'] == 'LabelSmoothingLoss':
            hyperparams['label_smoothing'] = trial.suggest_float('label_smoothing', 0.05, 0.3)
            
        return hyperparams
    
    def create_model(self, model_type, dropout_rate, input_channels=1, num_classes=3):
        """
        Create model based on hyperparameters
        """
        if model_type == 'Basic1DCNN':
            return models.Basic1DCNN(input_channels, num_classes, dropout_rate)
        elif model_type == 'Deeper1DCNN':
            return models.Deeper1DCNN(input_channels, num_classes, dropout_rate)
        elif model_type == 'Narrow_1DCNN':
            return models.Narrow_1DCNN(input_channels, num_classes, dropout_rate)
        else:
            raise ValueError(f"Unknown model type: {model_type}")
    
    def create_optimizer(self, model, optimizer_name, learning_rate, weight_decay):
        """
        Create optimizer based on hyperparameters
        """
        if optimizer_name == 'Adam':
            return optim.Adam(model.parameters(), 
                            lr=learning_rate, 
                            weight_decay=weight_decay)
        elif optimizer_name == 'AdamW':
            return optim.AdamW(model.parameters(), 
                             lr=learning_rate, 
                             weight_decay=weight_decay)
        else:
            raise ValueError(f"Unknown optimizer: {optimizer_name}")
    
    def create_scheduler(self, optimizer, scheduler_type, hyperparams):
        """
        Create learning rate scheduler based on hyperparameters
        """
        if scheduler_type == 'CosineAnnealingLR':
            eta_min = hyperparams['learning_rate'] * hyperparams['eta_min_factor']
            return optim.lr_scheduler.CosineAnnealingLR(
                optimizer, 
                T_max=hyperparams['num_epochs'], 
                eta_min=eta_min
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
        else:
            raise ValueError(f"Unknown scheduler: {scheduler_type}")
    
    def create_loss_function(self, loss_function, hyperparams, num_classes=3):
        """
        Create loss function based on hyperparameters
        """
        if loss_function == 'CrossEntropyLoss':
            return nn.CrossEntropyLoss()
        elif loss_function == 'FocalLoss':
            return FocalLoss(
                #alpha=hyperparams['focal_alpha'],
                gamma=hyperparams['focal_gamma']
            )
        elif loss_function == 'LabelSmoothingLoss':
            return LabelSmoothingLoss(
                num_classes=num_classes,
                smoothing=hyperparams['label_smoothing']
            )
        else:
            raise ValueError(f"Unknown loss function: {loss_function}")
    
    def objective(self, trial):
        """
        Objective function for Optuna optimization
        """
        try:
            # Set seed for reproducibility
            trainer.set_seed(self.seed + trial.number)
            
            # Get hyperparameters
            hyperparams = self.suggest_hyperparameters(trial)
            
            # Load data with suggested parameters
            X_train, Y_train, X_val, Y_val, X_test, Y_test, test_files, len_processed = \
                data_loader.data_import(cfg, 500)
                # data_loader.data_import(cfg, hyperparams['f_cutoff'])
            # Scale data
            # scale_factor = hyperparams['preprocessing_scale']
            # X_train = X_train * scale_factor
            # X_val = X_val * scale_factor
            # X_test = X_test * scale_factor

            X_train = X_train * 1e2
            X_val = X_val * 1e2
            X_test = X_test * 1e2
            
            # Update batch size temporarily
            original_batch_size = cfg.BATCH_SIZE
            cfg.BATCH_SIZE = hyperparams['batch_size']
            
            # Create data loaders
            train_loader, val_loader, test_loader = data_loader.create_dataloaders(  #test loader?? 
                X_train, Y_train, X_val, Y_val, X_test, Y_test, cfg, len_processed
            )
            
            # Restore original batch size
            cfg.BATCH_SIZE = original_batch_size
            
            # Create model
            model = self.create_model(
                hyperparams['model_type'],
                hyperparams['dropout_rate']
            )
            
            # Create optimizer
            optimizer = self.create_optimizer(
                model,
                hyperparams['optimizer_name'],
                hyperparams['learning_rate'],
                hyperparams['weight_decay']
            )
            
            # Create scheduler
            scheduler = self.create_scheduler(optimizer, hyperparams['scheduler_type'], hyperparams)
            
            # Create loss function
            criterion = self.create_loss_function(hyperparams['loss_function'], hyperparams)
            
            # Train model
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
                trial=trial  # Pass trial for pruning
            )
            
            # Evaluate on validation set
            val_accuracy = self.evaluate_model(trained_model, val_loader)
            
            # Clean up temporary model file
            temp_path = Path(f'temp_model_trial_{trial.number}.pth')
            if temp_path.exists():
                temp_path.unlink()
            
            # Store trial results
            trial_result = {
                'trial_number': trial.number,
                'hyperparams': hyperparams,
                'val_accuracy': val_accuracy,
                'val_history': history['val_accuracy'] if 'val_accuracy' in history else []
            }
            self.trial_results.append(trial_result)
            
            return val_accuracy
            
        except Exception as e:
            print(f"Trial {trial.number} failed with error: {e}")
            # Return a very low score for failed trials
            return 0.0
    
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
        """
        Run hyperparameter optimization
        """
        # Create study
        study = optuna.create_study(
            study_name=self.study_name,
            storage=self.storage,
            direction='maximize',
            load_if_exists=True,
            pruner=optuna.pruners.MedianPruner(
                n_startup_trials=5,
                n_warmup_steps=10,
                interval_steps=1
            )
        )
        
        # Optimize
        print(f"Starting optimization with {self.n_trials} trials...")
        study.optimize(
            self.objective,
            n_trials=self.n_trials,
            timeout=self.timeout,
            show_progress_bar=True
        )
        
        # Store best results
        self.best_params = study.best_params
        self.best_value = study.best_value
        
        print(f"Best trial: {study.best_trial.number}")
        print(f"Best value: {self.best_value:.4f}")
        print(f"Best params: {self.best_params}")
        
        return study
    
    def save_results(self, study, save_path="optuna_results"):
        """
        Save optimization results
        """
        save_path = Path(save_path)
        save_path.mkdir(exist_ok=True)
        
        # Save study
        joblib.dump(study, save_path / "study.pkl")
        
        # Save best parameters
        with open(save_path / "best_params.json", 'w') as f:
            json.dump(self.best_params, f, indent=2)
        
        # Save trial results
        with open(save_path / "trial_results.json", 'w') as f:
            json.dump(self.trial_results, f, indent=2, default=str)
        
        print(f"Results saved to {save_path}")
    
    def plot_optimization_results(self, study, save_path="optuna_plots"):
        """
        Create visualization plots for optimization results
        """
        save_path = Path(save_path)
        save_path.mkdir(exist_ok=True)
        
        # Plot optimization history
        fig1 = optuna.visualization.plot_optimization_history(study)
        fig1.write_html(save_path / "optimization_history.html")
        
        # Plot parameter importances
        try:
            fig2 = optuna.visualization.plot_param_importances(study)
            fig2.write_html(save_path / "param_importances.html")
        except:
            print("Could not create parameter importance plot (need more trials)")
        
        # Plot parallel coordinate
        fig3 = optuna.visualization.plot_parallel_coordinate(study)
        fig3.write_html(save_path / "parallel_coordinate.html")
        
        # Plot slice
        fig4 = optuna.visualization.plot_slice(study)
        fig4.write_html(save_path / "slice_plot.html")
        
        # Create custom accuracy plot
        self.plot_accuracy_distribution(save_path)
        
        print(f"Plots saved to {save_path}")
    
    def plot_accuracy_distribution(self, save_path):
        """
        Plot accuracy distribution across trials
        """
        if not self.trial_results:
            return
            
        accuracies = [result['val_accuracy'] for result in self.trial_results]
        
        plt.figure(figsize=(10, 6))
        
        # Histogram
        plt.subplot(1, 2, 1)
        plt.hist(accuracies, bins=20, alpha=0.7, edgecolor='black')
        plt.xlabel('Validation Accuracy')
        plt.ylabel('Frequency')
        plt.title('Distribution of Validation Accuracies')
        plt.grid(True, alpha=0.3)
        
        # Box plot
        plt.subplot(1, 2, 2)
        plt.boxplot(accuracies)
        plt.ylabel('Validation Accuracy')
        plt.title('Accuracy Distribution')
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(save_path / "accuracy_distribution.png", dpi=300, bbox_inches='tight')
        plt.close()
    
    def get_best_config(self):
        """
        Get the best configuration for final training
        """
        if self.best_params is None:
            raise ValueError("No optimization has been run yet")
            
        return {
            'best_params': self.best_params,
            'best_value': self.best_value,
            'cfg_updates': {
                'LEARNING_RATE': self.best_params['learning_rate'],
                'BATCH_SIZE': self.best_params['batch_size'],
                'NUM_EPOCHS': self.best_params['num_epochs']
            }
        }


# Custom Loss Functions
class FocalLoss(nn.Module):
    #def __init__(self, alpha=1, gamma=2, reduction='mean'):
    def __init__(self, gamma=2, reduction='mean'):
        super().__init__()
        #self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
        
    def forward(self, inputs, targets):
        ce_loss = nn.functional.cross_entropy(inputs, targets, reduction='none')
        pt = torch.exp(-ce_loss)
        focal_loss =(1 - pt) ** self.gamma * ce_loss
        
        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        return focal_loss


class LabelSmoothingLoss(nn.Module):
    def __init__(self, num_classes, smoothing=0.1):
        super().__init__()
        self.num_classes = num_classes
        self.smoothing = smoothing
        
    def forward(self, pred, target):
        with torch.no_grad():
            true_dist = torch.zeros_like(pred)
            true_dist.fill_(self.smoothing / (self.num_classes - 1))
            true_dist.scatter_(1, target.data.unsqueeze(1), 1.0 - self.smoothing)
        return torch.mean(torch.sum(-true_dist * nn.functional.log_softmax(pred, dim=1), dim=1))


# Usage example
def run_optimization():
    """
    Example of how to use the OptunaOptimizer
    """
    # Initialize optimizer
    optimizer = OptunaOptimizer(
        study_name="bearing_classification_v1",
        storage_path="bearing_optuna.db",
        n_trials=50,  # Reduce for testing, increase for production
        timeout=3600,  # 1 hour timeout
        seed=42
    )
    
    # Run optimization
    study = optimizer.optimize()
    
    # Save results
    optimizer.save_results(study)
    
    # Create plots
    optimizer.plot_optimization_results(study)
    
    # Get best configuration
    best_config = optimizer.get_best_config()
    print("\nBest Configuration:")
    print(f"Validation Accuracy: {best_config['best_value']:.4f}")
    print(f"Best Parameters: {best_config['best_params']}")
    
    return optimizer, study


if __name__ == "__main__":
    # Run the optimization
    optimizer, study = run_optimization()
    
    # Print summary
    print(f"\nOptimization completed!")
    print(f"Total trials: {len(study.trials)}")
    print(f"Best trial: {study.best_trial.number}")
    print(f"Best accuracy: {study.best_value:.4f}")
    
    # Print top 5 parameters by importance (if available)
    try:
        importances = optuna.importance.get_param_importances(study)
        print(f"\nTop 5 most important parameters:")
        for param, importance in sorted(importances.items(), 
                                      key=lambda x: x[1], 
                                      reverse=True)[:5]:
            print(f"  {param}: {importance:.4f}")
    except:
        print("Parameter importance analysis requires more trials")