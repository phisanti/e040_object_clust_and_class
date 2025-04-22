import os
import sys
import argparse
import json
import torch
from torch.nn import CrossEntropyLoss
from torch.optim import AdamW

# MONAI imports
from monai.metrics import ROCAUCMetric
from monai.transforms import (
    Compose,
    RandRotate90,
    RandGaussianNoise,
    RandGaussianSmooth,
    ScaleIntensity,
    RandZoom,
    ToTensor,
)


# Add semantic_bac_segment to path for imports
sys.path.insert(0, '/Users/santiago/switchdrive/boeck_lab_projects/Semantic_bac_segment/src')
sys.path.insert(0, '/Users/santiago/switchdrive/boeck_lab_projects/e040_object_clust_and_class/')

from semantic_bac_segment.confreader import ConfReader
from semantic_bac_segment.utils import get_device
from semantic_bac_segment.trainlogger import TrainLogger
from semantic_bac_segment.schedulerfactory import SchedulerFactory
from semantic_bac_segment.trainer import MonaiTrainer
from semantic_bac_segment.model_loader import ModelRegistry

# Local imports
from scripts.object_classif_loader import ObjectClassifDatasetCreator
from scripts.flexresnet import FlexResNet
from scripts.custom_metrics import (ConfusionMatrixMetricWrapper, TopKAccuracy)

def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Train object classifier")
    parser.add_argument("-c", "--config", required=True, help="Path to config file")
    args = parser.parse_args()

    # Read configs
    c_reader = ConfReader(args.config)
    configs = c_reader.opt
    device = get_device()
    
    # Setup logger
    debugging = configs.trainer_params.get("debugging", False)
    log_level = "DEBUG" if debugging else "INFO"
    trainlogger = TrainLogger("ObjectClassifierTrainer", level=log_level)
    trainlogger.log(c_reader.pretty_print(configs), level="INFO")

    # Setup transforms
    train_transform = Compose([
        #ScaleIntensity(),
        RandRotate90(prob=0.5),
        RandGaussianNoise(prob=0.5, mean=0.0, std=0.1),
        RandGaussianSmooth(prob=0.5, sigma_x=(0.1, 1.1), sigma_y=(0.1, 1.1)),
        RandZoom(prob=0.5, min_zoom=0.75, max_zoom=1.25),
        ToTensor()
    ])
    
    val_transform = Compose([
        #ScaleIntensity(),
        ToTensor()
    ])

    # Create dataset
    dataset_creator = ObjectClassifDatasetCreator(
        configs.dataset_params["image_dir"],
        configs.dataset_params["annotation_file"],
        val_ratio=configs.dataset_params.get("val_ratio", 0.2),
        resize_dim=configs.dataset_params.get("resize_dim", 64)
    )
    
    # Get categories
    categories = dataset_creator.get_categories()
    num_classes = len(categories)
    trainlogger.log(f"Found {num_classes} categories: {categories}", level="INFO")
    
    # Create dataloaders
    train_loader, val_loader = dataset_creator.create_datasets(
        train_transform=train_transform,
        val_transform=val_transform,
        batch_size=configs.dataset_params.get("batch_size", 32)
    )
    
    trainlogger.log(f"Train dataset size: {len(train_loader.dataset)} objects", level="INFO")
    trainlogger.log(f"Validation dataset size: {len(val_loader.dataset)} objects", level="INFO")

    # Register model
    model_registry = ModelRegistry()
    model_registry.register("FlexResNet", FlexResNet)
    
    # Load model architectures
    with open(configs.trainer_params["model_settings"]) as file:
        network_arch = json.load(file)
    
    # Register FlexResNet model
    model_registry = ModelRegistry()
    model_registry.register("FlexResNet", FlexResNet)
    
    # Setup loss function
    loss_function = CrossEntropyLoss()
    
    # Setup metrics
    metrics = {

        # Loss
        "CrossEntropy": CrossEntropyLoss(),
        
        # Accuracy metrics
        "Top3_Accuracy": TopKAccuracy(k=3),
        "Accuracy": ConfusionMatrixMetricWrapper("accuracy"),
        "F1_Score": ConfusionMatrixMetricWrapper("f1 score"),
    
        # Additional useful metrics from confusion matrix
        "Precision": ConfusionMatrixMetricWrapper("precision"),
        "Sensitivity": ConfusionMatrixMetricWrapper("sensitivity"),
        "Specificity": ConfusionMatrixMetricWrapper("specificity"),
        "BalancedAccuracy": ConfusionMatrixMetricWrapper("balanced accuracy"),
            
        }
    
    # Create trainer
    trainer = MonaiTrainer(
        None,  # Model will be set in multi_train
        train_loader,
        val_loader,
        None,  # Optimizer will be set in multi_train
        None,  # Scheduler will be set in multi_train
        device,
        output_transform=None,
        logger=trainlogger,
        debugging=configs.trainer_params.get("debugging", False),
        accumulation_steps=configs.trainer_params.get("accumulation_steps", 1)
    )
    
    # Set early stopping
    trainer.set_early_stop(patience=configs.trainer_params.get("early_stop_patiente", 5))
    
    # Train models
    trainer.multi_train(
        network_arch,
        loss_function,
        metrics,
        configs.trainer_params.get("num_epochs", 100),
        configs.trainer_params.get("model_save", "./results/"),
        configs.optimizer_params,
        SchedulerFactory,
        model_registry=model_registry
    )


if __name__ == "__main__":
    main()
