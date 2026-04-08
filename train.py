import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import random
import json
import argparse
import warnings
from pathlib import Path
from torchvision import transforms
import numpy as np
from datetime import datetime


from utils.paired_dataset import PairedContrastiveDataset, collate_fn
from utils.paired_model import PairedContrastiveModel
from utils.paired_trainer import train_one_epoch, validate, compute_confidence_stats
from utils.paired_loss import PairedContrastiveLoss
from utils.logger import setup_logger

warnings.filterwarnings('ignore', category=UserWarning, module='torchvision.io')
warnings.filterwarnings('ignore', category=UserWarning, module='torchvision.models._utils')



def load_dataset_index(split: str = "train", index_dir: str = "./data_splits"):
    index_file = Path(index_dir) / f"{split}_index.json"
    if not index_file.exists():
        raise FileNotFoundError(f"There is no such file: {index_file}\nPlease run prepare_dataset.py first")
    with open(index_file, 'r', encoding='utf-8') as f:
        index_data = json.load(f)
    return index_data



def main():
    parser = argparse.ArgumentParser(description="CoCoVideo Deepfake Detection Training Script")
    parser.add_argument("--data_dir", type=str, default="./data_splits")
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--num_workers", type=int, default=12)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--conf_weight", type=float, default=1.0, help="confidence loss weight")
    parser.add_argument("--con_weight", type=float, default=0.5, help="contrastive loss weight")
    parser.add_argument("--num_frames", type=int, default=16)
    parser.add_argument("--frame_size", type=int, default=224)
    parser.add_argument("--emb_dim", type=int, default=128)
    parser.add_argument("--backbone", type=str, default="r3d_18")
    parser.add_argument("--pretrained", action="store_true")
    parser.add_argument("--save_dir", type=str, default="./checkpoints")
    parser.add_argument("--log_interval", type=int, default=1)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--multi_gpu", action="store_true")
    parser.add_argument("--resume", type=str, default=None, 
                        help="resume training from checkpoint, pass the .pth file path")
    parser.add_argument("--test_batches", type=int, default=None, help="test mode: only run specified number of batches and stop")
    parser.add_argument("--max_batches", type=int, default=None, help="test mode: maximum number of batches per epoch")
    parser.add_argument("--confidence_stats_interval", type=int, default=0, 
                        help="print confidence statistics every N batches (0 means no printing, recommended: 100)")
    parser.add_argument("--margin", type=float, default=1.0, help="margin for paired contrastive loss")
    
    args = parser.parse_args()
    
    torch.manual_seed(args.seed)
    random.seed(args.seed)
    np.random.seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    save_dir = Path(args.save_dir) / f"run_{timestamp}"
    save_dir.mkdir(parents=True, exist_ok=True)
    
    log_dir = Path("logs")
    log_dir.mkdir(exist_ok=True)
    logger = setup_logger(
        name="PairedContrastive",
        log_dir=str(log_dir),
        log_level="DEBUG",
        console_output=True,
        file_output=True
    )
    
    logger.info("="*60)
    logger.info("Program started")
    logger.info(f"Python: {__import__('sys').version}")
    logger.info(f"PyTorch: {torch.__version__}")
    logger.info(f"Command line arguments: {vars(args)}")
    logger.info("="*60)
    
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    
    num_gpus = torch.cuda.device_count()
    if args.multi_gpu and num_gpus > 1:
        logger.info(f"Detected {num_gpus} GPUs, using DataParallel for multi-GPU training")
    else:
        logger.info(f"Using single GPU for training: {device}")
    

    print("\n" + "="*60)
    print("Loading paired dataset")
    print("="*60)
    logger.info("="*60)
    logger.info("Starting to load paired dataset")
    logger.info(f"Data directory: {args.data_dir}")
    
    train_index = load_dataset_index("train", args.data_dir)
    val_index = load_dataset_index("val", args.data_dir)
    
    train_transform = transforms.Compose([
        transforms.Resize((args.frame_size, args.frame_size)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.ColorJitter(0.1, 0.1, 0.1, 0.05),
        transforms.Normalize(
            mean=[0.43216, 0.394666, 0.37645],
            std=[0.22803, 0.22145, 0.216989]
            )
    ])
    
    val_transform = transforms.Compose([
        transforms.Resize((args.frame_size, args.frame_size)),
        transforms.Normalize(
            mean=[0.43216, 0.394666, 0.37645],
            std=[0.22803, 0.22145, 0.216989]
            )
    ])
    
    train_dataset = PairedContrastiveDataset(train_index, T=args.num_frames, transform=train_transform)
    val_dataset = PairedContrastiveDataset(val_index, T=args.num_frames, transform=val_transform)
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        collate_fn=collate_fn,
        pin_memory=True,
        prefetch_factor=2
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=collate_fn,
        pin_memory=True,
        prefetch_factor=2
    )
    
    print(f"Training set: {len(train_dataset)} pairs, {len(train_loader)} batches")
    print(f"Validation set: {len(val_dataset)} pairs, {len(val_loader)} batches")
    logger.info(f"Training set: {len(train_dataset)} pairs, {len(train_loader)} batches")
    logger.info(f"Validation set: {len(val_dataset)} pairs, {len(val_loader)} batches")
    
    print("\n" + "="*60)
    print("Creating paired contrastive model")
    print("="*60)
    model = PairedContrastiveModel(
        backbone_name=args.backbone,
        emb_dim=args.emb_dim,
        pretrained=args.pretrained
    )
    
    if args.multi_gpu and num_gpus > 1:
        model = nn.DataParallel(model)
        print(f" Using DataParallel to wrap model, number of GPUs: {num_gpus}")
    
    model = model.to(device)
    
    print(f"  Model: {args.backbone}")
    print(f"   - Embedding dimension: {args.emb_dim}")
    print(f"   - Number of frames: {args.num_frames}")
    print(f"   - Resolution: {args.frame_size}×{args.frame_size}")
    print(f"   - Pretrained: {args.pretrained}")
    logger.info(f"Model: {args.backbone}, Embedding dimension: {args.emb_dim}, Number of frames: {args.num_frames}, Resolution: {args.frame_size}×{args.frame_size}, Pretrained: {args.pretrained}")
    
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay
    )
    
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=args.epochs,
        eta_min=args.lr * 0.01
    )
    
    start_epoch = 1
    best_val_acc = 0.0
    history = {
        'train_loss': [],
        'train_conf_loss': [],
        'train_con_loss': [],
        'train_acc': [],
        'val_loss': [],
        'val_conf_loss': [],
        'val_con_loss': [],
        'val_acc': [],
    }
    
    if args.resume is not None:
        print("\n" + "="*60)
        print(f"Resuming training from checkpoint")
        print("="*60)
        
        if not os.path.exists(args.resume):
            raise FileNotFoundError(f"Checkpoint file does not exist: {args.resume}")
        
        logger.info(f"Resuming training from checkpoint: {args.resume}")
        
        checkpoint = torch.load(args.resume, map_location=device)
        
        if isinstance(model, nn.DataParallel):
            model.module.load_state_dict(checkpoint['model_state_dict'])
        else:
            model.load_state_dict(checkpoint['model_state_dict'])

        if 'optimizer_state_dict' in checkpoint:
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

        if 'scheduler_state_dict' in checkpoint:
            scheduler.load_state_dict(checkpoint['scheduler_state_dict'])

        if 'epoch' in checkpoint:
            start_epoch = checkpoint['epoch'] + 1
            logger.info(f"Resuming training from epoch {start_epoch}")
        
        if 'best_val_acc' in checkpoint:
            best_val_acc = checkpoint['best_val_acc']
        
        if 'history' in checkpoint:
            history = checkpoint['history']
            logger.info(f"Resuming training history: {len(history['train_loss'])} epochs")
        
        print(f"{'='*60}\n")

    
    paired_con_loss = PairedContrastiveLoss(margin=args.margin)

    logger.info("Loss functions: Confidence loss + Projection contrastive loss")
    logger.info(f"  - Confidence loss weight: {args.conf_weight}")
    logger.info(f"  - Contrastive loss weight: {args.con_weight}, Margin: {args.margin}")


    print("\n" + "="*60)
    print("Starting training loop")
    start_time = datetime.now()
    logger.info("="*60)
    logger.info(f"Starting training - Timestamp: {timestamp}")
    logger.info(f"Training parameters: batch_size={args.batch_size}, lr={args.lr}, epochs={args.epochs}")
    logger.info(f"Loss weights: conf_weight={args.conf_weight}, con_weight={args.con_weight}")
    
    if args.resume:
        logger.info(f"Resuming training mode - Starting from epoch {start_epoch}, current best acc: {best_val_acc:.2f}%")
    
    if args.test_batches is not None:
        logger.info(f"Testing mode: max_batches={args.test_batches}")
    
    if args.max_batches is not None:
        logger.info(f"Testing mode: max_batches={args.max_batches}")
    
    if args.confidence_stats_interval > 0:
        logger.info(f"Confidence statistics interval: {args.confidence_stats_interval} batches")
    
    for epoch in range(start_epoch, args.epochs + 1):
        print(f"\n{'='*60}")
        print(f"Epoch {epoch}/{args.epochs}")
        print(f"{'='*60}")
        
        train_loss, train_conf, train_con, train_acc = train_one_epoch(
            model, train_loader, optimizer, device, epoch,
            conf_weight=args.conf_weight,
            con_weight=args.con_weight,
            paired_con_loss=paired_con_loss,
            max_batches=args.max_batches if args.max_batches else args.test_batches,
            print_confidence_stats_interval=args.confidence_stats_interval
        )
        
        # Test mode
        if args.test_batches is not None:
            logger.info(f"Testing mode completed - Conf={train_conf:.6f}, Proj={train_con:.6f}, Acc={train_acc:.2f}%")
            return
        
        val_loss, val_conf, val_con, val_acc = validate(
            model, val_loader, device,
            paired_con_loss=None,
            conf_weight=args.conf_weight,
            print_confidence_stats=True
        )
        
        scheduler.step()
        current_lr = optimizer.param_groups[0]['lr']
        
        # Compute confidence statistics for validation set
        print(f"\n Computing confidence statistics...")
        val_conf_stats = compute_confidence_stats(model, val_loader, device)
        
        history['train_loss'].append({epoch: train_loss})
        history['train_conf_loss'].append({epoch: train_conf})
        history['train_con_loss'].append({epoch: train_con})
        history['train_acc'].append({epoch: train_acc})
        history['val_loss'].append({epoch: val_loss})
        history['val_conf_loss'].append({epoch: val_conf})
        history['val_acc'].append({epoch: val_acc})

        if 'val_conf_stats' not in history:
            history['val_conf_stats'] = []
        history['val_conf_stats'].append({epoch: val_conf_stats})
        
        if epoch % args.log_interval == 0:
            for threshold in [0.95, 0.9, 0.85, 0.8, 0.75, 0.7]:
                stats = val_conf_stats[threshold]
                print(f"   {threshold:<8.2f} {stats['count']:<10d} {stats['percentage']:<10.2f}% {stats['accuracy']:<10.2f}%")
            
            logger.info(f"Epoch {epoch}/{args.epochs} - Train Loss: {train_loss:.4f} (Conf: {train_conf:.4f}, Con: {train_con:.4f}), Acc: {train_acc:.2f}%")
            logger.info(f"Epoch {epoch}/{args.epochs} - Val Loss: {val_loss:.4f} (Conf: {val_conf:.4f}, Con: {val_con:.4f}), Acc: {val_acc:.2f}%, LR: {current_lr:.6f}")
            
            # print confidence statistics
            logger.info(f"Epoch {epoch} Confidence Statistics:")
            for threshold in [0.95, 0.9, 0.85, 0.8, 0.75, 0.7]:
                stats = val_conf_stats[threshold]
                logger.info(f"  Threshold {threshold:.2f}: Count={stats['count']}, Percentage={stats['percentage']:.2f}%, Accuracy={stats['accuracy']:.2f}%")
        
        # save checkpoint every epoch
        model_state = model.module.state_dict() if isinstance(model, nn.DataParallel) else model.state_dict()
        epoch_model_filename = f"epoch_{epoch:03d}_acc{val_acc:.2f}_loss{val_loss:.4f}.pth"
        epoch_model_path = save_dir / epoch_model_filename
        torch.save({
            'epoch': epoch,
            'model_state_dict': model_state,
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict(),
            'best_val_acc': best_val_acc,
            'val_acc': val_acc,
            'val_loss': val_loss,
            'val_conf_stats': val_conf_stats,
            'history': history,
            'timestamp': timestamp,
            'args': vars(args)
        }, epoch_model_path)
        logger.info(f"Saving Epoch {epoch} Model - Val Acc: {val_acc:.2f}%, Val Loss: {val_loss:.4f}, File: {epoch_model_filename}")
        
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_model_filename = "best_model.pth"
            best_model_path = save_dir / best_model_filename
            
            torch.save({
                'epoch': epoch,
                'model_state_dict': model_state,
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'best_val_acc': best_val_acc,
                'val_acc': val_acc,
                'val_loss': val_loss,
                'val_conf_stats': val_conf_stats,
                'history': history,
                'timestamp': timestamp,
                'args': vars(args)
            }, best_model_path)
            
            logger.info(f"Updating Best Model - Epoch {epoch}, Val Acc: {val_acc:.2f}%, Val Loss: {val_loss:.4f}")
    
    end_time = datetime.now()
    training_time = end_time - start_time
    
    print("\n" + "="*60)
    print("Training Complete")
    logger.info("="*60)
    logger.info("Training Complete")
    logger.info(f"Training Time: {training_time}")
    logger.info(f"Best Validation Accuracy: {best_val_acc:.2f}%")
    logger.info(f"Final Validation Accuracy: {val_acc:.2f}%")
    
    history_filename = f"training_history_{timestamp}.json"
    history_path = save_dir / history_filename
    with open(history_path, 'w') as f:
        history_data = {
            'training_params': vars(args),
            'history': history
        }
        json.dump(history_data, f, indent=2)
    
    
    final_model_filename = f"final_model_epoch{args.epochs:03d}_bestacc{best_val_acc:.2f}_{timestamp}.pth"
    final_model_path = save_dir / final_model_filename
    model_state = model.module.state_dict() if isinstance(model, nn.DataParallel) else model.state_dict()
    torch.save({
        'epoch': args.epochs,
        'model_state_dict': model_state,
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict(),
        'best_val_acc': best_val_acc,
        'final_val_acc': val_acc,
        'timestamp': timestamp,
        'args': vars(args)
    }, final_model_path)
    logger.info(f"Final Model Saved: {final_model_path}")
    logger.info("="*60)


if __name__ == "__main__":
    main()
