import torch
import torch.nn.functional as F
from tqdm import tqdm
import logging

logger = logging.getLogger("PairedContrastive")


def train_one_epoch(model, dataloader, optimizer, device, epoch,
                    conf_weight=1.0, con_weight=0.0, paired_con_loss=None, max_batches=None,
                    print_confidence_stats_interval=100):

    model.train()
    total_loss = 0.0
    total_conf_loss = 0.0
    total_proj_loss = 0.0
    correct = 0
    total = 0
    
    batch_confidences = []
    batch_labels = []
    
    error_count = 0
    
    pbar = tqdm(dataloader, desc=f"Epoch {epoch}", ncols=120)
    for batch_idx, batch_data in enumerate(pbar):
        
        if max_batches is not None and batch_idx >= max_batches:
            logger.info(f"Achieved {max_batches} batches, stopping training")
            break

        try:
            all_videos = torch.stack([item['video'] for item in batch_data]).to(device, non_blocking=True)
            labels = torch.tensor([item['label'] for item in batch_data], dtype=torch.float32).to(device, non_blocking=True)
            pair_indices = torch.tensor([item['pair_index'] for item in batch_data], dtype=torch.long)
            
            optimizer.zero_grad()
            
            confidences, projections = model(all_videos)  
            
            if torch.isnan(confidences).any() or torch.isnan(projections).any():
                logger.error(f"Epoch {epoch}, Batch {batch_idx}: NaN detected in outputs!")
                logger.error(f"Confidence range: [{confidences.min():.4f}, {confidences.max():.4f}]")
                logger.error(f"Projection range: [{projections.min():.4f}, {projections.max():.4f}]")
                continue
            
            targets = labels.unsqueeze(1)  # [B, 1]
            
            loss_conf = F.binary_cross_entropy(confidences, targets.float())
            
            loss_proj = torch.tensor(0.0, device=device)
            if paired_con_loss is not None and con_weight > 0:
                loss_proj = paired_con_loss(projections, labels, pair_indices)
            
            loss = conf_weight * loss_conf + con_weight * loss_proj
            
            if torch.isnan(loss) or torch.isinf(loss):
                logger.error(f"Epoch {epoch}, Batch {batch_idx}: Invalid total loss!")
                logger.error(f"Total loss: {loss.item()}, Conf loss: {loss_conf.item()}")
                continue
            
            loss.backward()
            
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            optimizer.step()
            
            batch_size = all_videos.size(0)
            total_loss += loss.item() * batch_size
            total_conf_loss += loss_conf.item() * batch_size
            total_proj_loss += loss_proj.item() * batch_size
            
            predictions = (confidences.squeeze() > 0.5).long()
            correct += predictions.eq(labels.long()).sum().item()
            total += batch_size
            
            if print_confidence_stats_interval > 0:
                batch_confidences.append(confidences.detach().cpu().squeeze())
                batch_labels.append(labels.cpu())
            
            if max_batches is not None:
                logger.info(f"Batch {batch_idx+1}/{max_batches if max_batches else len(dataloader)}: "
                           f"Conf={loss_conf.item():.6f}, Proj={loss_proj.item():.6f}, "
                           f"Total={loss.item():.6f}, Acc={100.*correct/total:.2f}%")
            
            if (print_confidence_stats_interval > 0 and 
                (batch_idx + 1) % print_confidence_stats_interval == 0 and
                len(batch_confidences) > 0):
                
                collected_confs = torch.cat(batch_confidences)  # [M]
                collected_labels = torch.cat(batch_labels)  # [M]
                
                logger.info(f"\n{'='*60}")
                logger.info(f"Epoch {epoch}, Batch {batch_idx+1} - Confidence Statistics (near {print_confidence_stats_interval} batches)")
                logger.info(f"{'='*60}")
                logger.info(f"{'Threshold':<8} {'Count':<10} {'Percentage':<12} {'Accuracy':<10}")
                logger.info(f"{'-'*50}")
                
                for threshold in [0.95, 0.9, 0.85, 0.8, 0.75, 0.7]:
                    high_conf_mask = (collected_confs > threshold) | (collected_confs < (1 - threshold))
                    
                    if high_conf_mask.sum() == 0:
                        logger.info(f"{threshold:<8.2f} {0:<10d} {'0.00%':<12} {'0.00%':<10}")
                        continue
                    
                    high_conf_preds = collected_confs[high_conf_mask]
                    high_conf_labels = collected_labels[high_conf_mask]
                    
                    preds = (high_conf_preds > 0.5).long()
                    correct_count = preds.eq(high_conf_labels.long()).sum().item()
                    
                    count = high_conf_mask.sum().item()
                    percentage = 100.0 * count / len(collected_confs)
                    accuracy = 100.0 * correct_count / count
                    
                    logger.info(f"{threshold:<8.2f} {count:<10d} {f'{percentage:.2f}%':<12} {f'{accuracy:.2f}%':<10}")
                
                logger.info(f"{'='*60}\n")
                
                batch_confidences = []
                batch_labels = []
            
            
        except Exception as e:
            error_count += 1
            logger.error(f"Epoch {epoch}, Batch {batch_idx}: Exception occurred!")
            logger.error(f"Error: {str(e)}", exc_info=True)
            if error_count > 10:
                logger.critical(f"Too many errors ({error_count}), stopping training!")
                raise
            continue
        
        pbar.set_postfix({
            'loss': f'{loss.item():.4f}',
            'acc': f'{100.*correct/total:.2f}%'
        })
    
    avg_loss = total_loss / len(dataloader.dataset)
    avg_conf_loss = total_conf_loss / len(dataloader.dataset)
    avg_proj_loss = total_proj_loss / len(dataloader.dataset)
    acc = 100. * correct / total
    
    if error_count > 0:
        logger.warning(f"Epoch {epoch} completed with {error_count} errors")
    
    logger.info(f"Epoch {epoch} training completed - Processed {total} samples, Errors: {error_count}")
    
    return avg_loss, avg_conf_loss, avg_proj_loss, acc


def validate(model, dataloader, device, paired_con_loss=None,
             conf_weight=1.0, con_weight=0.0, print_confidence_stats=True):

    model.eval()
    total_loss = 0.0
    total_conf_loss = 0.0
    total_proj_loss = 0.0
    correct = 0
    total = 0
    error_count = 0
    
    all_confidences = []
    all_labels = []
    
    with torch.no_grad():
        pbar = tqdm(dataloader, desc="Validating", ncols=120)
        for batch_idx, batch_data in enumerate(pbar):
            try:
                all_videos = torch.stack([item['video'] for item in batch_data]).to(device, non_blocking=True)
                labels = torch.tensor([item['label'] for item in batch_data], dtype=torch.float32).to(device, non_blocking=True)
                pair_indices = torch.tensor([item['pair_index'] for item in batch_data], dtype=torch.long)
                
                confidences, projections = model(all_videos)
                
                if torch.isnan(confidences).any() or torch.isnan(projections).any():
                    logger.error(f"Validation Batch {batch_idx}: NaN in outputs!")
                    continue
                
                targets = labels.unsqueeze(1)  # [B, 1]
                loss_conf = F.smooth_l1_loss(confidences, targets)
                
                loss_proj = torch.tensor(0.0, device=device)
                if paired_con_loss is not None and con_weight > 0:
                    loss_proj = paired_con_loss(projections, labels, pair_indices)
                
                loss = conf_weight * loss_conf + con_weight * loss_proj
                
                batch_size = all_videos.size(0)
                total_loss += loss.item() * batch_size
                total_conf_loss += loss_conf.item() * batch_size
                total_proj_loss += loss_proj.item() * batch_size
                
                predictions = (confidences.squeeze() > 0.5).long()
                correct += predictions.eq(labels.long()).sum().item()
                total += batch_size
                
                if print_confidence_stats:
                    all_confidences.append(confidences.cpu().squeeze())
                    all_labels.append(labels.cpu())
                
            except Exception as e:
                error_count += 1
                logger.error(f"Validation Batch {batch_idx}: Exception occurred!")
                logger.error(f"Error: {str(e)}", exc_info=True)
                continue
            
            pbar.set_postfix({
                'loss': f'{loss.item():.4f}',
                'acc': f'{100.*correct/total:.2f}%'
            })
    
    avg_loss = total_loss / len(dataloader.dataset)
    avg_conf_loss = total_conf_loss / len(dataloader.dataset)
    avg_proj_loss = total_proj_loss / len(dataloader.dataset)
    acc = 100. * correct / total
    
    if error_count > 0:
        logger.warning(f"Validation completed with {error_count} errors")
    
    logger.info(f"Validation completed - Processed {total} samples, Errors: {error_count}")
    
    if print_confidence_stats and len(all_confidences) > 0:
        all_confs = torch.cat(all_confidences)  # [N]
        all_labs = torch.cat(all_labels)  # [N]
        
        logger.info(f"\n{'='*60}")
        logger.info(f"Validation Results (Total Samples: {len(all_confs)})")
        logger.info(f"{'='*60}")
        logger.info(f"{'Threshold':<8} {'Count':<10} {'Percentage':<12} {'Accuracy':<10}")
        logger.info(f"{'-'*50}")
        
        for threshold in [0.95, 0.9, 0.85, 0.8, 0.75, 0.7]:
            high_conf_mask = (all_confs > threshold) | (all_confs < (1 - threshold))
            
            if high_conf_mask.sum() == 0:
                logger.info(f"{threshold:<8.2f} {0:<10d} {'0.00%':<12} {'0.00%':<10}")
                continue
            
            high_conf_preds = all_confs[high_conf_mask]
            high_conf_labels = all_labs[high_conf_mask]
            
            preds = (high_conf_preds > 0.5).long()
            correct_count = preds.eq(high_conf_labels.long()).sum().item()
            
            count = high_conf_mask.sum().item()
            percentage = 100.0 * count / len(all_confs)
            accuracy = 100.0 * correct_count / count
            
            logger.info(f"{threshold:<8.2f} {count:<10d} {f'{percentage:.2f}%':<12} {f'{accuracy:.2f}%':<10}")
        
        logger.info(f"{'='*60}\n")
    
    return avg_loss, avg_conf_loss, avg_proj_loss, acc


def compute_confidence_stats(model, dataloader, device, thresholds=[0.95, 0.9, 0.85, 0.8, 0.75, 0.7]):
    model.eval()
    
    all_confidences = []
    all_labels = []
    
    with torch.no_grad():
        for batch_data in tqdm(dataloader, desc="Computing stats", ncols=100, leave=False):
            try:
                all_videos = torch.stack([item['video'] for item in batch_data]).to(device, non_blocking=True)
                labels = torch.tensor([item['label'] for item in batch_data], dtype=torch.float32)
                
                confidences, _ = model(all_videos)
                
                all_confidences.append(confidences.cpu().squeeze())
                all_labels.append(labels)
                
            except Exception as e:
                logger.error(f"Error in compute_confidence_stats: {str(e)}")
                continue
    
    all_confidences = torch.cat(all_confidences)  # [N]
    all_labels = torch.cat(all_labels)  # [N]
    
    stats = {}
    
    for threshold in thresholds:
        high_conf_mask = (all_confidences > threshold) | (all_confidences < (1 - threshold))
        
        if high_conf_mask.sum() == 0:
            stats[threshold] = {
                'count': 0,
                'total': len(all_confidences),
                'percentage': 0.0,
                'accuracy': 0.0
            }
            continue
        
        high_conf_preds = all_confidences[high_conf_mask]
        high_conf_labels = all_labels[high_conf_mask]
        
        predictions = (high_conf_preds > 0.5).long()
        correct = predictions.eq(high_conf_labels.long()).sum().item()
        
        stats[threshold] = {
            'count': high_conf_mask.sum().item(),
            'total': len(all_confidences),
            'percentage': 100.0 * high_conf_mask.sum().item() / len(all_confidences),
            'accuracy': 100.0 * correct / high_conf_mask.sum().item()
        }
    
    return stats
