from tqdm import tqdm
import torch

def eval_accuracy(model, input_size, dataloader, device, topk=(1,), show_progress=True):
    """
    评估模型准确率
    Args:
        model: 待评估的模型
        input_size: 输入图像大小
        dataloader: 数据加载器
        device: 计算设备
        topk: 要计算的 top-k 准确率
        show_progress: 是否显示进度条
    Returns:
        list: 各个 top-k 的准确率
    """
    correct = {k: 0 for k in topk}
    total = 0
    model = model.to(device)
    model.eval()
    
    iterator = tqdm(dataloader) if show_progress else dataloader
    
    try:
        with torch.no_grad():
            for img, target in iterator:
                # resize and move to device
                img = torch.nn.functional.interpolate(img, size=input_size)
                img, target = img.to(device), target.to(device)
                
                # forward pass
                output = model(img)
                
                # compute accuracy
                maxk = max(topk)
                batch_size = target.size(0)
                
                _, pred = output.topk(maxk, 1, True, True)
                pred = pred.t()
                correct_mask = pred.eq(target.view(1, -1).expand_as(pred))
                
                for k in topk:
                    correct[k] += correct_mask[:k].reshape(-1).float().sum(0).item()
                total += batch_size
                
                if show_progress:
                    iterator.set_description(f'Evaluating - Batch Size: {batch_size}')
                    
    except Exception as e:
        print(f"Error during evaluation: {str(e)}")
        raise e
    
    finally:
        if show_progress:
            iterator.close()
    
    return [correct[k]/total for k in topk]