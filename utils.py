import torch
import torchvision
from dataset import PolypDataset
from torch.utils.data import DataLoader
from datetime import datetime


def save_checkpoint(state,epoch,model_name):
    date=datetime.date(datetime.now())
    time=datetime.time(datetime.now())
    date_time=str(date)+str("__")+str(time)
    date_time=date_time[0:20]
#     print(f' Date_time: {date_time}')
    filename=f'./All_ckpt/GAN_{model_name}_checkpoint__{date_time}__{epoch}.pth'
    print("=> Saving checkpoint")
    torch.save(state, filename)

# def load_checkpoint(checkpoint, model):
#     print("=> Loading checkpoint")
#     for keys in checkpoint["state_dict"].copy().keys():
#         if keys=="encoder1.conv.0.bias":
#             del checkpoint["state_dict"]["encoder1.conv.0.bias"]
#             print(keys)
#     model.load_state_dict(checkpoint["state_dict"])
#     print("==>Model Loaded")
# #     model.load_state_dict(checkpoint)

def get_loaders(
    train_dir,
    train_maskdir,
    val_dir,
    val_maskdir,
    batch_size,
    train_transform,
    val_transform,
    num_workers=4,
    pin_memory=True,
):
    train_ds = PolypDataset(
        image_dir=train_dir,
        mask_dir=train_maskdir,
        transform=train_transform,
        
    )

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=pin_memory,
        shuffle=True,
    )

    val_ds = PolypDataset(
        image_dir=val_dir,
        mask_dir=val_maskdir,
        transform=val_transform,
      

    )

    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=pin_memory,
        shuffle=True,
    )

    return train_loader, val_loader

def check_accuracy(loader, model, device="cuda"):
    num_correct = 0
    num_pixels = 0
    dice_score = 0
    model.eval()

    with torch.no_grad():
        for x, y in loader:
            x = x.to(device)
            y = y.to(device).unsqueeze(1)
            preds = torch.sigmoid(model(x))
            preds=preds[0]
            preds = (preds > 0.5).float()
            num_correct += (preds == y).sum()
            num_pixels += torch.numel(preds)
            dice_score += (2 * (preds * y).sum()) / (
                (preds + y).sum() + 1e-8
            )

    print(
        f"Got {num_correct}/{num_pixels} with acc {num_correct/num_pixels*100:.2f}"
    )
    print(f"Dice score: {dice_score/len(loader)}")
    model.train()

def save_predictions_as_imgs(
    loader, model, folder="saved_images/", device="cuda"
):
    model.eval()
    for idx, (x, y) in enumerate(loader):
        x = x.to(device=device)
        with torch.no_grad():
            preds = torch.sigmoid(model(x))
            preds = (preds > 0.5).float()
        torchvision.utils.save_image(
            preds, f"{folder}/pred_{idx}.png"
        )
        torchvision.utils.save_image(y.unsqueeze(1), f"{folder}{idx}.png")