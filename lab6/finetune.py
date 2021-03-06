import torch
import torchvision
from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor
from torchvision.datasets import CocoDetection

import utils
from coco_utils import get_city
import transforms

# Load a model pre-trained on COCO and put it in inference mode

print('Loading pretrained model...')
model = torchvision.models.detection.maskrcnn_resnet50_fpn(pretrained=True)
model.eval()

# Load the COCO 2017 train and val sets. We use the CocoDetection class definition
# from ./coco_utils.py, not the original torchvision.CocoDetection class. Also, we
# use transforms from ./transforms, not torchvision.transforms, because they need
# to transform the bboxes and masks along with the image.

# coco_path = "./COCO"

preprocess = transforms.Compose([
    transforms.ToTensor()
])
print('Loading COCO train, val datasets...')
coco_train_dataset = get_city('train',preprocess)
coco_val_dataset = get_city('val',preprocess)

def collate_fn(batch):
    return tuple(zip(*batch))

train_dataloader = torch.utils.data.DataLoader(coco_train_dataset, batch_size=4, shuffle=True, num_workers=0, collate_fn=collate_fn)
val_dataloader = torch.utils.data.DataLoader(coco_val_dataset, batch_size=2, shuffle=False, num_workers=0, collate_fn=collate_fn)



device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)
model.to(device)

from engine import train_one_epoch, evaluate
from mask_util import prediction_to_mask_image
from matplotlib import pyplot as plt

# Training
params = [p for p in model.parameters() if p.requires_grad]
optimizer = torch.optim.SGD(params, lr=0.005, momentum=0.9, weight_decay=0.0005)
# and a learning rate scheduler
lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer,step_size=3,gamma=0.1)
num_epochs = 100
for epoch in range(num_epochs):
    # train for one epoch, printing every 10 iterations
    print(f"=========== {epoch} ===========")
    # Save Image
    model.eval()
    images, targets = next(iter(val_dataloader))
    images = [img.to(device) for img in images]
    predictions = model(images)
    masked_img = prediction_to_mask_image(images, predictions, 0, 0.5)
    plt.imsave(f'./mask_results/epoch-{epoch}.jpg',masked_img)
    torch.save(model.state_dict(), (f'checkpoints/maskrcnn-{epoch}.pth'))

    train_one_epoch(model, optimizer, train_dataloader, device, epoch, print_freq=500)
    lr_scheduler.step()
    evaluate(model, val_dataloader, device=device)
