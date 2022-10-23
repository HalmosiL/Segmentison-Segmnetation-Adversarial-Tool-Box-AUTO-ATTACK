import torch
import json
import sys

sys.path.append('../')

from modules.model import load_model, get_model_dummy
from attacks.bim import BIM

CONFIG_PATH_MAIN = "../configs/config_main.json"
CONFIG_MAIN = json.load(open(CONFIG_PATH_MAIN ))

if(CONFIG_MAIN["MODE"] == "DUMMY"):
    model = get_model_dummy(CONFIG_MAIN["DEVICE"])
elif(CONFIG_MAIN["MODE"] == "NORMAL"):
    model = load_model(
        CONFIG_MAIN["MODEL_PATH"], 
        CONFIG_MAIN["DEVICE"]
    )

val_transform = transform.Compose([
    transform.Crop(
        [CONFIG_MAIN["HEIGHT"], CONFIG_MAIN["WIGHT"]],
        crop_type='center',
        padding=mean,
        ignore_label=CONFIG_MAIN["IGNOR_LABEL"]
    ),
    transform.ToTensor(),
    transform.Normalize(mean=mean, std=std)])

val_loader = torch.utils.data.DataLoader(
    dataset=SemData(
        split='val',
        data_root=CONFIG_MAIN['DATA_PATH'],
        data_list=CONFIG_MAIN['IMAGE_LIST'],
        transform=val_transform
    ),
    batch_size=1,
    num_workers=CONFIG_MAIN['NUMBER_OF_WORKERS'],
    pin_memory=CONFIG_MAIN['PIN_MEMORY']
)

for i in range(len(val_loader.__len__())):
    image, target = val_loader.__getitem__(i)
    image = BIM(
        image,
        target,
        model,
        eps=CONFIG_MAIN["EPS"],
        k_number=CONFIG_MAIN["NUMBER_OF_ITERS"],
        alpha=CONFIG_MAIN["ALPHA"],
        device=CONFIG_MAIN["DEVICE"]
    )