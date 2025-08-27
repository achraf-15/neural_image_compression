import os
import random
import requests
from pycocotools.coco import COCO
import zipfile
from tqdm import tqdm 

def download_coco_subset(
    out_dir="./data/coco_train_subset",
    split="train2017",
    n_images=1000,
    ann_url="http://images.cocodataset.org/annotations/annotations_trainval2017.zip"
):
    """
    Download a random subset of images from the COCO 2017 dataset.
    
    Args:
        out_dir (str): Where to save images.
        split (str): "train2017" or "val2017".
        n_images (int): Number of images to download.
        ann_url (str): COCO annotations url.
    """
    os.makedirs(out_dir, exist_ok=True)

    # Download annotations if not already
    if not os.path.exists("./data/annotations"):
        os.system(f"wget {ann_url}")
        
        with zipfile.ZipFile("annotations_trainval2017.zip", "r") as zip_ref:
            zip_ref.extractall("./data/")

    # Init COCO api
    annFile = f".data/annotations/instances_{split}.json"
    coco = COCO(annFile)

    # Get all image ids
    img_ids = coco.getImgIds()
    random.shuffle(img_ids)
    img_ids = img_ids[:n_images]

    for i, img_id in tqdm(enumerate(img_ids, 1), desc="Downloading Images"):
        img_info = coco.loadImgs(img_id)[0]
        url = img_info["coco_url"]
        filename = os.path.join(out_dir, img_info["file_name"])

        if not os.path.exists(filename):
            r = requests.get(url, stream=True)
            if r.status_code == 200:
                with open(filename, "wb") as f:
                    f.write(r.content)


    print(f"Download Done. Images saved in {out_dir}")

if __name__ == "__main__":
    download_coco_subset(out_dir="./data/coco_train_subset", split="train2017", n_images=5000)