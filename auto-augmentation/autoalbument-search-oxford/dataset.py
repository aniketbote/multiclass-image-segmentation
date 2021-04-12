import os
import torch.utils.data
import pandas as pd
from PIL import Image
import numpy as np
import json
import cv2
import albumentations as A


class SearchDataset(torch.utils.data.Dataset):

    def __init__(self, transform=None):
        self.transform = transform
        # Implement additional initialization logic if needed
        self.dir_path = 'C:/Users/Aniket/Documents/Aniket/multiclass-image-segmentation/data/train.csv'
        self.data_df = pd.read_csv(self.dir_path)
        self.data_config = json.loads(open('C:/Users/Aniket/Documents/Aniket/multiclass-image-segmentation/data/config.json').read())
        self.resize_transform = A.Resize(128,128, interpolation= cv2.INTER_AREA)

    def __len__(self):
        # Replace `...` with the actual implementation
        return len(self.data_df)

    def __getitem__(self, index):
        # Implement logic to get an image and its mask using the received index.
        #
        # `image` should be a NumPy array with the shape [height, width, num_channels].
        # If an image contains three color channels, it should use an RGB color scheme.
        #
        # `mask` should be a NumPy array with the shape [height, width, num_classes] where `num_classes`
        # is a value set in the `search.yaml` file. Each mask channel should encode values for a single class (usually
        # pixel in that channel has a value of 1.0 if the corresponding pixel from the image belongs to this class and
        # 0.0 otherwise). During augmentation search, `nn.BCEWithLogitsLoss` is used as a segmentation loss.
        img_id = self.data_df.iloc[index, 0]
        img_path = os.path.join(self.dir_path.split('.')[0], '{}_img.png'.format(img_id))
        mask_path = os.path.join(self.dir_path.split('.')[0], '{}_mask.png'.format(img_id))
        image = np.array(Image.open(img_path))
        mask = self.img_to_mask( np.array(Image.open(mask_path)) , self.data_config['color_map'] )
        aug = self.resize_transform(image = image, mask = mask)
        image = aug['image']
        mask = aug['mask']

        if self.transform is not None:
            transformed = self.transform(image=image, mask=mask)
            image = transformed["image"]
            mask = transformed["mask"]
        return image, mask

    def img_to_mask(self, img, color_map):
        num_classes = len(color_map.keys())
        mask_t = np.ones((img.shape[0], img.shape[1], num_classes))
        for i in range(num_classes):
            mask_t[..., i] = np.alltrue(img == color_map[str(i)], axis=2)
        return mask_t

    def mask_to_img(self, mask, color_map):
        img_t = np.ones((img.shape[0], img.shape[1], 3))
        for i in range(mask.shape[-1]):
            color_mask = mask[...,i] == 1
            img_t[color_mask] = color_map[str(i)]
        return img_t

if __name__ == "__main__":
    import matplotlib.pyplot as plt
    d = SearchDataset()
    print(d.data_config['color_map'])
    img, mask = d.__getitem__(18)
    fig, ax = plt.subplots(1,2)
    ax[0].imshow(img)
    ax[1].imshow(d.mask_to_img(mask, d.data_config['color_map']))
    plt.show()

