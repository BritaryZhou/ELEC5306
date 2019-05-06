from torch.utils.data import Dataset
import os
import PIL.Image as Image
import numpy as np
import cv2


class ArtifactDataset(Dataset):
    def __init__(self, xs):
        super(ArtifactDataset, self).__init__()
        self.xs = xs

    def __getitem__(self, index):
        batch_input = self.xs[0][index]  # input batch
        batch_label = self.xs[1][index]  # label batch
        return batch_input, batch_label

    def __len__(self):
        return self.xs[0].shape[0]


def generate_single_batch(train_input_dir, train_label_dir, batch_paths, crop_flag):
    train_patches = []
    train_input_patches = []
    train_label_patches = []

    for bp in batch_paths:
        train_label_image_path = os.path.join(train_label_dir, bp)
        train_input_image_path = os.path.join(train_input_dir, bp[:-5] + '_q40_' + bp[-5:-4] +'.jpg')
        img_label = Image.open(train_label_image_path)
        img_input = Image.open(train_input_image_path)
        # crop images to half size.
        if crop_flag:
            width, height = img_input.size
            img_label = img_label.crop((width / 4, height / 4, 3 * width / 4, 3 * height / 4))
            img_input = img_input.crop((width / 4, height / 4, 3 * width / 4, 3 * height / 4))
        train_label_patches.append(np.asarray(img_label))
        train_input_patches.append(np.asarray(img_input))
    train_input_patches = np.array(train_input_patches, dtype='uint8')
    train_label_patches = np.array(train_label_patches, dtype='uint8')

    train_patches.append(train_input_patches)
    train_patches.append(train_label_patches)
    return train_patches

def generate_cropped_images(train_input_dir, train_label_dir, batch_paths):

    # label_save_path = train_label_dir.replace('_temp', '_part') + '_crop'
    input_save_path = train_input_dir.replace('_temp', '_part') + '_crop'
    # if not os.path.exists(label_save_path):
    #     os.mkdir(label_save_path)
    if not os.path.exists(input_save_path):
        os.mkdir(input_save_path)

    for bp in batch_paths:
        # train_label_image_path = os.path.join(train_label_dir, bp)
        train_input_image_path = os.path.join(train_input_dir, bp[:-5] + '_lr4_' + bp[-5:-4] +'.jpg')
        # img_label = Image.open(train_label_image_path)
        img_input = Image.open(train_input_image_path)
        width, height = img_input.size
        # img_label_crop = img_label.crop((width / 4, height / 4, 3 * width / 4, 3 * height / 4))
        img_input_crop = img_input.crop((width / 4, height / 4, 3 * width / 4, 3 * height / 4))
        # sub_label_save_path = os.path.join(label_save_path, bp.split('/')[0])
        sub_input_save_path = os.path.join(input_save_path, bp.split('/')[0])
        # if not os.path.exists(sub_label_save_path):
        #     os.mkdir(sub_label_save_path)
        if not os.path.exists(sub_input_save_path):
            os.mkdir(sub_input_save_path)
        # sub_label_save_path = os.path.join(sub_label_save_path, bp.split('/')[1])
        sub_input_save_path = os.path.join(sub_input_save_path, bp.split('/')[1])
        # if not os.path.exists(sub_label_save_path):
        #     os.mkdir(sub_label_save_path)
        if not os.path.exists(sub_input_save_path):
            os.mkdir(sub_input_save_path)
        # img_label_crop.save(os.path.join(label_save_path, bp))
        img_input_crop.save(os.path.join(input_save_path, bp[:-5] + '_lr4_' + bp[-5:-4] +'.jpg'))
