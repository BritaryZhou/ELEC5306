from torch.utils.data import Dataset
import os
import PIL.Image as Image
import numpy as np
import cv2

def get_full_name_list(list_name, stage_flag):
    full_image_list = []
    with open(list_name) as f:
        image_list = f.readlines()
    if stage_flag == 'train':
        str_print = 'len(train_image_list):'
    elif stage_flag == 'test':
        str_print = 'len(test_image_list):'
    print(str_print + str(len(image_list)))
    for i in range(0, len(image_list)):
        til = image_list[i].rstrip()
        if stage_flag == 'train':
            for j in range(1, 8):
                til_png = til + '/im' + str(j) + '.png'
                full_image_list.append(til_png)
        elif stage_flag == 'test':
            til_png = til + '/im4.png'
            full_image_list.append(til_png)
    if stage_flag == 'train':
        str_print = 'len(full_train_image_list):'
    elif stage_flag == 'test':
        str_print = 'len(full_test_image_list):'
    print(str_print + str(len(full_image_list)))
    return full_image_list

def generate_cropped_images(train_input_dir, train_label_dir, batch_paths):

    # label_save_path = train_label_dir.replace('_temp', '_part') + '_crop'
    input_save_path = train_input_dir.replace('_temp', '_part') + '_crop'
    # if not os.path.exists(label_save_path):
    #     os.mkdir(label_save_path)
    if not os.path.exists(input_save_path):
        os.mkdir(input_save_path)

    for bp in batch_paths:
        # train_label_image_path = os.path.join(train_label_dir, bp)
        train_input_image_path = os.path.join(train_input_dir, bp[:-5] + '_sig20_' + bp[-5:-4] +'.jpg')
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
        img_input_crop.save(os.path.join(input_save_path, bp[:-5] + '_sig20_' + bp[-5:-4] +'.jpg'))

stage_flag = 'train'
full_train_image_list = get_full_name_list('temp_sep_trainlist.txt', stage_flag)
# stage_flag = 'test'
full_test_image_list = get_full_name_list('temp_sep_testlist.txt', stage_flag)


InputFolder ='/mnt/lustre/share/sunshuyang/vimeo_septuplet/sequences_sig20'
labelfolder = None

generate_cropped_images(train_input_dir=InputFolder, train_label_dir=labelfolder,
                                      batch_paths=full_train_image_list)
generate_cropped_images(train_input_dir=InputFolder, train_label_dir=labelfolder,
                        batch_paths=full_test_image_list)