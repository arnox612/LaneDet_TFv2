import cv2
import numpy as np
import os
from utils.process_labels import encode_labels


def crop_resize_data(image, label=None, image_size=[1024, 384], offset=690):
    roi_image = image[offset:, :]
    if label is not None:
        #roi_image = image_augmentation(roi_image)
        roi_label = label[offset:, :]
        # crop_image, crop_label = random_crop(roi_image, roi_label)
        train_image = cv2.resize(roi_image, (image_size[0], image_size[1]), interpolation=cv2.INTER_LINEAR)
        train_label = cv2.resize(roi_label, (image_size[0], image_size[1]), interpolation=cv2.INTER_NEAREST)
        # cv2.imshow('train_img', train_image)
        # cv2.imshow('train_label', train_label * 100)
        # cv2.waitKey(0)
        return train_image, train_label
    else:
        train_image = cv2.resize(roi_image, (image_size[0], image_size[1]), interpolation=cv2.INTER_LINEAR)
        return train_image


def train_image_gen(train_list, batch_size=2, image_size=(1024, 384), crop_offset=690):
    all_batches_index = np.arange(0, len(train_list))
    out_images = []
    out_masks = []
    image_dir = np.array(train_list['image'])
    label_dir = np.array(train_list['label'])

    while True:
        np.random.shuffle(all_batches_index)
        for index in all_batches_index:
            if os.path.exists(image_dir[index]):
                ori_image = cv2.imread(image_dir[index])
                ori_mask = cv2.imread(label_dir[index], cv2.IMREAD_GRAYSCALE)
                # resize to train size
                train_img, train_mask = crop_resize_data(ori_image, ori_mask, image_size, crop_offset)
                # Encode
                train_mask = encode_labels(train_mask)

                out_images.append(train_img)
                out_masks.append(train_mask)
                print('len,out_images,masks,batch_size:,',len(out_images), len(out_masks), batch_size)
        if len(out_images) >= batch_size:
            out_images = np.array(out_images)
            out_masks = np.array(out_masks)
            out_images = out_images[:, :, :, ::-1].transpose(0, 3, 1, 2).astype(np.float32) / (255.0 / 2) - 1
            out_masks = out_masks.astype(np.int64)
            yield out_images, out_masks
            out_images, out_masks = [], []
        else:
            print(image_dir, 'does not exist.')
