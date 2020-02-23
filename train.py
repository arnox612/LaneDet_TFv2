import tensorflow as tf
import pandas as pd
from model.unet import unet
from config import Config
from keras.preprocessing.image import ImageDataGenerator
from keras.models import save_model, load_model
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.metrics import MeanIoU
import cv2, os
from utils.process_labels import *
from keras_applications import resnet50
from utils.data_feeder import crop_resize_data, train_image_gen

# CUDA visible device
os.environ['CUDA_VISIBLE_DEVICES'] = '0'

def train_epoch(net, epoch, data_load, optimizer, train, config):
    # net.train() only for pytorch
    pass


def create_network():
    pass


'''class LaneDataset:

    def __init__(self, csv_file, transform=None):
        super(LaneDataset, self).__init__()
        self.data = pd.read_csv(os.path.join(os.getcwd(), "data_list", csv_file), header=None,
                                names=["image", "label"])
        self.images = self.data["image"].values[1:]
        self.labels = self.data["label"].values[1:]

        self.transform = transform

    def __len__(self):
        return self.labels.shape[0]

    def __getitem__(self, idx):
        ori_image = cv2.imread(self.images[idx])
        ori_mask = cv2.imread(self.labels[idx], cv2.IMREAD_GRAYSCALE)
        train_img, train_mask = crop_resize_data(ori_image, ori_mask)
        # Encode
        train_mask = encode_labels(train_mask)
        sample = [train_img.copy(), train_mask.copy()]
        if self.transform:
            sample = self.transform(sample)
        return train_img, train_mask


def LaneData(csv_file):
    data = pd.read_csv(os.path.join(os.getcwd(), "data_list", csv_file), header=None,
                       names=["image", "label"])
    images = data["image"].values[1:]
    labels = data["label"].values[1:]
    ori_image = cv2.imread(images)
    ori_mask = cv2.imread(labels, cv2.IMREAD_GRAYSCALE)
    train_img, train_mask = crop_resize_data(ori_image, ori_mask)
    # Encode
    train_mask = encode_labels(train_mask)
    sample = [train_img.copy(), train_mask.copy()]
    #if self.transform:
    #    sample = self.transform(sample)
    return train_img, train_mask'''


def main():
    # lane_config=Config()

    '''#create model adn define optimizer
    reduced_loss, miou, pred = create_network(image, labels, classes, network=network, image_size = (IMG_SIZE[1],IMG_SIZE[0], for_test=False))
    optimizer = Adam
    optimizer.minimize(reduced_loss, no_grad_set=no_grad_set)'''
    image_size = [1536, 512, 3]

    model = unet(image_size, Config)
    # if load pretrained model
    '''if use_pretrained == True:
        #model = load_model(loadpath)
        print('loaded model from {}'.format(model_path))
    else:
        print('Train from initialized model.')'''

    #traindata = LaneDataset("train.csv")
    #maskdata = LaneDataset('train.csv')

    # training
    data_gen_args = dict(featurewise_center=True,
                         featurewise_std_normalization=True,
                         rotation_range=90,
                         width_shift_range=0.1,
                         height_shift_range=0.1,
                         zoom_range=0.2)
    seed = 1
    # image_datagen = ImageDataGenerator(**data_gen_args)
    #train_datagen = ImageDataGenerator(**data_gen_args)
    # image_generator = image_datagen.flow(imagedata, batch_size=2,seed=seed)
    #train_generator = train_datagen.flow(traindata, batch_size=2)
    # train_generator = zip(image_generator, mask_generator)

    data_dir = './data_list/train.csv'
    train_list = pd.read_csv(data_dir)
    train_reader = train_image_gen(train_list, Config.BATCH_size, image_size=[1536, 512], crop_offset=690)

    # Compile model
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=[MeanIoU(num_classes=8)], )
    model_check_path = './save_checkpoints/'
    checkpoint = ModelCheckpoint(model_check_path, monitor='val_acc', verbose=1, save_weights_only=True,
                                 save_freq='epoch')
    callbacks_lists = [checkpoint]

    # fit the model
    model.fit_generator(train_reader, steps_per_epoch=134 // 2, epochs=Config.EPOCHS, callbacks=callbacks_lists)

    '''for epoch in range(epoches):
        print('start training epoch: %d'%(epoch + 1))
        train_length = len(train_list)
        for iteration in range(int(train_length/batch_size))
'''


if __name__ == '__main__':
    main()
