"""
    Author: Junzheng Wu, Gavin Heidenreich
    Email: jwu220@uottawa.ca, gheidenr@uottawa.ca
    Organization: University of Ottawa (Silasi Lab)
"""

from warnings import simplefilter 
simplefilter(action='ignore', category=FutureWarning)
simplefilter(action='ignore', category=UserWarning)

import keras
import os
import keras.backend as K
from data_utils import prepare_for_training
import numpy as np
import cv2
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import TensorBoard

# tensorboard wasnt working because of some dependency-hell reason, and isnt worth the time or risk to fix.
# maybe it will work on your computer, idk

# make sure youre using a relatively balanced training dataset
class Detector():
    def __init__(self, weights_path="None"):
        self.weights_path = None
        if os.path.exists(weights_path):
            self.weights_path = weights_path
        self.model = self.get_model()

    def get_model(self):
        '''
        Input shape = (224, 224, 3)
        '''
        input_shape = (224, 224, 3)
        input = keras.layers.Input(shape=input_shape)
        x = keras.applications.mobilenet_v2.MobileNetV2(include_top=False, weights='imagenet')(input)
        x = keras.layers.GlobalMaxPooling2D()(x)

        # x = keras.layers.Dense(512)(x)
        # x = keras.layers.BatchNormalization()(x)
        # x = keras.layers.Activation('relu')(x)
        #
        # x = keras.layers.Dense(1024)(x)
        # x = keras.layers.BatchNormalization()(x)
        # x = keras.layers.Activation('relu')(x)

        output = keras.layers.Dense(1, activation='sigmoid')(x)
        model = keras.Model(inputs=[input], outputs=[output])
        if self.weights_path:
            print("Model loaded")
            model.load_weights(self.weights_path)
        opt = keras.optimizers.Adam()
        loss = keras.losses.binary_crossentropy
        metric = keras.metrics.binary_accuracy
        # metric = keras.metrics.AUC()
        # tensorboard = TensorBoard(log_dir="logs/{}".format('pellet_detector_10e'))
        model.compile(optimizer=opt, loss=loss, metrics=[metric])
        # model.summary()
        return model

    def train(self, x, y, batch_size=32, epoch=20, split=0.1):
        te_index = int(x.shape[0] * split)
        teX, teY = x[0: te_index], y[0: te_index]
        trX, trY = x[te_index:], y[te_index:]

        datagen = ImageDataGenerator(
            featurewise_center=False,
            featurewise_std_normalization=False,
            rotation_range=15,
            width_shift_range=0.1,
            height_shift_range=0.1,
            vertical_flip=True,
            # zoom_range=[0.9, 1.1],
            # brightness_range=[0.5, 1.2],
            rescale=1./255.
        )
        datagen.fit(x)
        callBackList = []
        callBackList.append(keras.callbacks.ModelCheckpoint('model/model.h5', save_best_only=True))
        # callBackList.append(keras.callbacks.TensorBoard(log_dir="./logs"))
        # callBackList.append(keras.callbacks.ModelCheckpoint(self.weights_path, save_best_only=True))
        step_per_epoch = int(trX.shape[0] / batch_size)
        test_datagen = ImageDataGenerator(rescale=1./255.)
        self.model.fit_generator(datagen.flow(trX, trY, batch_size=batch_size), steps_per_epoch=step_per_epoch, epochs=epoch,
                                 callbacks=callBackList, validation_data=test_datagen.flow(teX, teY, batch_size=32), validation_steps=20)

    def predict(self, images):
        result = self.model.predict(images / 255.)
        return result

    def predict_on_single_raw_image(self, opencv_image):
        # pixel coords of pellet cam
        start_row, start_col = int(720 - 448), int((1280 // 2) - (448 // 2))
        end_row, end_col = int(720), int((1280 // 2) + (448 // 2))
        opencv_image = opencv_image[start_row:end_row, start_col:end_col]

        predict_image = np.asarray([cv2.resize(opencv_image, (224, 224))])
        predict_image = predict_image / 255.
        return self.model.predict(predict_image)[0]

    def predict_in_real_use(self, opencv_image):
        # pixel coords of pellet cam
        start_row, start_col = int(720 - 448), int((1280 // 2) - (448 // 2))
        end_row, end_col = int(720), int((1280 // 2) + (448 // 2))
        opencv_image = opencv_image[start_row:end_row, start_col:end_col]

        predict_image = np.asarray([cv2.resize(opencv_image, (224, 224))])
        predict_image = predict_image / 255.
        if self.model.predict(predict_image)[0] > 0.5:
            return True
        return False

def test_on_video(video_file):
    video_stream = cv2.VideoCapture(video_file)
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    # cap = cv2.VideoWriter('detector_sample.avi', fourcc, 30, (640, 360))
    grab, frame = video_stream.read()
    # print(frame.shape)
    d = Detector("model/model.h5")
    frame_cnt = 0
    while frame is not None:
        if frame_cnt % 1 == 0:
            result = d.predict_on_single_raw_image(frame)[0]
            cv2.putText(frame, str(result), (0, 0), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), lineType=cv2.LINE_AA)
            # cv2.putText(frame,"Confidence: %.4f" % result, (0, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), lineType=cv2.LINE_AA)
            if result > 0.5:
                cv2.putText(frame, "Display" % result, (0, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0),
                            lineType=cv2.LINE_AA)
            else:
                cv2.putText(frame, "Keep still" % result, (0, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0),
                            lineType=cv2.LINE_AA)
            print(frame_cnt, result)
            cv2.imshow("test_on_video", frame)
            # gray = cv2.cvt
            # cap.write(frame)

            frame_cnt += 1
            grab, frame = video_stream.read()
        else:
            cv2.waitKey(80)
            # cap.write(frame)
            grab, frame = video_stream.read()
            frame_cnt += 1
        cv2.waitKey(15)

    # cap.release()
    video_stream.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    TEST = True
    #
    #
    if not TEST:
        d = Detector()
        #
        # output_folder = "/mnt/4T/pellet_output"
        output_folder = 'labeled_data'
        x, y = prepare_for_training(output_folder)
        print(x.shape)
        print(y.shape)
        d.train(x, y)
        # # #
        # print(d.predict(x[10: 20, :, : ,:]))
        # print(y[10: 20])
    else:
        # d = Detector("model.h5")

        print(os.getcwd())
        # test_on_video('/mnt/4T/dlc_reaching_3d/2020-01-27_(11-26-05)_00784A16A581_86885_18892.avi')
        test_on_video('/home/silasi/3_mirror_homecage/detector_files/videos_for_testing/2020-10-20_(05-34-06)_00784A161B3F_17_3530.avi')
