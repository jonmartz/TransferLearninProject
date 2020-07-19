import numpy as np
import cv2
import pandas as pd
from tensorflow import keras


# tfds.disable_progress_bar()
#
# train_ds, validation_ds, test_ds = tfds.load(
#     "cats_vs_dogs",
#     # Reserve 10% for validation and 10% for test
#     split=["train[:40%]", "train[40%:50%]", "train[50%:60%]"],
#     as_supervised=True,  # Include labels
# )
#
# print("Number of training samples: %d" % tf.data.experimental.cardinality(train_ds))
# print(
#     "Number of validation samples: %d" % tf.data.experimental.cardinality(validation_ds)
# )
# print("Number of test samples: %d" % tf.data.experimental.cardinality(test_ds))
#
#
# train_ds = train_ds.map(lambda x, y: (tf.image.resize(x, size), y))
# validation_ds = validation_ds.map(lambda x, y: (tf.image.resize(x, size), y))
# test_ds = test_ds.map(lambda x, y: (tf.image.resize(x, size), y))
#
# """
# Besides, let's batch the data and use caching & prefetching to optimize loading speed.
# """
#
# batch_size = 32
#
# train_ds = train_ds.cache().batch(batch_size).prefetch(buffer_size=10)
# validation_ds = validation_ds.cache().batch(batch_size).prefetch(buffer_size=10)
# test_ds = test_ds.cache().batch(batch_size).prefetch(buffer_size=10)


def get_dataset(image_size, shuffle_seed=1):
    # create one hot target vectors
    labels = pd.read_csv('labels.csv')['label']
    y = np.zeros([len(labels), num_labels])
    for i, label in enumerate(labels):
        y[i][label - 1] = 1
    # get images
    x = []
    for i in range(len(labels)):
        image_id = str(i + 1)
        path = 'dataset/image_%s.jpg' % ('0' * (5 - len(image_id)) + image_id)  # add zeroes
        image = cv2.imread(path)
        resized_image = cv2.resize(image, (image_size, image_size), interpolation=cv2.INTER_AREA)
        x.append(resized_image)
    # x = np.array(x)
    # for i in [x, y]:  # shuffle
    #     np.random.seed(shuffle_seed)
    #     np.random.shuffle(i)
    return np.array(x), y


def split_dataset(x, y, fold):
    np.random.seed(fold)
    shuffled = np.arange(len(x))
    np.random.shuffle(shuffled)
    x_train, x_val, x_test = np.split(x[shuffled], [int(train_frac * len(x)), int((train_frac + val_frac) * len(x))])
    y_train, y_val, y_test = np.split(y[shuffled], [int(train_frac * len(x)), int((train_frac + val_frac) * len(x))])
    return x_train, y_train, x_val, y_val, x_test, y_test


def get_model():
    # make model above the pre-trained one
    input = keras.Input(shape=(150, 150, 3))
    augmented_data = keras.Sequential([keras.layers.experimental.preprocessing.RandomFlip("horizontal"),
                                       keras.layers.experimental.preprocessing.RandomRotation(0.1)])
    input = augmented_data(input)
    pretrained_model, input = get_pretrained_model(input)
    current_layer = pretrained_model(input, training=False)
    current_layer = keras.layers.GlobalAveragePooling2D()(current_layer)
    current_layer = keras.layers.Dropout(dropout_rate)(current_layer)
    output = keras.layers.Dense(num_labels)(current_layer)
    return keras.Model(input, output), pretrained_model


def get_pretrained_model(input):
    if pretrained_model_name == 'Xception':
        pretrained_model = keras.applications.Xception(weights="imagenet", input_shape=(150, 150, 3), include_top=False)
        pretrained_model.trainable = False
        # normalize from range (0, 255) to range (-1, 1) for using Xception
        normalization = keras.layers.experimental.preprocessing.Normalization()
        input = normalization(input)
        mean = np.array([255 / 2, 255 / 2, 255 / 2])
        normalization.set_weights([mean, mean ** 2])
    else:
        raise AttributeError('invalid pretrain model name')
    return pretrained_model, input


# todo: choose experiment parameters
pretrained_model_name = 'Xception'
train_frac = 0.7
val_frac = 0.2
cross_validation_folds = 1
dropout_rate = 0.2

num_labels = 102
if pretrained_model_name == 'Xception':
    img_size = 150
else:
    raise AttributeError('invalid pretrain model name')
print('getting dataset')
x, y = get_dataset(img_size)


for fold in range(cross_validation_folds):

    model, pretrained_model = get_model()
    model.summary()
    x_train, y_train, x_val, y_val, x_test, y_test = split_dataset(x, y, fold)

    # model.compile(
    #     optimizer=keras.optimizers.Adam(),
    #     loss=keras.losses.BinaryCrossentropy(from_logits=True),
    #     metrics=[keras.metrics.BinaryAccuracy()],
    # )
    # epochs = 20
    # model.fit(train_ds, epochs=epochs, validation_data=validation_ds)

    epochs = 10
    batch_size = 32

    model.compile(optimizer=keras.optimizers.Adam(), loss='categorical_crossentropy', metrics=['acc'])
    checkpoint = keras.callbacks.ModelCheckpoint('checkpoint.h5', save_best_only=True)
    history = model.fit(x_train, y_train, batch_size, epochs, callbacks=[checkpoint], validation_data=(x_val, y_val))
    model = keras.models.load_model('checkpoint.h5')
    test_loss, test_acc = model.evaluate(x_test, y_test)

    """
    ## Do a round of fine-tuning of the entire model
    Finally, let's unfreeze the base model and train the entire model end-to-end with a low
     learning rate.
    Importantly, although the base model becomes trainable, it is still running in
    inference mode since we passed `training=False` when calling it when we built the
    model. This means that the batch normalization layers inside won't update their batch
    statistics. If they did, they would wreck havoc on the representations learned by the
     model so far.
    """

    # Unfreeze the base_model. Note that it keeps running in inference mode
    # since we passed `training=False` when calling it. This means that
    # the batchnorm layers will not update their batch statistics.
    # This prevents the batchnorm layers from undoing all the training
    # we've done so far.
    pretrained_model.trainable = True
    model.summary()

    model.compile(optimizer=keras.optimizers.Adam(0.00001), loss='categorical_crossentropy', metrics=['acc'])

    epochs = 10
    model.fit(train_ds, epochs=epochs, validation_data=validation_ds)
