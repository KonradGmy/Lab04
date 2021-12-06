import tensorflow as tf
import tensorflow_datasets as tfds
import matplotlib.pyplot as plt
import numpy as np

repeats = 10

(ds_train, ds_test), ds_info = tfds.load(
    'mnist',
    split=['train', 'test'],
    shuffle_files=True,
    as_supervised=True,
    with_info=True,
)


def normalize_img(image, label):
    return tf.cast(image, tf.float32) / 255., label


ds_train = ds_train.map(normalize_img, num_parallel_calls=tf.data.AUTOTUNE)
ds_train = ds_train.cache()
ds_train = ds_train.shuffle(ds_info.splits['train'].num_examples)
ds_train = ds_train.batch(128)
ds_train = ds_train.prefetch(tf.data.AUTOTUNE)


ds_test = ds_test.map(normalize_img, num_parallel_calls=tf.data.AUTOTUNE)
ds_test = ds_test.batch(128)
ds_test = ds_test.cache()
ds_test = ds_test.prefetch(tf.data.AUTOTUNE)

MLP = tf.keras.models.Sequential([
    tf.keras.layers.Flatten(input_shape=(28, 28)),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dense(10)
])


def plot(dest, test, val):
    plt.plot(test, label='accuracy')
    plt.plot(val, label='val_accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.ylim([0.9, 1])
    plt.legend(loc='lower right')
    plt.savefig(dest)
    plt.clf()


def test_model(title, model):
    result_test = []
    result_valid = []

    for i in range(repeats):
        print(f"{title} repeat: {i + 1}")
        model.compile(
            loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
            metrics=[tf.keras.metrics.SparseCategoricalAccuracy()],
        )

        model_history = model.fit(
            ds_train,
            verbose=0,
            epochs=3,
            validation_data=ds_test,
        )

        result_test.append(np.array(model_history.history['sparse_categorical_accuracy']))
        result_valid.append(np.array(model_history.history['val_sparse_categorical_accuracy']))
    return np.array(result_test).mean(axis=0), np.array(result_valid).mean(axis=0)


def test_CNN():
    CNN_16_22 = tf.keras.models.Sequential([
        tf.keras.layers.Conv2D(filters=16, kernel_size=(1, 1), activation='relu', input_shape=(28, 28, 1)),
        # tf.keras.layers.MaxPooling2D(pool_size=2),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(128, activation='relu'),
        tf.keras.layers.Dense(10)
    ])

    CNN_16_33 = tf.keras.models.Sequential([
        tf.keras.layers.Conv2D(filters=16, kernel_size=(3, 3), activation='relu', input_shape=(28, 28, 1)),
        # tf.keras.layers.MaxPooling2D(pool_size=2),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(128, activation='relu'),
        tf.keras.layers.Dense(10)
    ])

    CNN_16_44 = tf.keras.models.Sequential([
        tf.keras.layers.Conv2D(filters=16, kernel_size=(5, 5), activation='relu', input_shape=(28, 28, 1)),
        # tf.keras.layers.MaxPooling2D(pool_size=2),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(128, activation='relu'),
        tf.keras.layers.Dense(10)
    ])

    result = test_model("cnn 16 22", CNN_16_22)
    plot("cnn1622.png", *result)
    print(result)
    result = test_model("cnn 16 33", CNN_16_33)
    plot("cnn1633.png", *result)
    print(result)
    result = test_model("cnn 16 44", CNN_16_44)
    plot("cnn1644.png", *result)
    print(result)

    CNN_16_33 = tf.keras.models.Sequential([
        tf.keras.layers.Conv2D(filters=16, kernel_size=(3, 3), activation='relu', input_shape=(28, 28, 1)),
        # tf.keras.layers.MaxPooling2D(pool_size=2),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(128, activation='relu'),
        tf.keras.layers.Dense(10)
    ])

    CNN_32_33 = tf.keras.models.Sequential([
        tf.keras.layers.Conv2D(filters=32, kernel_size=(3, 3), activation='relu', input_shape=(28, 28, 1)),
        # tf.keras.layers.MaxPooling2D(pool_size=2),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(128, activation='relu'),
        tf.keras.layers.Dense(10)
    ])

    CNN_48_33 = tf.keras.models.Sequential([
        tf.keras.layers.Conv2D(filters=48, kernel_size=(3, 3), activation='relu', input_shape=(28, 28, 1)),
        # tf.keras.layers.MaxPooling2D(pool_size=2),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(128, activation='relu'),
        tf.keras.layers.Dense(10)
    ])

    result = test_model("cnn 16 33", CNN_16_33)
    plot("cnn1633.png", *result)
    print(result)
    result = test_model("cnn 32 33", CNN_32_33)
    plot("cnn3233.png", *result)
    print(result)
    result = test_model("cnn 48 33", CNN_48_33)
    plot("cnn4833.png", *result)
    print(result)

    CNN_16_33p = tf.keras.models.Sequential([
        tf.keras.layers.Conv2D(filters=16, kernel_size=(3, 3), activation='relu', input_shape=(28, 28, 1)),
        tf.keras.layers.MaxPooling2D(pool_size=2),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(128, activation='relu'),
        tf.keras.layers.Dense(10)
    ])

    result = test_model("cnn 16 33p", CNN_16_33p)
    plot("cnn1633p.png", *result)
    print(result)


result = test_model("mlp", MLP)
plot("mlp.png", *result)
print(result)
test_CNN()
