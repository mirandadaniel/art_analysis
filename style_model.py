import os
import tensorflow as tf
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D
from tensorflow.keras.models import Model
from sklearn.metrics import confusion_matrix
import numpy as np
import pandas as pd
import cv2
import matplotlib.pyplot as plt
from PIL import Image, ImageFile
from matplotlib import image as mpimg
ImageFile.LOAD_TRUNCATED_IMAGES = True
Image.MAX_IMAGE_PIXELS = None
import random
import itertools 

def generate_gradcam(full_fp, model, img_array, layer_name):
    grad_model = Model(inputs=model.input, outputs=(model.get_layer(layer_name).output, model.output))
    with tf.GradientTape() as tape:
        conv_outputs, predictions = grad_model(img_array)
        class_idx = np.argmax(predictions[0])
        loss = predictions[:, class_idx]
    output = conv_outputs[0]
    grads = tape.gradient(loss, conv_outputs)[0]
    gate_f = tf.cast(output > 0, 'float32')
    gate_r = tf.cast(grads > 0, 'float32')
    guided_grads = gate_f * gate_r * grads
    weights = tf.reduce_mean(guided_grads, axis=(0, 1))
    cam = np.dot(output, weights)
    cam = cv2.resize(cam, (img_array.shape[2], img_array.shape[1]))
    cam = np.maximum(cam, 0)
    heatmap = cam / np.max(cam)
    orig = cv2.imread(full_fp)
    heatmap = cv2.resize(heatmap, (orig.shape[1], orig.shape[0]))
    return heatmap

image_folder = "/art_class/wikiart"
train_label_file = "/art_class/art_data/style_train2.csv"
test_label_file = "/art_class/art_data/style_val2.csv" 

train_label_df = pd.read_csv(train_label_file)
test_label_df = pd.read_csv(test_label_file)

image_files_path = "/art_class/style_class/image_files.txt"
with open(image_files_path, 'r') as file:
    image_files = file.read().splitlines()

train_pixel_array = []
test_pixel_array = []
train_labels = []
test_labels = []
test_filenames = []

for file in image_files:
    filename = os.path.relpath(file, "/art_class/wikiart") 
    matching_rows = train_label_df.loc[train_label_df['path_name'] == filename]
    if not matching_rows.empty:
        train_label = matching_rows['label'].values[0]
        train_labels.append(train_label)
        image = tf.keras.preprocessing.image.load_img(file, target_size=(256, 256))
        image_array = tf.keras.preprocessing.image.img_to_array(image)
        train_pixel_array.append(image_array)
    else:
        continue

for file in image_files:
    filename = os.path.relpath(file, "/art_class/wikiart") 
    matching_rows = test_label_df.loc[test_label_df['path_name'] == filename]
    if not matching_rows.empty:
        test_filenames.append(filename)
        test_label = matching_rows['label'].values[0]
        test_labels.append(test_label)
        image = tf.keras.preprocessing.image.load_img(file, target_size=(256, 256))
        image_array = tf.keras.preprocessing.image.img_to_array(image)
        test_pixel_array.append(image_array)
        test_filenames.append(filename)
    else:
        continue
 
test_filenames = test_filenames[1::2] 

X_train = np.array(train_pixel_array)
y_train = np.array(train_labels)
X_test = np.array(test_pixel_array)
y_test = np.array(test_labels)

X_train = X_train / 255.0
X_test = X_test / 255.0

num_classes = len(np.unique(train_labels))
y_train = tf.keras.utils.to_categorical(y_train, num_classes)
y_test = tf.keras.utils.to_categorical(y_test, num_classes)

base_model = MobileNetV2(weights='imagenet', include_top=False, input_shape=(256, 256, 3))

for layer in base_model.layers:
    layer.trainable = False

x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dense(128, activation='relu')(x)
predictions = Dense(num_classes, activation='softmax')(x)

model = Model(inputs=base_model.input, outputs=predictions)
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
model.summary()
model.fit(X_train, y_train, epochs=10, batch_size=32, validation_data=(X_test, y_test))

test_loss, test_acc = model.evaluate(X_test, y_test)
print("Test loss:", test_loss)
print("Test accuracy:", test_acc)

y_pred = model.predict(X_test)
y_pred_labels = np.argmax(y_pred, axis=1)
y_true_labels = np.argmax(y_test, axis=1)

for i in range(len(y_pred_labels)):
    predicted_label = y_pred_labels[i]
    true_label = y_true_labels[i]

cm = confusion_matrix(y_true_labels, y_pred_labels)
print("CM is: ")
print(cm)

classes = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26]
plt.figure(figsize=(8, 6))
plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
plt.title('Confusion Matrix')
plt.colorbar()
tick_marks = np.arange(len(classes))
plt.xticks(tick_marks, classes, rotation=45)
plt.yticks(tick_marks, classes)

fmt = 'd'
thresh = cm.max() / 2.
for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
    plt.text(j, i, format(cm[i, j], fmt),
             horizontalalignment="center",
             color="white" if cm[i, j] > thresh else "black")

plt.ylabel('True label')
plt.xlabel('Predicted label')
plt.tight_layout()

plt.savefig('confusion_matrix_full.png')

count = len(X_test)
count = count - 1
random_list = []
for i in range(0,100):
    n = random.randint(1,count)
    random_list.append(n)

print(random_list)

for x in random_list:
    predicted_label = y_pred_labels[x]
    true_label = y_true_labels[x]
    selected_image = X_test[x]
    original_image_fp = test_filenames[x]

    full_fp = "/art_class/wikiart/" + original_image_fp
    original_image = mpimg.imread(full_fp)
    selected_image_batch = np.expand_dims(selected_image, axis=0)
    heatmap = generate_gradcam(full_fp, model, selected_image_batch, layer_name='out_relu')
    fig = plt.figure(figsize=(10, 5))

    ax1 = fig.add_subplot(131)
    im1 = ax1.imshow(heatmap, cmap='jet')
    fig.colorbar(im1, ax=ax1)
    ax1.set_title('Heatmap')
    ax1.axis('off')

    ax2 = fig.add_subplot(132)
    im2 = ax2.imshow(original_image)
    im3 = ax2.imshow(heatmap, cmap='jet', alpha=0.65)  
    ax2.set_title('Heatmap Overlay')
    ax2.axis('off')
    
    ax3 = fig.add_subplot(133)
    im4 = ax3.imshow(original_image)
    ax3.set_title('Original image: ')
    ax3.axis('off')
    text = original_image_fp + " predicted label + true label: " + str(predicted_label) + " " + str(true_label)
    fig.text(.5, .05, text, ha='center')
    fig.tight_layout()

    plt.savefig('/art_class/style_hm/annotated_fig{}.png'.format(x))
    plt.clf() 
