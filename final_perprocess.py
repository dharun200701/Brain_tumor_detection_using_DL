import tensorflow as tf
from tensorflow.keras import layers, models
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, classification_report
import os
# ==============================
# PARAMETERS
# ==============================
IMG_SIZE = 224
BATCH_SIZE = 32
EPOCHS = 10
BASE_DATASET_DIR = "Dataset"
TRAIN_DIR = os.path.join(BASE_DATASET_DIR, "Training")
TEST_DIR = os.path.join(BASE_DATASET_DIR, "Testing")
# ==============================
# LOAD DATASET
# ==============================
train_ds = tf.keras.preprocessing.image_dataset_from_directory(
    TRAIN_DIR,
    validation_split=0.2,
    subset="training",
    seed=123,
    image_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE
)

val_ds = tf.keras.preprocessing.image_dataset_from_directory(
    TRAIN_DIR,
    validation_split=0.2,
    subset="validation",
    seed=123,
    image_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE
)

test_ds = tf.keras.preprocessing.image_dataset_from_directory(
    TEST_DIR,
    image_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE,
    shuffle=False
)

class_names = train_ds.class_names
print("Classes:", class_names)

# ==============================
# NORMALIZATION
# ==============================
normalization_layer = layers.Rescaling(1./255)

train_ds = train_ds.map(lambda x, y: (normalization_layer(x), y))
val_ds = val_ds.map(lambda x, y: (normalization_layer(x), y))
test_ds = test_ds.map(lambda x, y: (normalization_layer(x), y))

AUTOTUNE = tf.data.AUTOTUNE
train_ds = train_ds.prefetch(AUTOTUNE)
val_ds = val_ds.prefetch(AUTOTUNE)
test_ds = test_ds.prefetch(AUTOTUNE)

# ==============================
# MODEL (TRANSFER LEARNING)
# ==============================
base_model = tf.keras.applications.MobileNetV2(
    input_shape=(224,224,3),
    include_top=False,
    weights='imagenet'
)

base_model.trainable = False

x = base_model.output
x = layers.GlobalAveragePooling2D()(x)
x = layers.Dense(128, activation='relu')(x)
x = layers.Dropout(0.5)(x)
output = layers.Dense(1, activation='sigmoid')(x)

model = tf.keras.Model(inputs=base_model.input, outputs=output)

# ==============================
# COMPILE
# ==============================
model.compile(
    optimizer='adam',
    loss='binary_crossentropy',
    metrics=['accuracy']
)

# ==============================
# TRAIN
# ==============================
history = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=EPOCHS
)

# ==============================
# PLOT RESULTS
# ==============================
acc = history.history['accuracy']
val_acc = history.history['val_accuracy']
loss = history.history['loss']
val_loss = history.history['val_loss']

plt.figure(figsize=(12,4))

plt.subplot(1,2,1)
plt.plot(acc, label='Train Accuracy')
plt.plot(val_acc, label='Val Accuracy')
plt.legend()
plt.title("Accuracy")

plt.subplot(1,2,2)
plt.plot(loss, label='Train Loss')
plt.plot(val_loss, label='Val Loss')
plt.legend()
plt.title("Loss")

plt.show()

# ==============================
# TEST EVALUATION
# ==============================
test_loss, test_acc = model.evaluate(test_ds)
print("Test Accuracy:", test_acc)

# ==============================
# CONFUSION MATRIX
# ==============================
y_true = []
y_pred = []

for images, labels in test_ds:
    preds = model.predict(images)
    preds = (preds > 0.5).astype(int)

    y_true.extend(labels.numpy())
    y_pred.extend(preds.flatten())

print("Confusion Matrix:\n", confusion_matrix(y_true, y_pred))
print("\nClassification Report:\n", classification_report(y_true, y_pred))

# ==============================
# 🔥 PREDICT FIRST TEST IMAGE
# ==============================
for images, labels in test_ds.take(1):
    first_image = images[0]
    first_label = labels[0]

img_array = tf.expand_dims(first_image, axis=0)
prediction = model.predict(img_array)

if prediction[0][0] > 0.5:
    pred_class = "Tumor"
else:
    pred_class = "No Tumor"

actual_class = "Tumor" if first_label.numpy() == 1 else "No Tumor"

print("\nFIRST TEST IMAGE RESULT")
print("Actual:", actual_class)
print("Predicted:", pred_class)

# ==============================
# SHOW IMAGE
# ==============================
plt.imshow(first_image.numpy())
plt.title(f"Actual: {actual_class} | Predicted: {pred_class}")
plt.axis("off")
plt.show()

# ==============================
# SAVE MODEL
# ==============================
model.save("brain_tumor_model.h5")