import cv2
import numpy as np
from keras.layers import Dense, Conv2D, Flatten, MaxPooling2D
from keras.models import Sequential
from sklearn.model_selection import train_test_split


# Load and preprocess images
def load_images(filenames):
    images = []
    for filename in filenames:
        img = cv2.imread(filename)
        img = cv2.resize(img, (64, 64))  # Resize the image
        images.append(img)
    return np.array(images)


# 'israel_flag_images' should be a list of filenames of images with Israeli flag
# 'non_israel_flag_images' should be a list of filenames of images without Israeli flag

israel_flag_images = load_images(['image1.jpg', 'image2.jpg', 'image3.jpg', 'image4.jpg', 'image5.jpg'])
non_israel_flag_images = load_images(['non_image1.jpg', 'non_image2.jpg', 'non_image3.jpg'])

X = np.concatenate((israel_flag_images, non_israel_flag_images))
y = np.concatenate((np.ones(len(israel_flag_images)), np.zeros(len(non_israel_flag_images))))

# Split the data into train and test datasets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# Normalize the data
X_train = X_train / 255.0
X_test = X_test / 255.0

# Create a model
model = Sequential()
model.add(Conv2D(32, kernel_size=3, activation='relu', input_shape=(64, 64, 3)))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Conv2D(64, kernel_size=3, activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Flatten())
model.add(Dense(64, activation='relu'))
model.add(Dense(1, activation='sigmoid'))

# Compile the model
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Train the model
model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=10)

# Evaluate the model
loss, accuracy = model.evaluate(X_test, y_test)
print(f'Accuracy: {accuracy * 100}%')

# Predict on a new image
new_image = cv2.imread('new_image.jpg')
new_image = cv2.resize(new_image, (64, 64))  # Resize the image
new_image = new_image / 255.0  # Normalize the image
new_image = np.expand_dims(new_image, axis=0)  # Add batch dimension

prediction = model.predict(new_image)
if prediction > 0.6:
    print("The image contains an Israeli flag")
    print(prediction)
else:
    print("The image does not contain an Israeli flag")
    print(prediction)
