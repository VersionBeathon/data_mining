#!/user/bin/env python
# _*_coding:utf-8_*_
# @Time   :2017/2/20 13:19
# @Author :Kira
# @Software：PyCharm
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from skimage import transform as tf
from matplotlib import pyplot as plt
from skimage.measure import label, regionprops
from sklearn.utils import check_random_state
from sklearn.preprocessing import OneHotEncoder
from skimage.transform import resize
from sklearn.cross_validation import train_test_split
from pybrain.datasets import SupervisedDataSet
from pybrain.tools.shortcuts import buildNetwork
from pybrain.supervised.trainers import BackpropTrainer
from sklearn.metrics import f1_score
from sklearn.metrics import classification_report
from nltk.corpus import words
from sklearn.metrics import confusion_matrix
from nltk.metrics import edit_distance
from operator import itemgetter


def create_captcha(text, shear=0, size=(100, 25)):
    im = Image.new("L", size, "black")
    draw = ImageDraw.Draw(im)
    font = ImageFont.truetype(r"Coval.otf", 22)
    draw.text((2, 2), text, fill=1, font=font)
    image = np.array(im)
    affine_tf = tf.AffineTransform(shear=shear)
    image = tf.warp(image, affine_tf)
    return image / image.max()

image = create_captcha("GENE", shear=0.5)
plt.imshow(image, cmap="gray")
plt.show()


def segment_image(image):
    labeled_image = label(image > 0)
    subimage = []
    for region in regionprops(labeled_image):
        start_x, start_y, end_x, end_y = region.bbox
        subimage.append(image[start_x: end_x, start_y: end_y])
    if len(subimage) == 0:
        return [image,]
    return subimage
subimages = segment_image(image)
f, axes = plt.subplots(1, len(subimages), figsize=(10, 3))
for i in range(len(subimages)):
    axes[i].imshow(subimages[i], cmap='gray')
plt.show()

random_state = check_random_state(14)
letters = list("ACBDEFGHIJKLMNOPQRSTUVWXYZ")
shear_values = np.arange(0, 0.5, 0.05)


def generate_sample(random_state=None):
    random_state = check_random_state(random_state)
    letter = random_state.choice(letters)
    shear = random_state.choice(shear_values)
    return create_captcha(letter, shear=shear, size=(40, 40)), letters.index(letter)
image, target = generate_sample(random_state)
plt.imshow(image, cmap="Greys")
plt.show()
print("The target for this image is {0}".format(target))
dataset, targets = zip(*(generate_sample(random_state) for i in range(3000)))
dataset = np.array(dataset, dtype='float')
targets =np.array(targets)

onehot = OneHotEncoder()
y = onehot.fit_transform(targets.reshape(targets.shape[0], 1))
y = y.todense()

dataset = np.array([resize(segment_image(sample)[0], (20, 20)) for sample in dataset])
X = dataset.reshape((dataset.shape[0], dataset.shape[1]*dataset.shape[2]))

X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.9)
training = SupervisedDataSet(X.shape[1], y.shape[1])
for i in range(X_train.shape[0]):
    training.addSample(X_train[i], y_train[i])

testing = SupervisedDataSet(X.shape[1], y.shape[1])
for i in range(X_test.shape[0]):
    testing.addSample(X_test[i], y_test[i])

net = buildNetwork(X.shape[1], 100, y.shape[1], bias=True)
trainer = BackpropTrainer(net, training, learningrate=0.01)
trainer.trainEpochs(epochs=20)
predictions = trainer.testOnClassData(dataset=testing)
print(classification_report(y_test.argmax(axis=1), predictions))


def predict_captcha(captcha_image, neural_network):
    subimages = segment_image(captcha_image)
    predicted_word = ""
    for subimage in subimages:
        subimage = resize(subimage, (20, 20))
        outputs = net.activate(subimage.flatten())
        prediction = np.argmax(outputs)
        predicted_word += letters[prediction]
    return predicted_word


word = "GENE"
captcha = create_captcha(word, shear=0.2)
print(predict_captcha(captcha, net))


def test_prediction(word, net, shear=0.2):
    captcha = create_captcha(word, shear=shear)
    prediction = predict_captcha(captcha, net)
    prediction = predictions[:4]
    return word == prediction, word, prediction

valid_words = [word.upper() for word in words.words() if len(word) == 4]
num_correct = 0
num_incorrect = 0
for word in valid_words:
    correct, word, prediction = test_prediction(word, net, shear=0.2)
    if correct:
        num_correct += 1
    else:
        num_incorrect += 1
print("Number correct is {0}".format(num_correct))
print("Number incorrect is {0}".format(num_incorrect))

cm = confusion_matrix(np.argmax(y_test, axis=1), predictions)

plt.figure(figsize=(10, 10))
plt.imshow(cm, cmap='Blues')
plt.show()

steps = edit_distance("STEP", "STOP")
print("The number of steps needed is: {0}".format(steps))


def compute_distance(prediction, word):
    return len(prediction) - sum(prediction[i] == word[i] for i in range(len(prediction)))


def improved_prediction(word, net, dictionary, shear=0.2):
    captcha = create_captcha(word, shear=shear)
    prediction = predict_captcha(captcha, net)
    prediction = prediction[:4]
    if prediction not in dictionary:
        distances = sorted([(word, compute_distance(prediction, word)) for word in dictionary], key=itemgetter(1))
        best_word = distances[0]
        prediction = best_word[0]
    return word == prediction, word, prediction
num_correct = 0
num_incorrect = 0
for word in valid_words:
    correct, word, prediction = improved_prediction (word, net, valid_words, shear=0.2)
    if correct:
        num_correct += 1
    else:
        num_incorrect += 1
print("Number correct is {0}".format(num_correct))
print("Number incorrect is {0}".format(num_incorrect))
