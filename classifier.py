import cv2                 
import numpy as np         
import os                  
from random import shuffle 
from tqdm import tqdm      
import tflearn
from tflearn.layers.conv import conv_2d, max_pool_2d
from tflearn.layers.core import input_data, dropout, fully_connected
from tflearn.layers.estimator import regression

import matplotlib.pyplot as plt



TRAIN_DIR = 'C:/Users/Saurabh/Desktop/Vadapav and Not a Vadapav/train'
TEST_DIR = 'C:/Users/Saurabh/Desktop/Vadapav and Not a Vadapav/test'
IMG_SIZE = 50
LR = 1e-3

MODEL_NAME = 'vadapav_vs_notvadapav-{}-{}.model'.format(LR, '6conv-basic') 

def label_img(img):
    word_label = img.split('.')[-3]
    
    if word_label == 'vadapav': return [1,0]
                               
    elif word_label == 'not_a_vadapav': return [0,1]


def create_train_data():
    training_data = []
    for img in tqdm(os.listdir(TRAIN_DIR)):
        label = label_img(img)
        path = os.path.join(TRAIN_DIR,img)
        img = cv2.imread(path,cv2.IMREAD_GRAYSCALE)
        #cv2.imshow('image', img)
        
        img = cv2.resize(img, (IMG_SIZE,IMG_SIZE))
        training_data.append([np.array(img),np.array(label)])
    shuffle(training_data)
    np.save('train_data.npy', training_data)
    return training_data

def process_test_data():
    testing_data = []
    for img in tqdm(os.listdir(TEST_DIR)):
        path = os.path.join(TEST_DIR,img)
        img_num = img.split('.')[0]
        img = cv2.imread(path,cv2.IMREAD_GRAYSCALE)
        img = cv2.resize(img, (IMG_SIZE,IMG_SIZE))
        
        testing_data.append([np.array(img), img_num])
        
    shuffle(testing_data)
    np.save('test_data.npy', testing_data)
    return testing_data

train_data = create_train_data()
# If you have already created the dataset:
# train_data = np.load('train_data.npy')

###############################################################################
test_data = process_test_data()
# If you have already created the dataset:
# test_data = np.load('test_data.npy')

###############################################################################
convnet = input_data(shape=[None, IMG_SIZE, IMG_SIZE, 1], name='input')

convnet = conv_2d(convnet, 32, 5, activation='relu')
convnet = max_pool_2d(convnet, 5)

convnet = conv_2d(convnet, 64, 5, activation='relu')
convnet = max_pool_2d(convnet, 5)

convnet = conv_2d(convnet, 128, 5, activation='relu')
convnet = max_pool_2d(convnet, 5)

convnet = conv_2d(convnet, 64, 5, activation='relu')
convnet = max_pool_2d(convnet, 5)

convnet = conv_2d(convnet, 32, 5, activation='relu')
convnet = max_pool_2d(convnet, 5)

convnet = fully_connected(convnet, 1024, activation='relu')
convnet = dropout(convnet, 0.8)

convnet = fully_connected(convnet, 2, activation='softmax')
convnet = regression(convnet, optimizer='adam', learning_rate=LR, loss='categorical_crossentropy', name='targets')

model = tflearn.DNN(convnet, tensorboard_dir='log')
###############################################################################
if os.path.exists('{}.meta'.format(MODEL_NAME)):
    model.load(MODEL_NAME)
    print('model loaded!')
###############################################################################    
train = train_data[:-60]
test = train_data[-60:]

X = np.array([i[0] for i in train]).reshape(-1,IMG_SIZE,IMG_SIZE,1)
Y = [i[1] for i in train]

test_x = np.array([i[0] for i in test]).reshape(-1,IMG_SIZE,IMG_SIZE,1)
test_y = [i[1] for i in test]

model.fit({'input': X}, {'targets': Y}, n_epoch=25, validation_set=({'input': test_x}, {'targets': test_y}), 
    snapshot_step=500, show_metric=True, run_id=MODEL_NAME)


model.save(MODEL_NAME)
###############################################################################

test_data = np.load('test_data.npy')

fig=plt.figure()

for num,data in enumerate(test_data[:72]):
   
    
    img_num = data[1]
    img_data = data[0]
    
    y = fig.add_subplot(8,9,num+1)
    orig = img_data
    data = img_data.reshape(IMG_SIZE,IMG_SIZE,1)
    #model_out = model.predict([data])[0]
    model_out = model.predict([data])[0]
    
    if np.argmax(model_out) == 1: str_label='Not A VadaPav'
    else: str_label='VadaPav'        
    y.imshow(orig)
    plt.title(str_label)
    y.axes.get_xaxis().set_visible(False)
    y.axes.get_yaxis().set_visible(False)
plt.axis('off')
plt.show()

#plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))












