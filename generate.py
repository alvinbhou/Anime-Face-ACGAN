import sys,os,csv, random
import numpy as np
from sklearn.externals import joblib
from keras.layers import Input, Dense, Reshape, Flatten, Dropout, multiply, Activation
from keras.layers import BatchNormalization, Activation, Embedding, ZeroPadding2D
from keras.layers import Conv2D, Conv2DTranspose, Dropout, UpSampling2D, MaxPooling2D
from keras.layers.advanced_activations import LeakyReLU
from keras.models import Sequential, Model
import keras.layers.merge as merge
from keras.optimizers import SGD, Adam, RMSprop
from sklearn.utils import shuffle
import time
import scipy
from keras.models import load_model
import keras.backend as K
from collections import deque
import scipy.misc
from keras.utils import to_categorical

MODEL_DIR = 'pretrained_models'
def build_generator():
    kernel_init = 'glorot_uniform'
    latent_size = 100
    model = Sequential()
    model.add(Reshape((1, 1, -1), input_shape=(latent_size+16,)))
    model.add( Conv2DTranspose(filters = 512, kernel_size = (4,4), strides = (1,1), padding = "valid", data_format = "channels_last", kernel_initializer = kernel_init, ))
    model.add( BatchNormalization(momentum = 0.5))
    model.add( LeakyReLU(0.2))
    model.add( Conv2DTranspose(filters = 256, kernel_size = (4,4), strides = (2,2), padding = "same", data_format = "channels_last", kernel_initializer = kernel_init))
    model.add( BatchNormalization(momentum = 0.5))
    model.add( LeakyReLU(0.2))
    model.add( Conv2DTranspose(filters = 128, kernel_size = (4,4), strides = (2,2), padding = "same", data_format = "channels_last", kernel_initializer = kernel_init))
    model.add( BatchNormalization(momentum = 0.5))
    model.add( LeakyReLU(0.2))
    model.add( Conv2DTranspose(filters = 64, kernel_size = (4,4), strides = (2,2), padding = "same", data_format = "channels_last", kernel_initializer = kernel_init))
    model.add( BatchNormalization(momentum = 0.5))
    model.add( LeakyReLU(0.2))
    model.add( Conv2D(filters = 64, kernel_size = (3,3), strides = (1,1), padding = "same", data_format = "channels_last", kernel_initializer = kernel_init))
    model.add( BatchNormalization(momentum = 0.5))
    model.add( LeakyReLU(0.2))
    model.add( Conv2DTranspose(filters = 3, kernel_size = (4,4), strides = (2,2), padding = "same", data_format = "channels_last", kernel_initializer = kernel_init))
    model.add( Activation('tanh'))
    # 3 inputs
    latent = Input(shape=(latent_size, ))
    eyes_class = Input(shape=(1,), dtype='int32')
    hairs_class = Input(shape=(1,), dtype='int32')
    # embedding 
    eyes = Flatten()(Embedding(num_class_eyes, 8,  init='glorot_normal')(eyes_class))
    hairs = Flatten()(Embedding(num_class_hairs, 8,  init='glorot_normal')(hairs_class))
    h = merge([latent, hairs, eyes], mode='concat')
    fake_image = model(h)
    m = Model(input=[latent, hairs_class, eyes_class], output=fake_image)
    return m

def gen_noise(batch_size, latent_size):
    #input noise to gen seems to be very important!
    return np.random.normal(0, 1, size=(batch_size,latent_size))
def norm_img(img):
    img = (img / 127.5) - 1
    return img

def denorm_img(img):
    img = (img + 1) * 127.5
    return img.astype(np.uint8) 

def generate_images(generator, latent_size, hair_color, eyes_color, testing_id):
    noise = gen_noise(5,latent_size)
    hairs = np.full(5, hair_color, dtype=int)
    eyes = np.full(5,eyes_color, dtype=int)
    fake_data_X = generator.predict([noise, hairs, eyes])
    for i in range(5):
        img = denorm_img(fake_data_X[i])
        scipy.misc.imsave(os.path.join(MODEL_DIR,'sample_' +str(testing_id) +'_' + str(i+1)  +'.jpg'), img)
    
        
    # idx = np.random.randint(1, size = 16)[0]
    # img = denorm_img(fake_data_X[idx])
    
def generator_mapping():
    gen_count = 0
    color_mapping = {}
    gen_idx = {}
    with open( os.path.join(MODEL_DIR, 'model_color_mapping.csv') )as f:
        lines = csv.reader(f, delimiter=',')
        for i, line in enumerate(lines):
            generator_id = line[1] + '_' + line[2]
            color_mapping[line[0]] = generator_id
            if(generator_id not in gen_idx):
                gen_idx[generator_id] = gen_count
                gen_count += 1
    return color_mapping, gen_idx

def load_generators(gen_idx):
    generators = []
    for key, value in gen_idx.items():
        g = build_generator()
        g.load_weights(os.path.join(MODEL_DIR, key+'_GENERATOR.hdf5'))
        generators.append(g)
    return generators

text_file = sys.argv[1]
HAIRS = [ 'orange hair', 'white hair', 'aqua hair', 'gray hair','green hair', 'red hair', 'purple hair', 'pink hair','blue hair', 'black hair', 'brown hair', 'blonde hair']
EYES = [  'gray eyes', 'black eyes', 'orange eyes','pink eyes', 'yellow eyes', 'aqua eyes', 'purple eyes','green eyes', 'brown eyes', 'red eyes', 'blue eyes']

latent_size = 100
num_class_hairs = 12
num_class_eyes = 11
color_mapping, gen_idx = generator_mapping()
generators = load_generators(gen_idx)

seed = 7122
np.random.seed(seed)
with open(text_file, 'r') as f:
    lines = csv.reader(f, delimiter=',')
    for line in lines:
        testing_id = line[0]
        testing_text = line[1]
        features = testing_text.split(' ')
        hair_color = None
        eyes_color = None
        for idx, f in enumerate(features):
            if(f == 'hair'):
                color = features[idx-1]
                hair_color = color + ' hair'
            elif(f == 'eyes'):
                color = features[idx-1]
                eyes_color = color + ' eyes'
        hair_index = 7 # default pink
        eyes_index = 10 #default blue
        generator_idx = 2  #default 2
        if(hair_color in HAIRS):
            hair_index = HAIRS.index(hair_color)
        if(eyes_color in EYES):
            eyes_index = EYES.index(eyes_color)
        generate_id = color_mapping[HAIRS[hair_index]]
        generator_idx = gen_idx[generate_id]
        generator = generators[generator_idx]
        generate_images(generator, latent_size,hair_index, eyes_index, testing_id)

        # print(hair_index, eyes_index, generator_idx)
                
                




