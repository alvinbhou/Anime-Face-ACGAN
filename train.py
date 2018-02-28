import sys,os,csv, random
import skimage, skimage.io, skimage.transform
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
from scipy.interpolate import spline
from collections import deque
import scipy.misc
from keras.utils import to_categorical
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

def tag_preprocess(data_path):
    HAIRS = [ 'orange hair', 'white hair', 'aqua hair', 'gray hair','green hair', 'red hair', 'purple hair', 'pink hair','blue hair', 'black hair', 'brown hair', 'blonde hair']
    EYES = [  'gray eyes', 'black eyes', 'orange eyes','pink eyes', 'yellow eyes', 'aqua eyes', 'purple eyes','green eyes', 'brown eyes', 'red eyes', 'blue eyes']
    LABELS = [ 'gray eyes', 'black eyes', 'orange eyes','pink eyes', 'yellow eyes', 'aqua eyes', 'purple eyes','green eyes', 'brown eyes', 'red eyes', 'blue eyes','orange hair', 'white hair', 'aqua hair', 'gray hair','green hair', 'red hair', 'purple hair', 'pink hair','blue hair', 'black hair', 'brown hair', 'blonde hair']
    with open(os.path.join(data_path, 'tags_clean.csv'), 'r') as file:
        lines = csv.reader(file, delimiter=',')
        y_hairs = []
        y_eyes = []
        y_index = []
        for i, line in enumerate(lines):
            y = np.zeros(len(LABELS))
            idx = line[0]
            feats = line[1]
            feats = feats.split('\t')[:-1]
            flag_hair = False
            flag_eyes = False
            y_hair = []
            y_eye = []
            for feat in feats:
                feat = feat[:feat.index(':')]
                if(feat in HAIRS):
                    y_hair.append(HAIRS.index(feat))
                if(feat in EYES):
                    y_eye.append(EYES.index(feat))
            if(len(y_hair) == 1 and len(y_eye) == 1):
                y_hairs.append(y_hair)
                y_eyes.append(y_eye)
                y_index.append(i)
            
        y_eyes = np.array(y_eyes)
        # y_eyes = to_categorical(y_eyes)
        y_hairs = np.array(y_hairs)
        # y_hairs = to_categorical(y_hairs)
        y_index = np.array(y_index)
        # print(y_hairs.shape)
        # print(y_eyes.shape)
        # print(y_index.shape)
        return y_hairs, y_eyes, y_index


def load_data(data_path, y_hairs, y_eyes, y_index):
    with open(os.path.join(data_path, 'X_data_norepeat.jlib'), 'rb') as file:
        X_data = joblib.load(file)
        return X_data


def build_generator():
    kernel_init = 'glorot_uniform'
    latent_size = 100
    model = Sequential(name = 'generator_model')
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
    latent_size = 100
    # 3 inputs
    latent = Input(shape=(latent_size, ))
    eyes_class = Input(shape=(1,), dtype='int32')
    hairs_class = Input(shape=(1,), dtype='int32')
    # embedding
    hairs = Flatten()(Embedding(num_class_hairs, 8,  init='glorot_normal')(hairs_class))    
    eyes = Flatten()(Embedding(num_class_eyes, 8,  init='glorot_normal')(eyes_class))
    # concat_style = merge([hairs, eyes], name='concat_style', mode='concat')
    h = merge([latent, hairs, eyes], mode='concat')
    fake_image = model(h)
    m = Model(input=[latent, hairs_class, eyes_class], output=fake_image)
    m.summary()
    return m

def build_discriminator(image_shape=(64,64,3), num_class = 12):
    kernel_init = 'glorot_uniform'
    discriminator_model = Sequential(name="discriminator_model")
    discriminator_model.add( Conv2D(filters = 64, kernel_size = (4,4), strides = (2,2), padding = "same", data_format = "channels_last", kernel_initializer = kernel_init, input_shape=image_shape))
    discriminator_model.add( LeakyReLU(0.2))
    discriminator_model.add( Conv2D(filters = 128, kernel_size = (4,4), strides = (2,2), padding = "same", data_format = "channels_last", kernel_initializer = kernel_init))
    discriminator_model.add( BatchNormalization(momentum = 0.5))
    discriminator_model.add( LeakyReLU(0.2))
    discriminator_model.add( Conv2D(filters = 256, kernel_size = (4,4), strides = (2,2), padding = "same", data_format = "channels_last", kernel_initializer = kernel_init))
    discriminator_model.add( BatchNormalization(momentum = 0.5))
    discriminator_model.add( LeakyReLU(0.2))
    discriminator_model.add( Conv2D(filters = 512, kernel_size = (4,4), strides = (2,2), padding = "same", data_format = "channels_last", kernel_initializer = kernel_init))
    discriminator_model.add( BatchNormalization(momentum = 0.5))
    discriminator_model.add( LeakyReLU(0.2))
    discriminator_model.add( Flatten())
    dis_input = Input(shape = image_shape)
    features = discriminator_model(dis_input)
    validity = Dense(1, activation="sigmoid")(features)
    label_hair = Dense(num_class_hairs, activation="softmax")(features)
    label_eyes = Dense(num_class_eyes, activation="softmax")(features)
    m = Model(dis_input, [validity, label_hair, label_eyes])
    m.summary()
    return m

def norm_img(img):
    img = (img / 127.5) - 1
    return img

def denorm_img(img):
    img = (img + 1) * 127.5
    return img.astype(np.uint8) 

def gen_noise(batch_size, latent_size):
    #input noise to gen seems to be very important!
    return np.random.normal(0, 1, size=(batch_size,latent_size))

def generate_images(generator, latent_size, img_path):
    noise = gen_noise(16,latent_size)
    hairs = np.full(16, 0, dtype=int)
    for h in range(12):
        hairs[h] = h
    eyes = np.random.randint(11, size=16)
    fake_data_X = generator.predict([noise, hairs, eyes])
    plt.figure(figsize=(4,4))
    gs1 = gridspec.GridSpec(4, 4)
    gs1.update(wspace=0, hspace=0)
    for i in range(16):
        ax1 = plt.subplot(gs1[i])
        ax1.set_aspect('equal')
        image = fake_data_X[i, :,:,:]
        fig = plt.imshow(denorm_img(image))
        plt.axis('off')
        fig.axes.get_xaxis().set_visible(False)
        fig.axes.get_yaxis().set_visible(False)
    plt.tight_layout()
    plt.savefig(img_path,bbox_inches='tight',pad_inches=0)
 
model_id = sys.argv[1]
if not (os.path.exists("model/" + model_id)):
    os.makedirs("model/" + model_id)

model_dir = os.path.join('model', model_id)

np.random.seed(42)
num_steps = 100000
latent_size = 100
num_class_hairs = 12
num_class_eyes = 11
batch_size = 128
half_batch = 64
image_shape = (64,64,3)
img_save_dir = model_dir
save_model_dir = model_dir
log_dir = model_dir


y_hairs, y_eyes, y_index = tag_preprocess('data')
# face_preprocess('data', y_index)
X_data = load_data('data', y_hairs, y_eyes, y_index )
print('X_data: {}, y_hairs: {},  y_eyes"{}'.format(X_data.shape, y_hairs.shape,y_eyes.shape ))
print(y_hairs[0], y_eyes[0], y_index[0])

generator = build_generator()
gen_opt = Adam(lr=0.00015, beta_1=0.5)
generator.compile(loss='binary_crossentropy', optimizer=gen_opt, metrics=['accuracy'])


dis_opt = Adam(lr=0.0002, beta_1=0.5)
losses = ['binary_crossentropy', 'categorical_crossentropy', 'categorical_crossentropy']
alpha = K.variable(1.4)
beta = K.variable(0.8)
gamma = K.variable(0.8)
discriminator = build_discriminator()
discriminator.compile(loss=losses, loss_weights=[1.4,0.8, 0.8], optimizer=dis_opt, metrics=['accuracy'])
discriminator.trainable = False
opt = Adam(lr=0.00015, beta_1=0.5) #same as gen
gen_inp = Input(shape=(latent_size, ))
hairs_inp = Input(shape=(1,), dtype='int32')
eyes_inp = Input(shape=(1,), dtype='int32')
GAN_inp = generator([gen_inp,hairs_inp,eyes_inp])
GAN_opt = discriminator(GAN_inp)
gan = Model(input = [gen_inp,hairs_inp,eyes_inp], output = GAN_opt)
gan.compile(loss = losses, optimizer = opt, metrics=['accuracy'])
gan.summary()

avg_disc_fake_loss = deque([0], maxlen=250)     
avg_disc_real_loss = deque([0], maxlen=250)
avg_GAN_loss = deque([0], maxlen=250)
with open (os.path.join(log_dir,"log.csv"), "w") as f:
    f.write("step,real loss,fake loss, GAN loss\n")

for step in range(0, num_steps): 
    #  train Discriminator
    real_data_X, real_label_hairs,  real_label_eyes = sample_from_dataset(half_batch, image_shape, X_data, y_hairs, y_eyes)

    # label to_categorical
    real_label_hairs_cat = to_categorical(real_label_hairs, num_classes = num_class_hairs )
    real_label_eyes_cat = to_categorical(real_label_eyes, num_classes = num_class_eyes )
    noise = gen_noise(half_batch,latent_size)

    # sample data
    sampled_label_hairs = np.random.randint(0, num_class_hairs, half_batch).reshape(-1, 1)
    sampled_label_eyes = np.random.randint(0, num_class_eyes, half_batch).reshape(-1, 1)
    sampled_label_hairs_cat = to_categorical(sampled_label_hairs, num_classes = num_class_hairs )
    sampled_label_eyes_cat = to_categorical(sampled_label_eyes, num_classes = num_class_eyes )

    fake_data_X = generator.predict([noise, sampled_label_hairs, sampled_label_eyes])
    
    # generate images
    if (step % 100) == 0:
        step_num = str(step).zfill(4)
        generate_images(generator, latent_size, os.path.join(img_save_dir, step_num + "_img.png"))
    
    # valid data
    real_data_Y = np.ones(half_batch) - np.random.random_sample(half_batch)*0.2
    fake_data_Y = np.random.random_sample(half_batch)*0.2
    data_Y = np.concatenate((real_data_Y,fake_data_Y))

    discriminator.trainable = True
    generator.trainable = False
    
    dis_metrics_real = discriminator.train_on_batch(real_data_X,[real_data_Y,real_label_hairs_cat, real_label_eyes_cat ])   #training seperately on real
    dis_metrics_fake = discriminator.train_on_batch(fake_data_X,[fake_data_Y, sampled_label_hairs_cat,sampled_label_eyes_cat ])   #training seperately on fake
    
    avg_disc_fake_loss.append(dis_metrics_fake[0])
    avg_disc_real_loss.append(dis_metrics_real[0])
    

    #  train Generator
    generator.trainable = True
    noise = gen_noise(batch_size,latent_size)
    sampled_label_hairs = np.random.randint(0, num_class_hairs, batch_size).reshape(-1, 1)
    sampled_label_eyes = np.random.randint(0, num_class_eyes, batch_size).reshape(-1, 1)

    sampled_label_hairs_cat = to_categorical(sampled_label_hairs, num_classes = num_class_hairs )
    sampled_label_eyes_cat = to_categorical(sampled_label_eyes, num_classes = num_class_eyes )
    real_data_Y = np.ones(batch_size) - np.random.random_sample(batch_size)*0.2
    
    GAN_X = [noise, sampled_label_hairs, sampled_label_eyes]
    GAN_Y = [real_data_Y, sampled_label_hairs_cat, sampled_label_eyes_cat]
    
    discriminator.trainable = False
    gan_metrics = gan.train_on_batch(GAN_X,GAN_Y)
    avg_GAN_loss.append(gan_metrics[0])

    with open (os.path.join(log_dir,"log.csv"), "a") as f:
        f.write("%d,%f,%f,%f\n" % (step, dis_metrics_real[0], dis_metrics_fake[0],gan_metrics[0]))

    if step % 100 == 0:
        print("Step: ", step)
        print("Discriminator: real/fake loss %f, %f" % (dis_metrics_real[0], dis_metrics_fake[0]))
        print("GAN loss: %f" % (gan_metrics[0]))
        print("Average Discriminator fake loss: %f" % (np.mean(avg_disc_fake_loss)))    
        print("Average Discriminator real loss: %f" % (np.mean(avg_disc_real_loss)))    
        print("Average GAN loss: %f" % (np.mean(avg_GAN_loss)))
        discriminator.trainable = True
        generator.trainable = True
        generator.save(os.path.join(save_model_dir,str(step+1)+"_GENERATOR_weights_and_arch.hdf5"))
    if step % 1000 == 0:
        discriminator.save(os.path.join(save_model_dir,str(step+1)+"_DISCRIMINATOR_weights_and_arch.hdf5"))

