import sys, os, csv, time, random
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
import skimage, skimage.io, skimage.transform
import numpy as np
from sklearn.externals import joblib
from keras.layers import Input, Dense, Reshape, Flatten, Dropout, multiply, Activation
from keras.layers import BatchNormalization, Activation, Embedding, ZeroPadding2D
from keras.layers import Conv2D, Conv2DTranspose, Dropout, UpSampling2D, MaxPooling2D
from keras.layers.advanced_activations import LeakyReLU
from keras.models import Sequential, Model, load_model
from keras.optimizers import SGD, Adam, RMSprop
from keras.utils import to_categorical
import keras.layers.merge as merge
import keras.backend as K
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import argparse, json

def parse():
    parser = argparse.ArgumentParser(description="Anime ACGAN")
    parser.add_argument('--uid', type=str, help='training uid', required=True)
    parser.add_argument('--train_path',type=str,  default='data', help='training data path')
    parser.add_argument('--gen_lr', type=float, default=0.00015, help='learning rate of generator')
    parser.add_argument('--dis_lr', type=float, default=0.0002, help='learning rate of discriminator')
    parser.add_argument('--batch_size', type=int, default=128, help='batch size for training')
    parser.add_argument('--epochs', type=int, default=100000, help='epochs for training')
    parser.add_argument('--latent', type=int, default=100, help='latent size')
    try:
        from argument import add_arguments
        parser = add_arguments(parser)
    except:
        pass
    args = parser.parse_args()
    return args

HAIRS = [ 'orange hair', 'white hair', 'aqua hair', 'gray hair','green hair', 'red hair', 'purple hair', 'pink hair','blue hair', 'black hair', 'brown hair', 'blonde hair']
EYES = [  'gray eyes', 'black eyes', 'orange eyes','pink eyes', 'yellow eyes', 'aqua eyes', 'purple eyes','green eyes', 'brown eyes', 'red eyes', 'blue eyes']

def tag_preprocess(data_path):
   
    with open(os.path.join(data_path, 'tags_clean.csv'), 'r') as file:
        lines = csv.reader(file, delimiter=',')
        y_hairs = []
        y_eyes = []
        y_index = []
        for i, line in enumerate(lines):
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
        return y_hairs, y_eyes, y_index


def load_data(data_path, y_hairs, y_eyes, y_index):
    with open(os.path.join(data_path, 'X_data_norepeat.jlib'), 'rb') as file:
        X_data = joblib.load(file)
        return X_data


def norm_img(img):
    img = (img / 127.5) - 1
    return img

def denorm_img(img):
    img = (img + 1) * 127.5
    return img.astype(np.uint8) 

class AnimeACGAN(object):
    ''' Initialize the parameters for the model '''
    def __init__(self, args):
        self.uid = args.uid
        self.train_path = args.train_path
        self.batch_size = args.batch_size
        self.half_batch = self.batch_size // 2
        self.gen_lr = args.gen_lr
        self.dis_lr = args.dis_lr
        self.epochs = args.epochs
        self.latent_size = args.latent
        self.image_shape = (64,64,3)
        self.model_dir = os.path.join('models',  self.uid)
        self.num_class_hairs = 12
        self.num_class_eyes = 11

        if not (os.path.exists(self.model_dir)):
            os.makedirs(self.model_dir)

        self.y_hairs, self.y_eyes, self.y_index = tag_preprocess(self.train_path)
        self.X_data = load_data(self.train_path, self.y_hairs, self.y_eyes, self.y_index )
        print('X_data: {}, y_hairs: {},  y_eyes"{}'.format(self.X_data.shape, self.y_hairs.shape,self.y_eyes.shape ))

        self.generator, self.discriminator, self.gan = self.build_ACGAN()

    def build_generator_model(self):
        kernel_init = 'glorot_uniform'
        model = Sequential(name = 'generator_model')
        model.add(Reshape((1, 1, -1), input_shape=(self.latent_size+16,)))
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
        latent = Input(shape=(self.latent_size, ))
        eyes_class = Input(shape=(1,), dtype='int32')
        hairs_class = Input(shape=(1,), dtype='int32')
        # embedding
        hairs = Flatten()(Embedding(self.num_class_hairs, 8,  init='glorot_normal')(hairs_class))    
        eyes = Flatten()(Embedding(self.num_class_eyes, 8,  init='glorot_normal')(eyes_class))
        # concat_style = merge([hairs, eyes], name='concat_style', mode='concat')
        h = merge([latent, hairs, eyes], mode='concat')
        fake_image = model(h)
        m = Model(input=[latent, hairs_class, eyes_class], output=fake_image)
        # m.summary()
        return m

    def build_discriminator_model(self, num_class = 12):
        kernel_init = 'glorot_uniform'
        discriminator_model = Sequential(name="discriminator_model")
        discriminator_model.add( Conv2D(filters = 64, kernel_size = (4,4), strides = (2,2), padding = "same", data_format = "channels_last", kernel_initializer = kernel_init, input_shape=self.image_shape))
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
        dis_input = Input(shape = self.image_shape)
        features = discriminator_model(dis_input)
        validity = Dense(1, activation="sigmoid")(features)
        label_hair = Dense(self.num_class_hairs, activation="softmax")(features)
        label_eyes = Dense(self.num_class_eyes, activation="softmax")(features)
        m = Model(dis_input, [validity, label_hair, label_eyes])
        # m.summary()
        return m

    def build_ACGAN(self):
        generator = self.build_generator_model()
        gen_opt = Adam(lr = self.gen_lr, beta_1 = 0.5)
        generator.compile(loss = 'binary_crossentropy', optimizer = gen_opt, metrics=['accuracy'])

        discriminator = self.build_discriminator_model()
        dis_opt = Adam(lr = self.dis_lr, beta_1 = 0.5)
        losses = ['binary_crossentropy', 'categorical_crossentropy', 'categorical_crossentropy']
        discriminator.compile(loss=losses, loss_weights=[1.4, 0.8, 0.8], optimizer=dis_opt, metrics=['accuracy'])
        discriminator.trainable = False

        opt = Adam(lr = self.gen_lr, beta_1 = 0.5) 
        gen_inp = Input(shape=(self.latent_size, ))
        hairs_inp = Input(shape=(1,), dtype='int32')
        eyes_inp = Input(shape=(1,), dtype='int32')
        GAN_inp = generator([gen_inp, hairs_inp, eyes_inp])
        GAN_opt = discriminator(GAN_inp)
        gan = Model(input = [gen_inp,hairs_inp,eyes_inp], output = GAN_opt)
        gan.compile(loss = losses, optimizer = opt, metrics=['accuracy'])
        gan.summary()
        return generator, discriminator, gan

    def norm_img(self, img):
        img = (img / 127.5) - 1
        return img

    def denorm_img(self, img):
        img = (img + 1) * 127.5
        return img.astype(np.uint8) 

    def gen_noise(self, batch_size, latent_size):
        return np.random.normal(0, 1, size=(batch_size,latent_size))

    def generate_images(self, generator, img_path):
        noise = self.gen_noise(16, self.latent_size)
        hairs = np.full(16, 0, dtype=int)
        for h in range(self.num_class_hairs):
            hairs[h] = h
        eyes = np.random.randint(self.num_class_eyes, size=16)
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

    def sample_from_dataset(self, batch_size,  X_data, y_hairs, y_eyes):
        sample_dim = (batch_size,) + self.image_shape
        sample = np.empty(sample_dim, dtype=np.float32)
        choice_indices = np.random.choice(len(X_data), batch_size)
        sample = []
        y_hair_label = []
        y_eyes_label = []
        for i in choice_indices:
            x = X_data[i]
            x = norm_img(x)
            y_hair_label.append(y_hairs[i])
            y_eyes_label.append(y_eyes[i])
            sample.append(x)
        sample = np.array(sample)
        y_hair_label = np.array(y_hair_label)
        y_eyes_label = np.array(y_eyes_label)
        return sample, y_hair_label, y_eyes_label

    def train(self):
      
        with open (os.path.join(self.model_dir, 'log.csv'), 'w') as f:
            f.write("step,real loss,fake loss, GAN loss\n")

        for step in range(0, self.epochs): 
            #  train Discriminator
            real_data_X, real_label_hairs, real_label_eyes = self.sample_from_dataset(self.half_batch,  self.X_data, self.y_hairs, self.y_eyes)

            # label to_categorical
            real_label_hairs_cat = to_categorical(real_label_hairs, num_classes = self.num_class_hairs )
            real_label_eyes_cat = to_categorical(real_label_eyes, num_classes = self.num_class_eyes )
            noise = self.gen_noise(self.half_batch, self.latent_size)

            # sample data
            sampled_label_hairs = np.random.randint(0, self.num_class_hairs, self.half_batch).reshape(-1, 1)
            sampled_label_eyes = np.random.randint(0, self.num_class_eyes, self.half_batch).reshape(-1, 1)
            sampled_label_hairs_cat = to_categorical(sampled_label_hairs, num_classes = self.num_class_hairs )
            sampled_label_eyes_cat = to_categorical(sampled_label_eyes, num_classes = self.num_class_eyes )

            fake_data_X = self.generator.predict([noise, sampled_label_hairs, sampled_label_eyes])
            
            # generate images
            if (step % 100) == 0:
                step_num = str(step).zfill(4)
                self.generate_images(self.generator, os.path.join(self.model_dir, step_num + "_img.png"))
            
            # valid data
            real_data_Y = np.ones(self.half_batch) - np.random.random_sample(self.half_batch) * 0.2
            fake_data_Y = np.random.random_sample(self.half_batch)*0.2
            data_Y = np.concatenate((real_data_Y,fake_data_Y))

            self.discriminator.trainable = True
            self.generator.trainable = False

            #training seperately on real
            dis_metrics_real = self.discriminator.train_on_batch(real_data_X,[real_data_Y,real_label_hairs_cat, real_label_eyes_cat ])   
            #training seperately on fake
            dis_metrics_fake = self.discriminator.train_on_batch(fake_data_X,[fake_data_Y, sampled_label_hairs_cat,sampled_label_eyes_cat ])  
                       
            #  train Generator
            self.generator.trainable = True
            noise = self.gen_noise(self.batch_size,self.latent_size)
            sampled_label_hairs = np.random.randint(0, self.num_class_hairs, self.batch_size).reshape(-1, 1)
            sampled_label_eyes = np.random.randint(0, self.num_class_eyes, self.batch_size).reshape(-1, 1)

            sampled_label_hairs_cat = to_categorical(sampled_label_hairs, num_classes = self.num_class_hairs )
            sampled_label_eyes_cat = to_categorical(sampled_label_eyes, num_classes = self.num_class_eyes )
            real_data_Y = np.ones(self.batch_size) - np.random.random_sample(self.batch_size) * 0.2
            
            GAN_X = [noise, sampled_label_hairs, sampled_label_eyes]
            GAN_Y = [real_data_Y, sampled_label_hairs_cat, sampled_label_eyes_cat]
            
            self.discriminator.trainable = False
            gan_metrics = self.gan.train_on_batch(GAN_X,GAN_Y)

            with open (os.path.join( self.model_dir,"log.csv"), "a") as f:
                f.write("%d,%f,%f,%f\n" % (step, dis_metrics_real[0], dis_metrics_fake[0],gan_metrics[0]))

            if step % 100 == 0:
                print("Step: ", step)
                print("Discriminator: real/fake loss %f, %f" % (dis_metrics_real[0], dis_metrics_fake[0]))
                print("GAN loss: %f" % (gan_metrics[0]))
                self.generator.trainable = True
                self.generator.save(os.path.join( self.model_dir, str(step)+ "_GENERATOR.hdf5"))
            
            if step % 1000 == 0:
                self.discriminator.trainable = True
                self.discriminator.save(os.path.join(self.model_dir, str(step)+ "_DISCRIMINATOR.hdf5"))
    def test(self, time_step):
        generator = self.build_generator_model()
        generator.load_weights(os.path.join(self.model_dir, str(time_step) + '_GENERATOR.hdf5'))
        save_test_img_dir = os.path.join(self.model_dir, 'img' + str(time_step))
        if not (os.path.exists(save_test_img_dir)):
            os.makedirs(save_test_img_dir)
        
        for i in range(self.num_class_hairs):
            for j in range(self.num_class_eyes):
                self.generate_test_images(generator, self.latent_size, i, j, save_test_img_dir)    

    def generate_test_images(self, generator, latent_size, hair_color, eye_color, save_dir):
        noise = self.gen_noise(16,latent_size)
        hairs = np.full(16, hair_color, dtype=int)
        eyes = np.full(16,eye_color, dtype=int)
        fake_data_X = generator.predict([noise, hairs, eyes])
        
        print("Displaying generated images")
        plt.figure(figsize=(4,4))
        gs1 = gridspec.GridSpec(4, 4)
        gs1.update(wspace=0, hspace=0)
        for i in range(16):
            #plt.subplot(4, 4, i+1)
            ax1 = plt.subplot(gs1[i])
            ax1.set_aspect('equal')
            image = fake_data_X[i, :,:,:]
            fig = plt.imshow(denorm_img(image))
            plt.axis('off')
            fig.axes.get_xaxis().set_visible(False)
            fig.axes.get_yaxis().set_visible(False)
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir,HAIRS[hair_color] +'_' + EYES[eye_color] +'.jpg'),bbox_inches='tight',pad_inches=0)
        plt.close()

args = parse()
acgan = AnimeACGAN(args)
acgan.train()
# acgan.test(acgan.epochs)

