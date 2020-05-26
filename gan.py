from keras.layers import Input, Reshape, Dropout, Dense, Flatten, BatchNormalization, Activation, ZeroPadding2D
from keras.layers.advanced_activations import LeakyReLU
from keras.layers.convolutional import UpSampling2D, Conv2D
from keras.models import Sequential, Model, load_model
from keras.optimizers import Adam

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from tqdm import tqdm
import random

import os

img_extensions=['.jpg','.jpeg','.png']
def is_image(path):
    _, file_extension = os.path.splitext(path)
    return file_extension.lower() in img_extensions

def get_files(db_dir):
    return [os.path.join(db_dir,d,f) for d in next(os.walk(db_dir))[1] for f in next(os.walk(os.path.join(db_dir,d)))[2] if not f.startswith(".") and is_image(f)]

class FaceGenerator:
    def __init__(self,image_width,image_height,channels):
        self.image_width = image_width
        self.image_height = image_height

        self.channels = channels

        self.image_shape = (self.image_width,self.image_height,self.channels)

        self.random_noise_dimension = 100

        optimizer = Adam(0.0002,0.5)

        self.discriminator = self.build_discriminator()
        self.discriminator.compile(loss="binary_crossentropy",optimizer=optimizer,metrics=["accuracy"])
        self.generator = self.build_generator()

        random_input = Input(shape=(self.random_noise_dimension,))

        generated_image = self.generator(random_input)

        self.discriminator.trainable = False

        validity = self.discriminator(generated_image)

        self.combined = Model(random_input,validity)
        self.combined.compile(loss="binary_crossentropy",optimizer=optimizer)

    
    
    def get_training_data(self,datafolder):
        print("Loading training data...")

        training_data = []
        filenames = np.array(get_files(datafolder))
        cnt = 0
        for filename in filenames:
            if cnt < 200000:
                path = filename
                image = Image.open(path)
                image = image.resize((self.image_width,self.image_height),Image.ANTIALIAS)
                pixel_array = np.asarray(image) / 127.5 - 1.
    
                training_data.append(pixel_array)
                
                print(cnt)
                cnt = cnt + 1

        training_data = np.reshape(training_data,(-1,self.image_width,self.image_height,self.channels))
        return training_data


    def build_generator(self):
        model = Sequential()

        model.add(Dense(256*4*4,activation="relu",input_dim=self.random_noise_dimension))
        model.add(Reshape((4,4,256)))

        model.add(UpSampling2D())
        model.add(Conv2D(256,kernel_size=3,padding="same"))
        model.add(BatchNormalization(momentum=0.8))
        model.add(Activation("relu"))

        model.add(UpSampling2D())
        model.add(Conv2D(256,kernel_size=3,padding="same"))
        model.add(BatchNormalization(momentum=0.8))
        model.add(Activation("relu"))

        model.add(UpSampling2D())
        model.add(Conv2D(128,kernel_size=3,padding="same"))
        model.add(BatchNormalization(momentum=0.8))
        model.add(Activation("relu"))

        model.add(UpSampling2D())
        model.add(Conv2D(128,kernel_size=3,padding="same"))
        model.add(BatchNormalization(momentum=0.8))
        model.add(Activation("relu"))

        model.add(Conv2D(self.channels,kernel_size=3,padding="same"))
        model.add(Activation("tanh"))

        model.summary()

        input = Input(shape=(self.random_noise_dimension,))
        generated_image = model(input)

        return Model(input,generated_image)


    def build_discriminator(self):
        model = Sequential()

        model.add(Conv2D(32, kernel_size=3, strides=2, input_shape=self.image_shape, padding="same"))
        model.add(LeakyReLU(alpha=0.2))

        model.add(Dropout(0.25))
        model.add(Conv2D(64, kernel_size=3, strides=2, padding="same"))
        model.add(ZeroPadding2D(padding=((0,1),(0,1))))
        model.add(BatchNormalization(momentum=0.8))
        model.add(LeakyReLU(alpha=0.2))

        model.add(Dropout(0.25))
        model.add(Conv2D(128, kernel_size=3, strides=2, padding="same"))
        model.add(BatchNormalization(momentum=0.8))
        model.add(LeakyReLU(alpha=0.2))

        model.add(Dropout(0.25))
        model.add(Conv2D(256, kernel_size=3, strides=1, padding="same"))
        model.add(BatchNormalization(momentum=0.8))
        model.add(LeakyReLU(alpha=0.2))

        model.add(Dropout(0.25))
        model.add(Conv2D(512, kernel_size=3, strides=1, padding="same"))
        model.add(BatchNormalization(momentum=0.8))
        model.add(LeakyReLU(alpha=0.2))

        model.add(Dropout(0.25))
        model.add(Flatten())
        model.add(Dense(1, activation='sigmoid'))

        model.summary()

        input_image = Input(shape=self.image_shape)

        validity = model(input_image)

        return Model(input_image, validity)

    def train(self, datafolder ,epochs,batch_size,save_images_interval):
        training_data = self.get_training_data(datafolder)

        labels_for_real_images = np.ones((batch_size,1))
        labels_for_generated_images = np.zeros((batch_size,1))

        for epoch in range(epochs):
            
            list_indices = list(range(0,training_data.shape[0]))
            random.shuffle(list_indices)
            
            cnt_step = 0
            for i in range(0, training_data.shape[0], batch_size):
                el_before = i + batch_size
                indices = list_indices[i:el_before]
                
                real_images = training_data[indices]

                random_noise = np.random.normal(0,1,(batch_size,self.random_noise_dimension))
                generated_images = self.generator.predict(random_noise)

                discriminator_loss_real = self.discriminator.train_on_batch(real_images,labels_for_real_images)
                discriminator_loss_generated = self.discriminator.train_on_batch(generated_images,labels_for_generated_images)
                discriminator_loss = 0.5 * np.add(discriminator_loss_real,discriminator_loss_generated)

                generator_loss = self.combined.train_on_batch(random_noise,labels_for_real_images)
                print ("%d / %d [Discriminator loss: %f, acc.: %.2f%%] [Generator loss: %f]" % (epoch, cnt_step, discriminator_loss[0], 100*discriminator_loss[1], generator_loss))
                
                if cnt_step % save_images_interval == 0:
                    self.save_images(epoch, cnt_step)
                
                cnt_step = cnt_step + 1  

        self.generator.save("/home/student/asokolova/gan2/facegenerator.h5")
        self.discriminator.save("/home/student/asokolova/gan2/facediscriminator.h5")


    def save_images(self,epoch, cnt_step):
        rows, columns = 5, 5
        noise = np.random.normal(0, 1, (rows * columns, self.random_noise_dimension))
        generated_images = self.generator.predict(noise)

        generated_images = 0.5 * generated_images + 0.5

        figure, axis = plt.subplots(rows, columns)
        image_count = 0
        for row in range(rows):
            for column in range(columns):
                axis[row,column].imshow(generated_images[image_count, :], cmap='spring')
                axis[row,column].axis('off')
                image_count += 1
        figure.savefig("/home/student/asokolova/gan2/generated_images/generated_%d_%d.png" % (epoch, cnt_step))
        plt.close()

    def generate_single_image(self,model_path,image_save_path):
        noise = np.random.normal(0,1,(1,self.random_noise_dimension))
        model = load_model(model_path)
        generated_image = model.predict(noise)
        #Normalized (-1 to 1) pixel values to the real (0 to 256) pixel values.
        generated_image = (generated_image+1)*127.5
        print(generated_image)
        generated_image = np.reshape(generated_image,self.image_shape)

        image = Image.fromarray(generated_image,"RGB")
        image.save(image_save_path)

if __name__ == '__main__':
    facegenerator = FaceGenerator(64,64,3)
    facegenerator.train(datafolder="/home/datasets/images/vgg_face_dataset/vggface-2/all/",epochs=100, batch_size=64, save_images_interval=100)