from model import Generator
from model import Discriminator
from generator import Generator as G2
from discriminator import Discriminator as D2
from torch.autograd import Variable
from torchvision.utils import save_image
import torch
import torch.nn.functional as F
import numpy as np
import os
import time
import datetime

import gc
from glob import glob
from itertools import product
from random import choice

from imageio import imwrite
import tensorflow as tf
from tqdm import tqdm

from logger_cartoonGAN import get_logger

@tf.function
def gram(x):
    shape_x = tf.shape(x)
    b = shape_x[0]
    c = shape_x[3]
    x = tf.reshape(x, [b, -1, c])
    return tf.matmul(tf.transpose(x, [0, 2, 1]), x) / tf.cast((tf.size(x) // b), tf.float32)


# for StarGAN
class Solver(object):
    def __init__(self,rafdb_loader, config):
        """Initialize configurations."""
        "StarGAN."
        # Data loader.
        self.rafdb_loader = rafdb_loader

        # Model configurations.
        self.c_dim = config.c_dim
        self.image_size = config.image_size
        self.g_conv_dim = config.g_conv_dim
        self.d_conv_dim = config.d_conv_dim
        self.g_repeat_num = config.g_repeat_num
        self.d_repeat_num = config.d_repeat_num
        self.lambda_cls = config.lambda_cls
        self.lambda_rec = config.lambda_rec
        self.lambda_gp = config.lambda_gp

        # Training configurations.
#        self.dataset = config.dataset
        self.batch_size_starGAN = config.batch_size_starGAN
        self.num_iters = config.num_iters
        self.num_iters_decay = config.num_iters_decay
        self.g_lr = config.g_lr
        self.d_lr = config.d_lr
        self.n_critic = config.n_critic
        self.beta1 = config.beta1
        self.beta2 = config.beta2
        self.resume_iters = config.resume_iters
#        self.selected_attrs = config.selected_attrs

        # Test configurations.
        self.test_iters = config.test_iters

        # Miscellaneous.
        self.use_tensorboard = config.use_tensorboard
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
#        self.device = torch.device('cpu')

        # Directories.
        self.log_dir = config.log_dir
        self.sample_dir = config.sample_dir
        self.model_save_dir = config.model_save_dir
        #Adding this directory to 
        self.intermediate_dir = config.intermediate_dir        
        self.result_dir_starGAN = config.result_dir_starGAN

        # Step size.
        self.log_step = config.log_step
        self.sample_step = config.sample_step
        self.model_save_step = config.model_save_step
        self.lr_update_step = config.lr_update_step

        # Build the model and tensorboard.
        self.build_model()
        if self.use_tensorboard:
            self.build_tensorboard()
        
        "CartoonGAN"
        self.debug = config.debug
        self.ascii = os.name == "nt"
        self.dataset_name = config.dataset_name
        self.light = config.light
        self.source_domain = config.source_domain
        self.target_domain = config.target_domain
        self.gan_type = config.gan_type
        self.epochs = config.epochs
        self.input_size = config.input_size
        self.multi_scale = config.multi_scale
        self.batch_size_cartoonGAN = config.batch_size_cartoonGAN
        self.sample_size = config.sample_size
        self.reporting_steps = config.reporting_steps
        self.content_lambda = float(config.content_lambda)
        self.style_lambda = float(config.style_lambda)
        self.g_adv_lambda =config. g_adv_lambda
        self.d_adv_lambda = config.d_adv_lambda
        self.generator_lr = config.generator_lr
        self.discriminator_lr = config.discriminator_lr
        self.data_dir =config.data_dir
        self.log_dir_cartoonGAN = config.log_dir_cartoonGAN
        self.result_dir_cartoonGAN = config.result_dir_cartoonGAN
        self.checkpoint_dir = config.checkpoint_dir
        self.generator_checkpoint_prefix = config.generator_checkpoint_prefix
        self.discriminator_checkpoint_prefix = config.discriminator_checkpoint_prefix
        self.pretrain_checkpoint_prefix = config.pretrain_checkpoint_prefix
        self.pretrain_model_dir = config.pretrain_model_dir
        self.model_dir = config.model_dir
        self.disable_sampling = config.disable_sampling
        self.ignore_vgg = config.ignore_vgg
        self.pretrain_learning_rate = config.pretrain_learning_rate
        self.pretrain_epochs = config.pretrain_epochs
        self.pretrain_saving_epochs = config.pretrain_saving_epochs
        self.pretrain_reporting_steps = config.pretrain_reporting_steps
        self.pretrain_generator_name = config.pretrain_generator_name
        self.generator_name = config.generator_name
        self.discriminator_name = config.discriminator_name

        self.logger_cartoonGAN = get_logger("Solver", debug=False)
        # NOTE: just minimal demonstration of multi-scale training
        self.sizes = [self.input_size - 32, self.input_size, self.input_size + 32]

        if not self.ignore_vgg:
            self.logger_cartoonGAN.info("Setting up VGG19 for computing content loss...")
            from tensorflow.keras.applications import VGG19
            from tensorflow.keras.layers import Conv2D
            input_shape = (self.input_size, self.input_size, 3)
            # download model using kwarg weights="imagenet"
            base_model = VGG19(weights="imagenet", include_top=False, input_shape=input_shape)
            tmp_vgg_output = base_model.get_layer("block4_conv3").output
            tmp_vgg_output = Conv2D(512, (3, 3), activation='linear', padding='same',
                                    name='block4_conv4')(tmp_vgg_output)
            self.vgg = tf.keras.Model(inputs=base_model.input, outputs=tmp_vgg_output)
            self.vgg.load_weights(os.path.expanduser(os.path.join(
                "~", ".keras", "models",
                "vgg19_weights_tf_dim_ordering_tf_kernels_notop.h5")), by_name=True)
        else:
            self.logger_cartoonGAN.info("VGG19 will not be used. "
                             "Content loss will simply imply pixel-wise difference.")
            self.vgg = None

        self.logger_cartoonGAN.info(f"Setting up objective functions and metrics using {self.gan_type}...")
        self.mae = tf.keras.losses.MeanAbsoluteError()
        self.generator_loss_object = tf.keras.losses.BinaryCrossentropy(from_logits=True)
        if self.gan_type == "gan":
            self.discriminator_loss_object = tf.keras.losses.BinaryCrossentropy(
                from_logits=True)
        elif self.gan_type == "lsgan":
            self.discriminator_loss_object = tf.keras.losses.MeanSquaredError()
        else:
            wrong_msg = f"Non-recognized 'gan_type': {self.gan_type}"
            self.logger_cartoonGAN.critical(wrong_msg)
            raise ValueError(wrong_msg)

        self.g_total_loss_metric = tf.keras.metrics.Mean("g_total_loss", dtype=tf.float32)
        self.g_adv_loss_metric = tf.keras.metrics.Mean("g_adversarial_loss", dtype=tf.float32)
        if self.content_lambda != 0.:
            self.content_loss_metric = tf.keras.metrics.Mean("content_loss", dtype=tf.float32)
        if self.style_lambda != 0.:
            self.style_loss_metric = tf.keras.metrics.Mean("style_loss", dtype=tf.float32)
        self.d_total_loss_metric = tf.keras.metrics.Mean("d_total_loss", dtype=tf.float32)
        self.d_real_loss_metric = tf.keras.metrics.Mean("d_real_loss", dtype=tf.float32)
        self.d_fake_loss_metric = tf.keras.metrics.Mean("d_fake_loss", dtype=tf.float32)
        self.d_smooth_loss_metric = tf.keras.metrics.Mean("d_smooth_loss", dtype=tf.float32)

        self.metric_and_names = [
            (self.g_total_loss_metric, "g_total_loss"),
            (self.g_adv_loss_metric, "g_adversarial_loss"),
            (self.d_total_loss_metric, "d_total_loss"),
            (self.d_real_loss_metric, "d_real_loss"),
            (self.d_fake_loss_metric, "d_fake_loss"),
            (self.d_smooth_loss_metric, "d_smooth_loss"),
        ]
        if self.content_lambda != 0.:
            self.metric_and_names.append((self.content_loss_metric, "content_loss"))
        if self.style_lambda != 0.:
            self.metric_and_names.append((self.style_loss_metric, "style_loss"))

        self.logger_cartoonGAN.info("Setting up checkpoint paths...")
        self.pretrain_checkpoint_prefix = os.path.join(
            self.checkpoint_dir, "pretrain", self.pretrain_checkpoint_prefix)
        self.generator_checkpoint_dir = os.path.join(
            self.checkpoint_dir, self.generator_checkpoint_prefix)
        self.generator_checkpoint_prefix = os.path.join(
            self.generator_checkpoint_dir, self.generator_checkpoint_prefix)
        self.discriminator_checkpoint_dir = os.path.join(
            self.checkpoint_dir, self.discriminator_checkpoint_prefix)
        self.discriminator_checkpoint_prefix = os.path.join(
            self.discriminator_checkpoint_dir, self.discriminator_checkpoint_prefix)

    #################################################################################################
    "functions for StarGAN"
    def build_model(self):
        """Create a generator and a discriminator."""
        self.G = Generator(self.g_conv_dim, self.c_dim, self.g_repeat_num)
        self.D = Discriminator(self.image_size, self.d_conv_dim, self.c_dim, self.d_repeat_num) 

        self.g_optimizer = torch.optim.Adam(self.G.parameters(), self.g_lr, [self.beta1, self.beta2])
        self.d_optimizer = torch.optim.Adam(self.D.parameters(), self.d_lr, [self.beta1, self.beta2])
        self.print_network(self.G, 'G')
        self.print_network(self.D, 'D')
            
        self.G.to(self.device)
        self.D.to(self.device)

    def print_network(self, model, name):
        """Print out the network information."""
        num_params = 0
        for p in model.parameters():
            num_params += p.numel()
        print(model)
        print(name)
        print("The number of parameters: {}".format(num_params))

    def restore_model(self, resume_iters):
        """Restore the trained generator and discriminator."""
        print('Loading the trained models from step {}...'.format(resume_iters))
        G_path = os.path.join(self.model_save_dir, '{}-G.ckpt'.format(resume_iters))
        D_path = os.path.join(self.model_save_dir, '{}-D.ckpt'.format(resume_iters))
        self.G.load_state_dict(torch.load(G_path, map_location=lambda storage, loc: storage))
        self.D.load_state_dict(torch.load(D_path, map_location=lambda storage, loc: storage))

    def build_tensorboard(self):
        """Build a tensorboard logger."""
        from logger import Logger
        self.logger = Logger(self.log_dir)

    def update_lr(self, g_lr, d_lr):
        """Decay learning rates of the generator and discriminator."""
        for param_group in self.g_optimizer.param_groups:
            param_group['lr'] = g_lr
        for param_group in self.d_optimizer.param_groups:
            param_group['lr'] = d_lr

    def reset_grad(self):
        """Reset the gradient buffers."""
        self.g_optimizer.zero_grad()
        self.d_optimizer.zero_grad()

    def denorm(self, x):
        """Convert the range from [-1, 1] to [0, 1]."""
        out = (x + 1) / 2
        return out.clamp_(0, 1)

    def gradient_penalty(self, y, x):
        """Compute gradient penalty: (L2_norm(dy/dx) - 1)**2."""
        weight = torch.ones(y.size()).to(self.device)
        dydx = torch.autograd.grad(outputs=y,
                                   inputs=x,
                                   grad_outputs=weight,
                                   retain_graph=True,
                                   create_graph=True,
                                   only_inputs=True)[0]

        dydx = dydx.view(dydx.size(0), -1)
        dydx_l2norm = torch.sqrt(torch.sum(dydx**2, dim=1))
        return torch.mean((dydx_l2norm-1)**2)

    def label2onehot(self, labels, dim):
        """Convert label indices to one-hot vectors."""
        batch_size = labels.size(0)
        out = torch.zeros(batch_size, dim)
        out[np.arange(batch_size), labels.long()] = 1
        return out

    def create_labels(self, c_org, c_dim=5):
        """Generate target domain labels for debugging and testing."""
        c_trg_list = []
        for i in range(c_dim):
            c_trg = self.label2onehot(torch.ones(c_org.size(0))*i, c_dim)

            c_trg_list.append(c_trg.to(self.device))
        return c_trg_list

    def classification_loss(self, logit, target):
        """Compute binary or softmax cross entropy loss."""
        return F.cross_entropy(logit, target)

    #################################################################################################
    "functions for CartoonGAN"
    def _save_generated_images(self, batch_x, image_name, nrow=2, ncol=4, dir_):
        # NOTE: 0 <= batch_x <= 1, float32, numpy.ndarray
        if not isinstance(batch_x, np.ndarray):
            batch_x = batch_x.numpy()
        n, h, w, c = batch_x.shape
        out_arr = np.zeros([h * nrow, w * ncol, 3], dtype=np.uint8)
        for (i, j), k in zip(product(range(nrow), range(ncol)), range(n)):
            out_arr[(h * i):(h * (i+1)), (w * j):(w * (j+1))] = batch_x[k]
        if not os.path.isdir(self.dir_):
            os.makedirs(self.dir_)
        imwrite(os.path.join(self.dir_, image_name), out_arr)
        gc.collect()
        return out_arr

    @tf.function
    def random_resize(self, x):
        size = choice(self.sizes)
        return tf.image.resize(x, (size, size))

    @tf.function
    def image_processing(self, filename, is_train=True):
        crop_size = self.input_size
        if self.multi_scale and is_train:
            crop_size += 32
        x = tf.io.read_file(filename)
        x = tf.image.decode_jpeg(x, channels=3)
        if is_train:
            sizes = tf.cast(
                crop_size * tf.random.uniform([2], 0.9, 1.1), tf.int32)
            shape = tf.shape(x)[:2]
            sizes = tf.minimum(sizes, shape)
            x = tf.image.random_crop(x, (sizes[0], sizes[1], 3))
            x = tf.image.random_flip_left_right(x)
        x = tf.image.resize(x, (crop_size, crop_size))
        img = tf.cast(x, tf.float32) / 127.5 - 1
        return img

    def get_dataset(self, dataset_name, domain, _type, batch_size):
        files = glob(os.path.join(self.data_dir, dataset_name, f"{_type}{domain}", "*"))
        num_images = len(files)
        self.logger_cartoonGAN.info(
            f"Found {num_images} domain{domain} images in {_type}{domain} folder."
        )
        ds = tf.data.Dataset.from_tensor_slices(files)
        ds = ds.apply(tf.data.experimental.shuffle_and_repeat(num_images))

        def fn(fname):
            if self.multi_scale:
                return self.random_resize(self.image_processing(fname, True))
            else:
                return self.image_processing(fname, True)

        ds = ds.apply(tf.data.experimental.map_and_batch(fn, batch_size))
        steps = int(np.ceil(num_images/batch_size))
        # user iter(ds) to avoid generating iterator every epoch
        return iter(ds), steps

    @tf.function
    def pass_to_vgg(self, tensor):
        # NOTE: self.vgg should be fixed
        if self.vgg is not None:
            tensor = self.vgg(tensor)
        return tensor

    @tf.function
    def content_loss(self, input_images, generated_images):
        return self.mae(input_images, generated_images)

    @tf.function
    def style_loss(self, input_images, generated_images):
        input_images = gram(input_images)
        generated_images = gram(generated_images)
        return self.mae(input_images, generated_images)

    @tf.function
    def discriminator_loss(self, real_output, fake_output, smooth_output):
        real_loss = self.discriminator_loss_object(tf.ones_like(real_output), real_output)
        fake_loss = self.discriminator_loss_object(tf.zeros_like(fake_output), fake_output)
        smooth_loss = self.discriminator_loss_object(
            tf.zeros_like(smooth_output), smooth_output)
        total_loss = real_loss + fake_loss + smooth_loss
        return real_loss, fake_loss, smooth_loss, total_loss

    @tf.function
    def generator_adversarial_loss(self, fake_output):
        return self.generator_loss_object(tf.ones_like(fake_output), fake_output)

    @tf.function
    def pretrain_step(self, input_images, generator, optimizer):

        with tf.GradientTape() as tape:
            generated_images = generator(input_images, training=True)
            c_loss = self.content_lambda * self.content_loss(
                self.pass_to_vgg(input_images), self.pass_to_vgg(generated_images))

        gradients = tape.gradient(c_loss, generator.trainable_variables)
        optimizer.apply_gradients(zip(gradients, generator.trainable_variables))

        self.content_loss_metric(c_loss)

    @tf.function
    def train_step(self, source_images, target_images, smooth_images,
                   generator, discriminator, g_optimizer, d_optimizer):

        with tf.GradientTape() as g_tape, tf.GradientTape() as d_tape:
            real_output = discriminator(target_images, training=True)
            generated_images = generator(source_images, training=True)
            fake_output = discriminator(generated_images, training=True)
            smooth_out = discriminator(smooth_images, training=True)
            d_real_loss, d_fake_loss, d_smooth_loss, d_total_loss = \
                self.discriminator_loss(real_output, fake_output, smooth_out)

            g_adv_loss = self.g_adv_lambda * self.generator_adversarial_loss(fake_output)
            g_total_loss = g_adv_loss
            # NOTE: self.*_lambdas are fixed
            if self.content_lambda != 0. or self.style_lambda != 0.:
                vgg_generated_images = self.pass_to_vgg(generated_images)
                if self.content_lambda != 0.:
                    c_loss = self.content_lambda * self.content_loss(
                        self.pass_to_vgg(source_images), vgg_generated_images)
                    g_total_loss = g_total_loss + c_loss
                if self.style_lambda != 0.:
                    s_loss = self.style_lambda * self.style_loss(
                        self.pass_to_vgg(target_images[:vgg_generated_images.shape[0]]),
                        vgg_generated_images)
                    g_total_loss = g_total_loss + s_loss

        d_grads = d_tape.gradient(d_total_loss, discriminator.trainable_variables)
        g_grads = g_tape.gradient(g_total_loss, generator.trainable_variables)

        d_optimizer.apply_gradients(zip(d_grads, discriminator.trainable_variables))
        g_optimizer.apply_gradients(zip(g_grads, generator.trainable_variables))

        self.g_total_loss_metric(g_total_loss)
        self.g_adv_loss_metric(g_adv_loss)
        if self.content_lambda != 0.:
            self.content_loss_metric(c_loss)
        if self.style_lambda != 0.:
            self.style_loss_metric(s_loss)
        self.d_total_loss_metric(d_total_loss)
        self.d_real_loss_metric(d_real_loss)
        self.d_fake_loss_metric(d_fake_loss)
        self.d_smooth_loss_metric(d_smooth_loss)

    def pretrain_generator(self):
        summary_writer = tf.summary.create_file_writer(os.path.join(self.log_dir_cartoonGAN, "pretrain"))
        self.logger_cartoonGAN.info(f"Starting to pretrain generator with {self.pretrain_epochs} epochs...")
        self.logger_cartoonGAN.info(
            f"Building `{self.dataset_name}` dataset with domain `{self.source_domain}`..."
        )
        dataset, steps_per_epoch = self.get_dataset(dataset_name=self.dataset_name,
                                                    domain=self.source_domain,
                                                    _type="train",
                                                    batch_size=self.batch_size_cartoonGAN)
        if self.multi_scale:
            self.logger_cartoonGAN.info(f"Initializing generator with "
                             f"batch_size_cartoonGAN: {self.batch_size_cartoonGAN}, input_size: multi-scale...")
        else:
            self.logger_cartoonGAN.info(f"Initializing generator with "
                             f"batch_size_cartoonGAN: {self.batch_size_cartoonGAN}, input_size: {self.input_size}...")
        generator = G2(base_filters=2 if self.debug else 64, light=self.light)
        generator(tf.keras.Input(
            shape=(self.input_size, self.input_size, 3),
            batch_size=self.batch_size_cartoonGAN))
        generator.summary()

        self.logger_cartoonGAN.info("Setting up optimizer to update generator's parameters...")
        optimizer = tf.keras.optimizers.Adam(
            learning_rate=self.pretrain_learning_rate,
            beta_1=0.5)

        self.logger_cartoonGAN.info(f"Try restoring checkpoint: `{self.pretrain_checkpoint_prefix}`...")
        try:
            checkpoint = tf.train.Checkpoint(generator=generator)
            status = checkpoint.restore(tf.train.latest_checkpoint(
                os.path.join(self.checkpoint_dir, "pretrain")))
            status.assert_consumed()

            self.logger_cartoonGAN.info(f"Previous checkpoints has been restored.")
            trained_epochs = checkpoint.save_counter.numpy()
            epochs = self.pretrain_epochs - trained_epochs
            if epochs <= 0:
                self.logger_cartoonGAN.info(f"Already trained {trained_epochs} epochs. "
                                 "Set a larger `pretrain_epochs`...")
                return
            else:
                self.logger_cartoonGAN.info(f"Already trained {trained_epochs} epochs, "
                                 f"{epochs} epochs left to be trained...")
        except AssertionError:
            self.logger_cartoonGAN.info(f"Checkpoint is not found, "
                             f"training from scratch with {self.pretrain_epochs} epochs...")
            trained_epochs = 0
            epochs = self.pretrain_epochs

        if not self.disable_sampling:
            val_files = glob(os.path.join(
                self.data_dir, self.dataset_name, f"test{self.source_domain}", "*"))
            val_real_batch = tf.map_fn(
                lambda fname: self.image_processing(fname, False),
                tf.constant(val_files), tf.float32, back_prop=False)
            real_batch = next(dataset)
            while real_batch.shape[0] < self.sample_size:
                real_batch = tf.concat((real_batch, next(dataset)), 0)
            real_batch = real_batch[:self.sample_size]
            with summary_writer.as_default():
                img = np.expand_dims(self._save_generated_images(
                    tf.cast((real_batch + 1) * 127.5, tf.uint8),
                    image_name="pretrain_sample_images.png"), 0,result_dir_cartoonGAN)
                tf.summary.image("pretrain_sample_images", img, step=0)
                img = np.expand_dims(self._save_generated_images(
                    tf.cast((val_real_batch + 1) * 127.5, tf.uint8),
                    image_name="pretrain_val_sample_images.png",result_dir_cartoonGAN), 0,)
                tf.summary.image("pretrain_val_sample_images", img, step=0)
            gc.collect()
        else:
            self.logger_cartoonGAN.info("Proceeding pretraining without sample images...")

        self.logger_cartoonGAN.info("Starting pre-training loop, "
                         "setting up summary writer to record progress on TensorBoard...")

        for epoch in range(epochs):
            epoch_idx = trained_epochs + epoch + 1

            for step in tqdm(
                    range(1, steps_per_epoch + 1),
                    desc=f"Pretrain Epoch {epoch + 1}/{epochs}"):
                # NOTE: not following official "for img in dataset" example
                #       since it generates new iterator every epoch and can
                #       hardly be garbage-collected by python
                image_batch = dataset.next()
                self.pretrain_step(image_batch, generator, optimizer)

                if step % self.pretrain_reporting_steps == 0:

                    global_step = (epoch_idx - 1) * steps_per_epoch + step
                    with summary_writer.as_default():
                        tf.summary.scalar('content_loss',
                                          self.content_loss_metric.result(),
                                          step=global_step)
                        if not self.disable_sampling:
                            fake_batch = tf.cast(
                                (generator(real_batch, training=False) + 1) * 127.5, tf.uint8)
                            img = np.expand_dims(self._save_generated_images(
                                    fake_batch,
                                    image_name=(f"pretrain_generated_images_at_epoch_{epoch_idx}"
                                                f"_step_{step}.png"),result_dir_cartoonGAN),
                                    0,
                            )
                            tf.summary.image('pretrain_generated_images', img, step=global_step)
                    self.content_loss_metric.reset_states()
            with summary_writer.as_default():
                if not self.disable_sampling:
                    val_fake_batch = tf.cast(
                        (generator(val_real_batch, training=False) + 1) * 127.5, tf.uint8)
                    img = np.expand_dims(self._save_generated_images(
                            val_fake_batch,
                            image_name=("pretrain_val_generated_images_at_epoch_"
                                        f"{epoch_idx}_step_{step}.png"),result_dir_cartoonGAN),
                            0,
                    )
                    tf.summary.image('pretrain_val_generated_images', img, step=epoch)

            if epoch % self.pretrain_saving_epochs == 0:
                self.logger_cartoonGAN.info(f"Saving checkpoints after epoch {epoch_idx} ended...")
                checkpoint.save(file_prefix=self.pretrain_checkpoint_prefix)
            gc.collect()
        del dataset
        gc.collect()

    #############################################################################################################
    "Train StarGAN and CartoonGAN in a stacked model"
    def train(self):
        """Train StarGAN setting"""
        # Set data loader.
        data_loader = self.rafdb_loader
    
        # Fetch fixed inputs for debugging.
        data_iter = iter(data_loader)
        x_fixed_db, c_org_db = next(data_iter)
        x_fixed_db = x_fixed_db.to(self.device)
        c_fixed_list_db = self.create_labels(c_org_db, self.c_dim)
        
        # Get 15*16*5 faked data for second gan
        for i in range(16):
            data_iter = iter(data_loader)
            x_fixed, c_org = next(data_iter)
            x_fixed = x_fixed.to(self.device)
            if i == 0:
                x_fixed_all = x_fixed
                c_org_all = c_org
            else:
                x_fixed_all = torch.cat((x_fixed_all,x_fixed),0)
                c_org_all = torch.cat((c_org_all,c_org),0)
        c_fixed_list = self.create_labels(c_org_all, self.c_dim)
        
    
        # Learning rate cache for decaying.
        g_lr = self.g_lr
        d_lr = self.d_lr
    
        # Start training from scratch or resume training.
        start_iters = 0
        if self.resume_iters:
            start_iters = self.resume_iters
            self.restore_model(self.resume_iters)
    
    
        "Train CartoonGAN setting"
        self.logger_cartoonGAN.info("Setting up summary writer to record progress on TensorBoard...")
        summary_writer = tf.summary.create_file_writer(self.log_dir_cartoonGAN)
        self.logger_cartoonGAN.info(
            f"Starting adversarial training with {self.epochs} epochs, "
            f"batch size: {self.batch_size_cartoonGAN}..."
        )
        self.logger_cartoonGAN.info(f"Building `{self.dataset_name}` "
                         "datasets for source/target/smooth domains...")
        
        ds_source, steps_per_epoch = self.get_dataset(dataset_name=self.dataset_name,
                                                      domain=self.source_domain,
                                                      _type="train",
                                                      batch_size=self.batch_size_cartoonGAN)
        ds_target, _ = self.get_dataset(dataset_name=self.dataset_name,
                                        domain=self.target_domain,
                                        _type="train",
                                        batch_size=self.batch_size_cartoonGAN)
        ds_smooth, _ = self.get_dataset(dataset_name=self.dataset_name,
                                        domain=f"{self.target_domain}_smooth",
                                        _type="train",
                                        batch_size=self.batch_size_cartoonGAN)
        self.logger_cartoonGAN.info("Setting up optimizer to update generator and discriminator...")
        g_optimizer = tf.keras.optimizers.Adam(learning_rate=self.generator_lr, beta_1=.5)
        d_optimizer = tf.keras.optimizers.Adam(learning_rate=self.discriminator_lr, beta_1=.5)
        if self.multi_scale:
            self.logger_cartoonGAN.info(f"Initializing generator with "
                             f"batch_size: {self.batch_size_cartoonGAN}, input_size: multi-scale...")
        else:
            self.logger_cartoonGAN.info(f"Initializing generator with "
                             f"batch_size: {self.batch_size_cartoonGAN}, input_size: {self.input_size}...")
        generator = G2(base_filters=2 if self.debug else 64, light=self.light)
        generator(tf.keras.Input(
            shape=(self.input_size, self.input_size, 3),
            batch_size=self.batch_size_cartoonGAN))
    
        self.logger_cartoonGAN.info(f"Searching existing checkpoints: `{self.generator_checkpoint_prefix}`...")
        try:
            g_checkpoint = tf.train.Checkpoint(generator=generator)
            g_checkpoint.restore(
                tf.train.latest_checkpoint(
                    self.generator_checkpoint_dir)).assert_existing_objects_matched()
            self.logger_cartoonGAN.info(f"Previous checkpoints has been restored.")
            trained_epochs = g_checkpoint.save_counter.numpy()
            epochs = self.epochs - trained_epochs
            if epochs <= 0:
                self.logger_cartoonGAN.info(f"Already trained {trained_epochs} epochs. "
                                 "Set a larger `epochs`...")
                return
            else:
                self.logger_cartoonGAN.info(f"Already trained {trained_epochs} epochs, "
                                 f"{epochs} epochs left to be trained...")
        except AssertionError as e:
            self.logger_cartoonGAN.warning(e)
            self.logger_cartoonGAN.warning(
                "Previous checkpoints are not found, trying to load checkpoints from pretraining..."
            )
    
            try:
                g_checkpoint = tf.train.Checkpoint(generator=generator)
                g_checkpoint.restore(tf.train.latest_checkpoint(
                    os.path.join(
                        self.checkpoint_dir, "pretrain"))).assert_existing_objects_matched()
                self.logger_cartoonGAN.info("Successfully loaded "
                                 f"`{self.pretrain_checkpoint_prefix}`...")
            except AssertionError:
                self.logger_cartoonGAN.warning("specified pretrained checkpoint is not found, "
                                    "training from scratch...")
    
            trained_epochs = 0
            epochs = self.epochs
    
        if self.multi_scale:
            self.logger_cartoonGAN.info(f"Initializing discriminator with "
                             f"batch_size: {self.batch_size_cartoonGAN}, input_size: multi-scale...")
        else:
            self.logger_cartoonGAN.info(f"Initializing discriminator with "
                             f"batch_size: {self.batch_size_cartoonGAN}, input_size: {self.input_size}...")
        if self.debug:
            d_base_filters = 2
        elif self.light:
            d_base_filters = 24
        else:
            d_base_filters = 32
        d = D2(base_filters=d_base_filters)
        d(tf.keras.Input(
            shape=(self.input_size, self.input_size, 3),
            batch_size=self.batch_size_cartoonGAN))
    
        self.logger_cartoonGAN.info("Searching existing checkpoints: "
                         f"`{self.discriminator_checkpoint_prefix}`...")
        try:
            d_checkpoint = tf.train.Checkpoint(d=d)
            d_checkpoint.restore(
                tf.train.latest_checkpoint(
                    self.discriminator_checkpoint_dir)).assert_existing_objects_matched()
            self.logger_cartoonGAN.info(f"Previous checkpoints has been restored.")
        except AssertionError:
            self.logger_cartoonGAN.info("specified checkpoint is not found, training from scratch...")
    
        if not self.disable_sampling:
            val_files = glob(os.path.join(
                self.data_dir, self.dataset_name, f"test{self.source_domain}", "*"))
            val_real_batch = tf.map_fn(
                lambda fname: self.image_processing(fname, False),
                tf.constant(val_files), tf.float32, back_prop=False)
            real_batch = next(ds_source)
            while real_batch.shape[0] < self.sample_size:
                real_batch = tf.concat((real_batch, next(ds_source)), 0)
            real_batch = real_batch[:self.sample_size]
            with summary_writer.as_default():
                img = np.expand_dims(self._save_generated_images(
                    tf.cast((real_batch + 1) * 127.5, tf.uint8),
                    image_name="gan_sample_images.png",result_dir_cartoonGAN), 0,)
                tf.summary.image("gan_sample_images", img, step=0)
                img = np.expand_dims(self._save_generated_images(
                    tf.cast((val_real_batch + 1) * 127.5, tf.uint8),
                    image_name="gan_val_sample_images.png",result_dir_cartoonGAN), 0,)
                tf.summary.image("gan_val_sample_images", img, step=0)
            gc.collect()
        else:
            self.logger_cartoonGAN.info("Proceeding training without sample images...")
    
        self.logger_cartoonGAN.info("Starting training loop...")
    
        self.logger_cartoonGAN.info(f"Number of trained epochs: {trained_epochs}, "
                         f"epochs to be trained: {epochs}, "
                         f"batch size: {self.batch_size_cartoonGAN}")
    
        "Start Training!"
        # Start training.
        print('Start training...')
        start_time = time.time()
        for i in range(start_iters, self.num_iters):
            
            "StarGAN"
            # =================================================================================== #
            #                             1. Preprocess input data                                #
            # =================================================================================== #
    
            # Fetch real images and labels.
            try:
                x_real, label_org = next(data_iter)
            except:
                data_iter = iter(data_loader)
                x_real, label_org = next(data_iter)
    
            # Generate target domain labels randomly.
            rand_idx = torch.randperm(label_org.size(0))
            label_trg = label_org[rand_idx]
    
            c_org = self.label2onehot(label_org, self.c_dim)
            c_trg = self.label2onehot(label_trg, self.c_dim)
    
            x_real = x_real.to(self.device)           # Input images.
            c_org = c_org.to(self.device)             # Original domain labels.
            c_trg = c_trg.to(self.device)             # Target domain labels.
            label_org = label_org.to(self.device)     # Labels for computing classification loss.
            label_trg = label_trg.to(self.device)     # Labels for computing classification loss.
    
            # =================================================================================== #
            #                             2. Train the discriminator                              #
            # =================================================================================== #
    
            # Compute loss with real images.
            out_src, out_cls = self.D(x_real)
            d_loss_real = - torch.mean(out_src)
            d_loss_cls = self.classification_loss(out_cls, label_org)
    
            # Compute loss with fake images.
            x_fake = self.G(x_real, c_trg)
            out_src, out_cls = self.D(x_fake.detach())
            d_loss_fake = torch.mean(out_src)
    
            # Compute loss for gradient penalty.
            alpha = torch.rand(x_real.size(0), 1, 1, 1).to(self.device)
            x_hat = (alpha * x_real.data + (1 - alpha) * x_fake.data).requires_grad_(True)
            out_src, _ = self.D(x_hat)
            d_loss_gp = self.gradient_penalty(out_src, x_hat)
    
            # Backward and optimize.
            d_loss = d_loss_real + d_loss_fake + self.lambda_cls * d_loss_cls + self.lambda_gp * d_loss_gp
            self.reset_grad()
            d_loss.backward()
            self.d_optimizer.step()
    
            # Logging.
            loss = {}
            loss['D/loss_real'] = d_loss_real.item()
            loss['D/loss_fake'] = d_loss_fake.item()
            loss['D/loss_cls'] = d_loss_cls.item()
            loss['D/loss_gp'] = d_loss_gp.item()
            
            # =================================================================================== #
            #                               3. Train the generator                                #
            # =================================================================================== #
            
            if (i+1) % self.n_critic == 0:
                # Original-to-target domain.
                x_fake = self.G(x_real, c_trg)
                out_src, out_cls = self.D(x_fake)
                g_loss_fake = - torch.mean(out_src)
                g_loss_cls = self.classification_loss(out_cls, label_trg)
    
                # Target-to-original domain.
                x_reconst = self.G(x_fake, c_org)
                g_loss_rec = torch.mean(torch.abs(x_real - x_reconst))
    
                # Backward and optimize.
                g_loss = g_loss_fake + self.lambda_rec * g_loss_rec + self.lambda_cls * g_loss_cls
                self.reset_grad()
                g_loss.backward()
                self.g_optimizer.step()
    
                # Logging.
                loss['G/loss_fake'] = g_loss_fake.item()
                loss['G/loss_rec'] = g_loss_rec.item()
                loss['G/loss_cls'] = g_loss_cls.item()
    
            # =================================================================================== #
            #                                 4. Miscellaneous                                    #
            # =================================================================================== #
    
            # Print out training information.
            if (i+1) % self.log_step == 0:
                et = time.time() - start_time
                et = str(datetime.timedelta(seconds=et))[:-7]
                log = "Elapsed [{}], Iteration [{}/{}]".format(et, i+1, self.num_iters)
                for tag, value in loss.items():
                    log += ", {}: {:.4f}".format(tag, value)
                print(log)
    
                if self.use_tensorboard:
                    for tag, value in loss.items():
                        self.logger.scalar_summary(tag, value, i+1)
                        self.logger.writer.flush()
    
            # Translate fixed images for debugging.
            if (i+1) % self.sample_step == 0:
                with torch.no_grad():
                    x_fake_list = [x_fixed_db]
                    for c_fixed in c_fixed_list_db:
                        x_fake_list.append(self.G(x_fixed_db, c_fixed))
                    x_concat = torch.cat(x_fake_list, dim=3)
                    sample_path = os.path.join(self.sample_dir, '{}-images.jpg'.format(i+1))
                    save_image(self.denorm(x_concat.data.cpu()), sample_path, nrow=1, padding=0)
                    print('Saved real and fake images into {}...'.format(sample_path))

            # Saving images for next generator.
            # The iteration should be consistent with the next generator
            if stack_mode=='A':                     
                if (i+1) % ((self.n_critic)*200) == 0:
                    with torch.no_grad():
                        #create labels for the data
                        for j in range(len(c_fixed_list)):
                            photos = self.denorm(self.G(x_fixed_all, c_fixed_list[j]))
                            for index in range(len(c_fixed_list[0])):
                                intermediate_path = os.path.join(self.intermediate_dir, '{0}-{1}-images.jpg'.format(index,j))
                                save_image(photos[index],intermediate_path)             
                        print('Saved intermediate images for next GAN into {}...'.format(intermediate_path))            
            
    
            # Save model checkpoints.
            if (i+1) % self.model_save_step == 0:
                G_path = os.path.join(self.model_save_dir, '{}-G.ckpt'.format(i+1))
                D_path = os.path.join(self.model_save_dir, '{}-D.ckpt'.format(i+1))
                torch.save(self.G.state_dict(), G_path)
                torch.save(self.D.state_dict(), D_path)
                print('Saved model checkpoints into {}...'.format(self.model_save_dir))
    
            # Decay learning rates.
            if (i+1) % self.lr_update_step == 0 and (i+1) > (self.num_iters - self.num_iters_decay):
                g_lr -= (self.g_lr / float(self.num_iters_decay))
                d_lr -= (self.d_lr / float(self.num_iters_decay))
                self.update_lr(g_lr, d_lr)
                print ('Decayed learning rates, g_lr: {}, d_lr: {}.'.format(g_lr, d_lr))
            
            "CartoonGAN"
            if (i+1)% ((self.n_critic)*200) ==0:
                epoch= i
                epoch_idx= i +1
        
                for step in tqdm(
                        range(1, steps_per_epoch + 1),
                        desc=f'Train {epoch + 1}/{epochs}',
                        total=steps_per_epoch):
                    source_images, target_images, smooth_images = (
                        ds_source.next(), ds_target.next(), ds_smooth.next())
                    self.train_step(source_images, target_images, smooth_images,
                                    generator, d, g_optimizer, d_optimizer)
        
                    if step % self.reporting_steps == 0:
        
                        global_step = (epoch_idx - 1) * steps_per_epoch + step
                        with summary_writer.as_default():
                            for metric, name in self.metric_and_names:
                                tf.summary.scalar(name, metric.result(), step=global_step)
                                metric.reset_states()
                            if not self.disable_sampling:
                                fake_batch = tf.cast(
                                    (generator(real_batch, training=False) + 1) * 127.5, tf.uint8)
                                img = np.expand_dims(self._save_generated_images(
                                        fake_batch,
                                        image_name=("gan_generated_images_at_epoch_"
                                                    f"{epoch_idx}_step_{step}.png"),result_dir_cartoonGAN),
                                        0,)
                                tf.summary.image('gan_generated_images', img, step=global_step)
                                
                                # output intermediate images for next GAN
                                if stack_mode=='B':
                                    img = np.expand_dims(self._save_generated_images(
                                            fake_batch,
                                            image_name=("gan_generated_images_at_epoch_"
                                                        f"{epoch_idx}_step_{step}.png"),result_dir_cartoonGAN),
                                            0,dir_='datasets/RaFDB/train')
                                    tf.summary.image('gan_generated_images', img, step=global_step)
    
                        self.logger_cartoonGAN.debug(f"Epoch {epoch_idx}, Step {step} finished, "
                                          f"{global_step * self.batch_size_cartoonGAN} images processed.")
        
                with summary_writer.as_default():
                    if not self.disable_sampling:
                        val_fake_batch = tf.cast(
                            (generator(val_real_batch, training=False) + 1) * 127.5, tf.uint8)
                        img = np.expand_dims(self._save_generated_images(
                                val_fake_batch,
                                image_name=("gan_val_generated_images_at_epoch_"
                                            f"{epoch_idx}_step_{step}.png")),
                                0,
                        )
                        tf.summary.image('gan_val_generated_images', img, step=epoch)
                self.logger_cartoonGAN.info(f"Saving checkpoints after epoch {epoch_idx} ended...")
                g_checkpoint.save(file_prefix=self.generator_checkpoint_prefix)
                d_checkpoint.save(file_prefix=self.discriminator_checkpoint_prefix)
        
                generator.save_weights(os.path.join(self.model_dir, "generator"))
                gc.collect()
#            del ds_source, ds_target, ds_smooth
            gc.collect()
    

    def test(self):
        """Translate images using StarGAN trained on a single dataset."""
        # Load the trained generator.
        self.restore_model(self.test_iters)
        
        # Set data loader.
        data_loader = self.rafdb_loader
        
        with torch.no_grad():
            for i, (x_real, c_org) in enumerate(data_loader):
                # Prepare input images and target domain labels.
                x_real = x_real.to(self.device)
                c_trg_list = self.create_labels(c_org, self.c_dim)
                # Translate images.
                cnt = 1
                for c_trg in c_trg_list:
                    for j in range(16):
                        result_path = os.path.join(self.result_dir, '{0}-{1}-images.jpg'.format((i*16 + j),cnt))
                        file = self.denorm(self.G(x_real, c_trg))
                        save_image(file[j],result_path)    
                    cnt= cnt+1


    
    
    
