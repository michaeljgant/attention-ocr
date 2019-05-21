import tensorflow as tf
import os

from aocr.model.model import Model
from aocr.defaults import Config
from aocr.util import dataset
from aocr.util.data_gen import DataGen
from aocr.util.export import Exporter

os.chdir("../phase_1/images")
os.getcwd()

dataset_train = dataset.generate(
    "train_labels.txt",
    './dataset.tfrecords',
    500,
    True,
    False
    )

sess = tf.Session(config=tf.ConfigProto(allow_soft_placement=True))

model_train = Model(
            phase="train",
            visualize=False,
            output_dir="./results",
            batch_size=65,
            initial_learning_rate=1.0,
            steps_per_checkpoint=200,
            model_dir="./checkpoints",
            target_embedding_size=10,
            attn_num_hidden=128,
            attn_num_layers=2,
            clip_gradients=True,
            max_gradient_norm=5.0,
            session=sess,
            load_model=False,
            gpu_id=0,
            use_gru=True,
            use_distance=True,
            max_image_width=1300,
            max_image_height=75,
            max_prediction_length=19,
            channels=1,
        )

model_train.max_original_width
model_train.max_width
model_train.height

model_train.encoder_size