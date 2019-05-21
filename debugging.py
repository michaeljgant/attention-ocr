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
model_train.decoder_size

model_train.conv_output

model_train.attention_decoder_model.output


model_train.encoder_masks
model_train.img_data


len(model_train.decoder_inputs)
len(model_train.target_weights)

len(model_train.attention_decoder_model.target_weights)
len(model_train.attention_decoder_model.decoder_inputs)
len(model_train.attention_decoder_model.encoder_masks)
len(model_train.attention_decoder_model.output)
model_train.attention_decoder_model.output[0].shape


s_gen = DataGen(
            data_path, model_train.buckets,
            , max_width=self.max_original_width
        )




"""
Out[40]: 1300
model_train.max_width
Out[41]: 555
model_train.height
Out[42]: <tf.Tensor 'Const:0' shape=() dtype=int32>
model_train.encoder_size
Out[43]: 139
model_train.decoder_size
Out[44]: 21
model_train.conv_output
Out[46]: <tf.Tensor 'Squeeze:0' shape=(?, 139, 512) dtype=float32>
model_train.img_data
Out[50]: <tf.Tensor 'map/TensorArrayStack/TensorArrayGatherV3:0' shape=(?, ?, 555, 1) dtype=float32>
model_train.attention_decoder_model.output[0].shape
Out[61]: TensorShape([Dimension(None), Dimension(39)])
Out[57]: 22
len(model_train.attention_decoder_model.encoder_masks)
Out[58]: 140
len(model_train.attention_decoder_model.output)
Out[59]: 21

"""