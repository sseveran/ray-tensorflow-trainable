#  Copyright 2016 The TensorFlow Authors. All Rights Reserved.
#
#  Licensed under the Apache License, Version 2.0 (the "License");
#  you may not use this file except in compliance with the License.
#  You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
#  limitations under the License.
"""An Example of a custom Estimator for the Iris dataset."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
from tensorflow.python.training import training_util
import ray
from ray.tune.result import TrainingResult
from ray.tune.trainable import Trainable
from ray.tune.hpo_scheduler import HyperOptScheduler
import tensorflow as tf
import hyperopt.hp as hp

import iris_data

parser = argparse.ArgumentParser()
parser.add_argument('--batch_size', default=100, type=int, help='batch size')
parser.add_argument('--train_steps', default=1000, type=int,
                    help='number of training steps')

def my_model(features, labels, mode, params):
    """DNN with three hidden layers, and dropout of 0.1 probability."""
    # Create three fully connected layers each layer having a dropout
    # probability of 0.1.
    net = tf.feature_column.input_layer(features, params['feature_columns'])
    for units in params['hidden_units']:
        net = tf.layers.dense(net, units=units, activation=tf.nn.relu)

    # Compute logits (1 per class).
    logits = tf.layers.dense(net, params['n_classes'], activation=None)

    # Compute predictions.
    predicted_classes = tf.argmax(logits, 1)
    if mode == tf.estimator.ModeKeys.PREDICT:
        predictions = {
            'class_ids': predicted_classes[:, tf.newaxis],
            'probabilities': tf.nn.softmax(logits),
            'logits': logits,
        }
        return tf.estimator.EstimatorSpec(mode, predictions=predictions)

    # Compute loss.
    loss = tf.losses.sparse_softmax_cross_entropy(labels=labels, logits=logits)

    # Compute evaluation metrics.
    accuracy = tf.metrics.accuracy(labels=labels,
                                   predictions=predicted_classes,
                                   name='acc_op')
    metrics = {'accuracy': accuracy}
    tf.summary.scalar('accuracy', accuracy[1])

    if mode == tf.estimator.ModeKeys.EVAL:
        return tf.estimator.EstimatorSpec(
            mode, loss=loss, eval_metric_ops=metrics)

    # Create training op.
    assert mode == tf.estimator.ModeKeys.TRAIN

    optimizer = tf.train.AdagradOptimizer(learning_rate=0.1)
    train_op = optimizer.minimize(loss, global_step=tf.train.get_global_step())
    return tf.estimator.EstimatorSpec(mode, loss=loss, train_op=train_op)


class TestTrainable(Trainable):
    def _setup(self):
        self.steps = 0
        self.session = tf.Session()
        (train_x, train_y), (test_x, test_y) = iris_data.load_data()
        self.train_x = train_x
        self.train_y = train_y

        self.test_x = test_x
        self.test_y = test_y

        # Feature columns describe how to use the input.
        my_feature_columns = []
        for key in train_x.keys():
            my_feature_columns.append(tf.feature_column.numeric_column(key=key))

        layer_size = int(self.config['layer_size'])

        # Build 2 hidden layer DNN with 10, 10 units respectively.
        self.classifier = tf.estimator.Estimator(
            model_fn=my_model,
            params={
                'feature_columns': my_feature_columns,
                # Two hidden layers of 10 nodes each.
                'hidden_units': [layer_size, layer_size],
                # The model must choose between 3 classes.
                'n_classes': 3,
            })

        self.saver = None
        self.global_step_tensor = training_util._get_or_create_global_step_read()  # pylint: disable=protected-access

    def _train(self):
        self.classifier.train(
            input_fn=lambda: iris_data.train_input_fn(self.train_x, self.train_y, 10),
            steps=100)

        self.steps = self.steps + 100

        eval_result = self.classifier.evaluate(
            input_fn=lambda: iris_data.eval_input_fn(self.test_x, self.test_y, 10))

        return TrainingResult(timesteps_this_iter=100, timesteps_total=self.steps, mean_validation_accuracy=eval_result['accuracy'])

    def _save(self, checkpoint_dir):
        #saver must be set here, otherwise there will be no variables to have
        if self.saver is None:
            self.saver = tf.train.Saver()
        return self.saver.save(
            self.session, checkpoint_dir + "/save",
            global_step=self.steps)

    def _restore(self, checkpoint_path):
        return self.saver.restore(self.session, checkpoint_path)


if __name__ == '__main__':
    ray.init()
    config = {'iris_test': {
        'run': 'iris_test',
        'stop': {'mean_validation_accuracy': 0.999999999999},
        "trial_resources": {"cpu": 1, "gpu": 0},
        'repeat': 1,
        'config': {
            'space': {
                'layer_size': hp.uniform('layer_size', 10, 100),
            },
        }
    }}

    hpo_sched = HyperOptScheduler(max_concurrent=4, reward_attr="mean_validation_accuracy")
    ray.tune.register_trainable("iris_test", TestTrainable)
    ray.tune.run_experiments(config, verbose=True, scheduler=hpo_sched)

    #tf.logging.set_verbosity(tf.logging.INFO)
    #tf.app.run(main)