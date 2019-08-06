from model import RPF
from habitat_run_setup import *
from dagger_data_process import *
import tensorflow as tf
import os
import shutil
import sys
import cv2
import numpy as np

sys.path.append('../habitat-api')
import habitat
from habitat.tasks.nav.shortest_path_follower import ShortestPathFollower
from habitat.utils.visualizations import maps
from habitat.utils.visualizations.utils import images_to_video

import joblib
import quaternion as q
sys.path.append('../habitat-api/examples/')
import NoisyAction
import time

CONTENT_PATH = '../habitat-api/data/datasets/pointnav/mp3d/v1/train/content/'
NAV_DATA_DIR = '../habitat-api/demo_data'
IMAGE_DIR ='dagger_test_run'
if not os.path.exists(IMAGE_DIR):
    os.makedirs(IMAGE_DIR)
RENDER= True

version = 'v1'
DATA_SAVE_DIR = 'dagger_data/' + version
if not os.path.exists(DATA_SAVE_DIR): os.mkdir(DATA_SAVE_DIR)
BC_restore_version = 'v2.4'
MODEL_SAVE_DIR = 'models/' + 'dagger{}from{}'.format(version,BC_restore_version)
if not os.path.exists(MODEL_SAVE_DIR): os.mkdir(MODEL_SAVE_DIR)

DEMO_LENGTH = 30
FOLLOW_LENGTH = 50
MAX_TRAIN_EPOCH = 0
MAX_AGGREGATION_EPISODE = 2

TRAIN_PRINT_STEP = 10
DATA_AGG_PRINT_STEP = 10




## build_network
build_network = False
if build_network :
    rpf = RPF()
    rpf.build_network_for_train()
    rpf.build_network_for_running()

    train_loss_summary = tf.summary.scalar('train_loss',rpf.loss)
    valid_loss_summary = tf.summary.scalar('valid_loss',rpf.loss)

    init = tf.global_variables_initializer()
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    network_saver = tf.train.Saver(tf.trainable_variables(), max_to_keep=100)
    # saver_a = tf.train.Saver([v for v in tf.all_variables() if v.name == "a1:0"])
    writer = tf.summary.FileWriter('logs/'+'dagger{}from{}'.format(version,BC_restore_version))

    sess = tf.Session(config=config)
    sess.run(init)
    pre_ckpt = tf.train.get_checkpoint_state('models/' + BC_restore_version)
    if pre_ckpt and pre_ckpt.model_checkpoint_path:
        past_ckpt = pre_ckpt.model_checkpoint_path
        #network_saver.restore(sess, past_ckpt)
        print("LOADED pretrained model", past_ckpt)
        ep = int(past_ckpt[past_ckpt.find('ep') + 2:past_ckpt.find('ep') + 6])
        step = int(past_ckpt[past_ckpt.find('step') + 4:past_ckpt.find('step') + 8])
    else:
        print("1! Could not find old network weights")
        ep = 0
        step = 0

    rpf.sess = sess



def data_aggregate(space, config, file_list):
    space_name = space  # [:space.find('.json')]
    print("START TRAINING ON {}".format(space_name))
    config.defrost()
    config.DATASET.DATA_PATH = CONTENT_PATH + space + '.json.gz'
    config.DATASET.SCENES_DIR = '../habitat-api/data/scene_datasets'
    config.SIMULATOR.SCENE = '../habitat-api/' + config.SIMULATOR.SCENE
    config.freeze()

    num_of_envs = 1
    envs = SimpleRLEnv(config)

    # envs = habitat.VectorEnv(
    #    make_env_fn=make_env_fn,
    #    env_fn_args=tuple(
    #        tuple(
    #            zip([config] * num_of_envs, range(num_of_envs))
    ##        )
    #   ),
    # )

    ep_demo_list = []
    for ep in range(MAX_AGGREGATION_EPISODE):

        config.defrost()
        config.SIMULATOR.ACTION_SPACE_CONFIG = "NoNoisyMove"
        config.freeze()
        envs.reconfigure(config)

        observations = envs.reset()
        demonstrations = run_env(envs, DEMO_LENGTH, observations)

        config.defrost()
        config.SIMULATOR.ACTION_SPACE_CONFIG = "NoisyMove"
        config.freeze()
        envs.reconfigure(config)

        observations = envs.reset_curr_episode()
        if space_idx > 0:
            noisy_demo = run_env(envs, FOLLOW_LENGTH, observations, run_mode='actor', actor=rpf)
        else:
            noisy_demo = run_env(envs, FOLLOW_LENGTH, observations)

        # for i in range(3):
        ep_demo_list.append([demonstrations[i], noisy_demo[i]])

        if ep % DATA_AGG_PRINT_STEP == 0:
            print('[%02d/%02d] DATA AGGREGATION DONE' % (ep, MAX_AGGREGATION_EPISODE))

    envs.close()
    # SAVE DATA
    ret_file_list = file_list
    ret_file_list.extend(preprocess_and_save_demo(ep_demo_list, '{}_demos.npy'.format(space_name)))
    print('TOTAL %03d DEMONSTRAIONS COLLECTED' % (len(ep_demo_list)))
    return ret_file_list


def train(space_id, loader):
    training_start = time.time()
    for ep in range(MAX_TRAIN_EPOCH):
        loss_record = 0
        step = 0
        times = []
        for batch_num in range(loader.num_batches):
            start = time.time()

            batch = loader.get_next_batch()
            feed_dict = {rpf.demo_seqs_list: batch[0],
                         rpf.demo_acts_list: batch[1],
                         rpf.inp_rgb_list: batch[2],
                         rpf.inp_act_list: batch[3],
                         rpf.act_loss_mask_list: batch[4]}

            summary, _, curr_loss = sess.run([train_loss_summary, rpf.opti_step, rpf.loss], feed_dict=feed_dict)
            loss_record += curr_loss
            # input_feats = sess.run(rpf.record['inp_feats_t'],feed_dict={handle: training_handle})
            # print(np.min(input_feats), np.max(input_feats))
            writer.add_summary(summary, global_step=step)
            if step % TRAIN_PRINT_STEP == 0:
                print('\t step %d, mean loss %.7f, mean step time %.3f' % (
                    step, loss_record / (batch_num + 1), np.mean(times)))
            step += 1
            times.append(time.time() - start)

    print('Training Done : MEAN LOSS %.7f, TOTAL TIME %.3f' % (
    loss_record / loader.num_batches, time.time() - training_start))
    model_name = os.path.join('models', 'dagger{}from{}'.format(version, BC_restore_version),
                              'ep%04dstep%06d' % (space_id, step))
    network_saver.save(sess, model_name)


if __name__ == '__main__':
    h_t = None
    memories = None
    eta = None

    habitat.SimulatorActions.extend_action_space("NOISY_FORWARD")
    habitat.SimulatorActions.extend_action_space("NOISY_LEFT")
    habitat.SimulatorActions.extend_action_space("NOISY_RIGHT")


    space_list = os.listdir(NAV_DATA_DIR)
    #for space_id, space in enumerate(space_list):
    config = habitat.get_config(config_paths="../habitat-api/configs/tasks/pointnav_mp3d.yaml")

    config.defrost()
    config.TASK.MEASUREMENTS.append("TOP_DOWN_MAP")
    config.TASK.SENSORS.append("HEADING_SENSOR")
    config.SIMULATOR.TURN_ANGLE = NoisyAction.THETA
    config.SIMULATOR.FORWARD_STEP_SIZE = NoisyAction.X
    config.freeze()



    file_list = []
    loader = data_loader()

    ## 1. data aggregation

    ##### 1.1 Demonstration
    for space_idx, space in enumerate(space_list):

        file_list = data_aggregate(space, config, file_list)
        loader.load_data_from_files(file_list)

        ## 2. Training
        train(space_idx, loader)

