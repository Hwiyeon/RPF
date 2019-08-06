from model import RPF
from habitat_run_setup import *
from dagger_data_process import *
import tensorflow as tf
import os
import shutil
import sys
import cv2
import numpy as np
import habitat
from habitat.tasks.nav.shortest_path_follower import ShortestPathFollower
from habitat.utils.visualizations import maps
from habitat.utils.visualizations.utils import images_to_video
from habitat.tasks.nav.shortest_path_follower import ShortestPathFollower
from habitat.utils.visualizations import maps
from habitat.utils.visualizations.utils import images_to_video
from habitat.core.simulator import Observations
import numpy as np
import quaternion as q


import joblib
import quaternion as q
#sys.path.append('../habitat-api/examples/')
import NoisyAction
import time

CONTENT_PATH = 'data/datasets/pointnav/mp3d/v1/train/content/'
# you should download task datasets from https://github.com/facebookresearch/habitat-api
NAV_DATA_DIR = 'demo_data'
IMAGE_DIR ='dagger_test_run'
if not os.path.exists(IMAGE_DIR):
    os.makedirs(IMAGE_DIR)
RENDER= True

version = 'v2'
DATA_SAVE_DIR = 'RPF_DAgger/dagger_data/' + version
if not os.path.exists(DATA_SAVE_DIR): os.mkdir(DATA_SAVE_DIR)
BC_restore_version = 'v3'
MODEL_SAVE_DIR = 'RPF_DAgger/models/' + 'dagger{}from{}'.format(version,BC_restore_version)
if not os.path.exists(MODEL_SAVE_DIR): os.mkdir(MODEL_SAVE_DIR)

DEMO_LENGTH = 30
FOLLOW_LENGTH = 50
MAX_TRAIN_EPOCH = 5
MAX_AGGREGATION_EPISODE = 50

TRAIN_PRINT_STEP = 2
DATA_AGG_PRINT_STEP = 2

def build_network(whatfor='train',loader=None, keep_trainin=True):
    rpf = RPF()

    if whatfor == 'train' :
        rpf.batch_size = 4
        rpf.build_network_for_train(loader.get_next_batch)
        train_loss_summary = tf.summary.scalar('train_loss', rpf.loss)
    else :
        rpf.batch_size = 1
        rpf.build_network_for_running()
        train_loss_summary = None

    init = tf.global_variables_initializer()
    tf_config = tf.ConfigProto()
    tf_config.gpu_options.allow_growth = True
    network_saver = tf.train.Saver(tf.trainable_variables(), max_to_keep=100)
    # saver_a = tf.train.Saver([v for v in tf.all_variables() if v.name == "a1:0"])
    if not os.path.exists('logs/' + 'dagger{}from{}'.format(version, BC_restore_version)):
        os.mkdir('logs/' + 'dagger{}from{}'.format(version, BC_restore_version))
    writer = tf.summary.FileWriter('logs/' + 'dagger{}from{}'.format(version, BC_restore_version))

    sess = tf.Session(config=tf_config)
    sess.run(init)
    if keep_trainin is False:
        pre_ckpt = tf.train.get_checkpoint_state('RPF_DAgger/models/' + BC_restore_version)
    else:
        pre_ckpt = tf.train.get_checkpoint_state(
            'RPF_DAgger/models/' + 'dagger{}from{}'.format(version, BC_restore_version))

    if pre_ckpt and pre_ckpt.model_checkpoint_path:
        past_ckpt = pre_ckpt.model_checkpoint_path
        network_saver.restore(sess, past_ckpt)
        print("LOADED pretrained model", past_ckpt)
        ep = int(past_ckpt[past_ckpt.find('ep') + 2:past_ckpt.find('ep') + 6])
        step = int(past_ckpt[past_ckpt.find('step') + 4:past_ckpt.find('step') + 8])
    else:
        print("1! Could not find old network weights")
        ep = 0
        step = 0

    rpf.sess = sess

    return rpf, writer, network_saver, train_loss_summary


def data_aggregate(space, config, file_list, rpf=None):
    space_name = space  # [:space.find('.json')]
    print("START TRAINING ON {}".format(space_name))
    config.defrost()
    config.DATASET.DATA_PATH = CONTENT_PATH + space + '.json.gz'
    config.DATASET.SCENES_DIR = 'data/scene_datasets'
    #config.SIMULATOR.SCENE = '../habitat-api/' + config.SIMULATOR.SCENE
    config.SIMULATOR.ACTION_SPACE_CONFIG = "NoNoisyMove"
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
    ep = 0
    #for ep in range(MAX_AGGREGATION_EPISODE):
    while (ep < MAX_AGGREGATION_EPISODE):

        config.defrost()
        config.SIMULATOR.ACTION_SPACE_CONFIG = "NoNoisyMove"
        config.freeze()
        envs.reconfigure(config)

        observations = envs.reset()
        demonstrations = run_env(envs, DEMO_LENGTH, observations)
        if len(demonstrations['action']) < 30 : continue
        else :
            demonstrations['rgb'] = demonstrations['rgb'][:30]
            demonstrations['action'] = demonstrations['action'][:30]
            demonstrations['position']  = demonstrations['position'][:30]
            demonstrations['rotation'] = demonstrations['rotation'][:30]
        config.defrost()
        config.SIMULATOR.ACTION_SPACE_CONFIG = "NoisyMove"
        config.freeze()
        envs.reconfigure(config)

        observations = envs.reset_curr_episode()
        if space_idx >= 0 and rpf is not None:
            noisy_demo = run_env(envs, FOLLOW_LENGTH, observations, run_mode='actor', actor=rpf, demo=demonstrations)
        else:
            noisy_demo = run_env(envs, FOLLOW_LENGTH, observations)

        # for i in range(3):
        if len(noisy_demo['rgb']) < 5 : continue
        ep_demo_list.append([demonstrations, noisy_demo])

        if ep % DATA_AGG_PRINT_STEP == 0:
            print('[%02d/%02d] DATA AGGREGATION DONE' % (ep, MAX_AGGREGATION_EPISODE))
        ep += 1

    envs.close()
    # SAVE DATA
    ret_file_list = file_list
    ret_file_list.extend(preprocess_and_save_demo(ep_demo_list, 'RPF_DAgger/dagger_data/{}/{}_demos_{}.npy'.format(version,space_name,'ep_num')))
    print('TOTAL %03d DEMONSTRAIONS COLLECTED' % (len(ep_demo_list)))
    return ret_file_list


def train(space_id, loader, rpf, writer, train_loss_summary, network_saver):
    training_start = time.time()
    loss_record = 0
    step = 0
    training_handle = rpf.sess.run(loader.training_iterator.string_handle())
    rpf.sess.run(loader.training_iterator.initializer)
    for ep in range(MAX_TRAIN_EPOCH):
        loss_record = 0
        step = 0
        times = []
        for batch_num in range(loader.num_batches):
            start = time.time()
            summary, _, curr_loss = rpf.sess.run([train_loss_summary, rpf.opti_step, rpf.loss], feed_dict={loader.handle : training_handle})
            loss_record += curr_loss
            # input_feats = sess.run(rpf.record['inp_feats_t'],feed_dict={handle: training_handle})
            # print(np.min(input_feats), np.max(input_feats))
            writer.add_summary(summary, global_step=step)
            if step % TRAIN_PRINT_STEP == 0:
                print('\t step %d, mean loss %.7f, mean step time %.3f' % (
                    step, loss_record / (batch_num + 1), np.mean(times)))
            step += 1
            times.append(time.time() - start)

    print('Training Done : MEAN LOSS %.7f, TOTAL TIME %.3f' % (loss_record / loader.num_batches, time.time() - training_start))
    model_name = os.path.join('RPF_DAgger','models', 'dagger{}from{}'.format(version, BC_restore_version),
                              'ep%04dstep%06d' % (space_id, step))
    network_saver.save(rpf.sess, model_name)


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
        keep_trainin = False if space_idx == 0 else True
        tf.reset_default_graph()
        rpf, _,_,_ = build_network('running',keep_trainin=keep_trainin)

        file_list = data_aggregate(space, config, file_list,rpf)
        if len(file_list) > 200 :
            file_list = file_list[len(file_list)-200 :]

        tf.reset_default_graph()
        loader.init_w_file_list(file_list)
        rpf, writer, network_saver, train_loss_summary = build_network('train',loader,keep_trainin=keep_trainin)
        # 2. Training
        train(space_idx, loader, rpf, writer, train_loss_summary, network_saver)

