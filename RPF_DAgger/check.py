# %%
import os
import joblib
import numpy as np
import matplotlib.pyplot as plt
import cv2
from model import *

# %%
tf.__version__
# %%
batch_size = 1
GRU_size = 512
action_dim = 4
learning_rate = 1e-4
demo_length = 30
max_follow_length = 50
memory_dim = 512
EPOCH = 500
img_size = 256
feature_dim = 512
# %%
DATA_DIR = '../RPF/preprocessed_habitat_data2'

train_list = [os.path.join(DATA_DIR, 'train', x) for x in os.listdir(os.path.join(DATA_DIR, 'train'))]
valid_list = [os.path.join(DATA_DIR, 'valid', x) for x in os.listdir(os.path.join(DATA_DIR, 'valid'))]

version = 'v1'
BC_restore_version = 'v3'
def build_network(whatfor='train',loader=None, keep_trainin=True):
    rpf = RPF()
    if whatfor == 'train' :
        rpf.batch_size = 4
        rpf.build_network_for_train(loader)
        train_loss_summary = tf.summary.scalar('train_loss', rpf.loss)
    else :
        rpf.batch_size = 1
        rpf.build_network_for_running()
        train_loss_summary = None

    return rpf, None, None, train_loss_summary

# %%
def load_data(data_path):
    try:
        noise_data = joblib.load(data_path)
        mode = 'train' if 'train' in str(data_path) else 'valid'
        ref_data_name = str(data_path).replace(mode, 'DEMON')
        ref_data_name = ref_data_name[:ref_data_name.find('.dat.gz') - 1] + '0.dat.gz'
        ref_data = joblib.load(str(ref_data_name))
        # ref_data = joblib.load(str(ref_data_name))

        noise_rgb = np.array(noise_data['rgb'], dtype=np.float32) / 255.0 * 2 - 1
        ref_rgb = np.array(ref_data['rgb'], dtype=np.float32) / 255.0 * 2 - 1

        noise_data['action'] = np.array(noise_data['action'], dtype=np.int8)
        ref_data['action'] = np.array(ref_data['action'], dtype=np.int8)

        a = np.where(noise_data['action'] > 5)
        b = np.where(ref_data['action'] > 5)
        noise_data['action'][a] = noise_data['action'][a] - 5
        ref_data['action'][b] = ref_data['action'][b] - 5

        noise_action = np.eye(action_dim)[noise_data['action']]
        ref_action = np.eye(action_dim)[ref_data['action']]

        add_t = max_follow_length - len(noise_rgb)
        if add_t < 0:
            noise_rgb = noise_rgb[:max_follow_length]
            noise_action = noise_action[:max_follow_length]
            action_mask = np.ones_like(noise_action)
            action_mask = action_mask[:max_follow_length]
            print('its weird..', data_path)

        else:
            img_size = noise_rgb[0].shape[1]
            action_mask = np.ones_like(noise_action)

            noise_rgb = np.concatenate([noise_rgb, np.zeros([add_t, img_size, img_size, 3])], 0)
            noise_action = np.concatenate([noise_action, np.zeros([add_t, action_dim])], 0)
            action_mask = np.concatenate([action_mask, np.zeros([add_t, action_dim])], 0)


    except:
        ref_rgb = np.zeros([30, 256, 256, 3])
        ref_action = np.zeros([30, 4])
        noise_rgb = np.zeros([50, 256, 256, 3])
        noise_action = np.zeros([50, 4])
        action_mask = np.zeros([50, 4])

    result = [ref_rgb, ref_action, noise_rgb, noise_action, action_mask]
    return [np.array(x, dtype=np.float32) for x in result]


# %%
tf.reset_default_graph()
#rpf = RPF()
#rpf.batch_size = batch_size
# rpf.build_network_for_train(None,placeholder=True)
#rpf.build_network_for_running()
rpf, _,_,_ = build_network('running',keep_trainin=False)

init = tf.global_variables_initializer()
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
network_saver = tf.train.Saver(tf.trainable_variables(), max_to_keep=100)
# saver_a = tf.train.Saver([v for v in tf.all_variables() if v.name == "a1:0"])

version_path = 'v3'
if not os.path.exists('models/' + version_path): os.makedirs('models/' + version_path)
if not os.path.exists('logs/' + version_path): os.makedirs('logs/' + version_path)

writer = tf.summary.FileWriter('logs/' + version_path)
# if not os.path.exists('images/'+version_path) : os.makedirs('images/'+version_path)

sess = tf.Session(config=config)
sess.run(init)
pre_ckpt = tf.train.get_checkpoint_state('../habitat-api/rpf/models/' + version_path)
if pre_ckpt and pre_ckpt.model_checkpoint_path:
    past_ckpt = pre_ckpt.model_checkpoint_path
    # past_ckpt = 'models/v1ep0058step761985'
    network_saver.restore(sess, past_ckpt)
    print("1. Successfully loaded:", past_ckpt)
    ep = int(past_ckpt[past_ckpt.find('ep') + 2:past_ckpt.find('ep') + 6])
    step = int(past_ckpt[past_ckpt.find('step') + 4:past_ckpt.find('step') + 8])
else:
    print("1! Could not find old network weights")
    ep = 0
    step = 0


# %%
def softmax(x):
    """Compute softmax values for each sets of scores in x."""
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum(axis=0)


def pprint(x, digit=4):
    element = "%" + ".{}f ".format(str(digit))
    log_str = ""
    for e in x:
        log_str += element % (e)
    print(log_str)
    return




# %% md
# BUILT FOR RUNNING
# %%
rpf.sess = sess
# %%

batch = load_data(valid_list[9])
demonstration = [np.expand_dims(batch[0], 0), np.expand_dims(batch[1], 0)]
rpf.reset()
rpf.encode_memory(demonstration)
for t in range(max_follow_length):
    plt.subplot(131)
    demo_img = (batch[0][t] + 1) / 2.
    plt.imshow(demo_img)
    plt.subplot(132)
    demo_img = (batch[0][int(rpf.curr_eta)] + 1) / 2.
    ref_act = batch[1][t]
    if np.argmax(ref_act) == 1:
        d_angle = 0
    elif np.argmax(ref_act) == 2:
        d_angle = 1
    elif np.argmax(ref_act) == 3:
        d_angle = -1
    else:
        d_angle = 0
        print('stop?')

    plt.imshow(demo_img)

    stop = 5
    if t > stop:
        follow_img = (batch[2][stop] + 1) / 2.
        fol_act = rpf.predict_action(np.expand_dims(batch[2][stop], 0)).squeeze()
    else:
        follow_img = (batch[2][t] + 1) / 2.
        fol_act = rpf.predict_action(np.expand_dims(batch[2][t], 0)).squeeze()
    if int(fol_act) == 1:
        f_angle = 0
    elif int(fol_act) == 2:
        f_angle = 1
    elif int(fol_act) == 3:
        f_angle = -1
    else:
        f_angle = 0
        print('stop?')
    cv2.line(follow_img, (128, 256), (int(128 - 40 * d_angle), 256 - 40), (255, 0, 0), 3)
    cv2.line(follow_img, (128, 256), (int(128 - 40 * f_angle), 256 - 40), (0, 255, 0), 3)

    plt.subplot(133)
    plt.imshow(follow_img)
    plt.show()

    print('eta : ', int(rpf.curr_eta))
    pprint(ref_act)
    pprint(np.eye(4)[int(fol_act)])
    pprint(np.stack([np.exp(-abs(rpf.curr_eta - float(j))) for j in range(demo_length)], 1).squeeze())
# %%
np.stack([np.exp(-abs(rpf.curr_eta - float(j))) for j in range(demo_length)], 1).squeeze()
