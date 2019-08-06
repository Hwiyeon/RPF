import numpy as np
import joblib
import quaternion as q
import tensorflow as tf

max_follow_length = 50
def slice_data_by_index(data, s_idx, e_idx):
    nu_data = dict()
    for key in data.keys():
        nu_data[key] = data[key][s_idx:e_idx+1]
    return nu_data

def arr_quat_to_rots(quats):
    ret = q.as_rotation_vector(q.as_quat_array(quats))
    ret = ret.reshape(-1,3)
    ret = ret[:,1]-3.14
    ret[np.where(ret > 3.14)] -= 6.28
    ret[np.where(ret < -3.14)] += 6.28
    return ret

def preprocess_and_save_demo(demonstration_data, file_name, preprocess=False):
    print(' {} EPISODES OF DATA COLLECTED'.format(len(demonstration_data)))
    print(' START PROCESSING DATA ')

    save_list = []
    data_to_save = dict()
    saved_e = 0
    for ep in demonstration_data:
        demo_data = ep[0]
        if len(demo_data['rgb']) < 30 : continue
        noise_data = ep[1]
        if len(noise_data['rgb']) > max_follow_length :
            noise_data['rgb'] = noise_data['rgb'][:max_follow_length]
            noise_data['action'] = noise_data['action'][:max_follow_length]
        
        if preprocess :
            noise_rgb = np.array(noise_data['rgb'], dtype=np.float32) / 255.0 * 2 - 1
            ref_rgb = np.array(demo_data['rgb'], dtype=np.float32) / 255.0 * 2 - 1

            noise_data['action'] = np.array(noise_data['action'])
            demo_data['action'] = np.array(demo_data['action'])

            a = np.where(noise_data['action'] > 5)
            b = np.where(demo_data['action'] > 5)
            noise_data['action'][a] = noise_data['action'][a] - 5
            demo_data['action'][b] = demo_data['action'][b] - 5

            noise_action = np.eye(4)[noise_data['action']]
            ref_action = np.eye(4)[demo_data['action']]

            add_t = max_follow_length - len(noise_rgb)
            img_size = noise_rgb[0].shape[1]
            action_mask = np.ones_like(noise_action)

            noise_rgb = np.concatenate([noise_rgb, np.zeros([add_t, img_size, img_size, 3])], 0)
            noise_action = np.concatenate([noise_action, np.zeros([add_t, 4])], 0)
            action_mask = np.concatenate([action_mask, np.zeros([add_t, 4])], 0)

            result = [ref_rgb, ref_action, noise_rgb, noise_action, action_mask]

        else :
            noise_rgb = noise_data['rgb']
            ref_rgb = demo_data['rgb']
            noise_action = noise_data['action']
            ref_action = demo_data['action']
            result = [ref_rgb, ref_action, noise_rgb, noise_action]#, action_mask]
        np.save(file_name.replace('ep_num',str(saved_e)), result)
        save_list.append(file_name.replace('ep_num',str(saved_e)))
        saved_e += 1


    print('TOTAL num: {} of {} saved'.format(saved_e, file_name))
    return save_list





class data_loader():
    def __init__(self,batch_size=4):
        self.batch_size = batch_size


    def init_w_file_list(self,file_list):
        data_list = file_list
        self.num_batches = int(len(file_list)/self.batch_size)
        print('num of batches {}'.format(self.num_batches))
        np.random.shuffle(data_list)
        file_list = file_list[:self.batch_size * self.num_batches]
        train_dataset = tf.data.Dataset.from_tensor_slices(file_list)
        dataset = train_dataset.map(lambda data_path: tf.py_func(self.load_data_from_files, [data_path], [tf.float32, ] * 5),
                                    num_parallel_calls=5)
        dataset = dataset.batch(self.batch_size).repeat()  # .shuffle(100).batch(4).repeat()
        self.handle = tf.placeholder(tf.string, shape=[])
        iterator = tf.data.Iterator.from_string_handle(self.handle, dataset.output_types, dataset.output_shapes)
        self.get_next_batch = iterator.get_next()


        self.training_iterator = dataset.make_initializable_iterator()


    def load_data_from_files(self, file_list): # TODO make Queue structure to efficient file mangee
        curr_batch = np.load(file_list,allow_pickle=True)

        noise_rgb = np.array(curr_batch[2], dtype=np.float32) / 255.0 * 2 - 1
        ref_rgb = np.array(curr_batch[0], dtype=np.float32) / 255.0 * 2 - 1

        curr_batch[3] = np.array(curr_batch[3])
        curr_batch[1] = np.array(curr_batch[1])

        a = np.where(curr_batch[3] > 5)
        b = np.where(curr_batch[1] > 5)
        curr_batch[3][a] = curr_batch[3][a] - 5
        curr_batch[1][b] = curr_batch[1][b] - 5

        noise_action = np.eye(4)[curr_batch[3]]
        ref_action = np.eye(4)[curr_batch[1]]

        add_t = max_follow_length - len(noise_rgb)
        img_size = noise_rgb[0].shape[1]
        action_mask = np.ones_like(noise_action)

        noise_rgb = np.concatenate([noise_rgb, np.zeros([add_t, img_size, img_size, 3])], 0)
        noise_action = np.concatenate([noise_action, np.zeros([add_t, 4])], 0)
        action_mask = np.concatenate([action_mask, np.zeros([add_t, 4])], 0)

        ret = [ref_rgb, ref_action, noise_rgb, noise_action, action_mask]


        #self.batch_pointer = (self.batch_pointer+1)%self.num_batches

        return [np.array(x,dtype=np.float32) for x in ret]

