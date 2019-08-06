import numpy as np
import joblib
import quaternion as q

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

def preprocess_and_save_demo(demonstration_data, file_name, preprocess=True):
    print(' {} EPISODES OF DATA COLLECTED'.format(len(demonstration_data)))

    save_list = []
    data_to_save = dict()

    for ep in demonstration_data:
        demo_data = ep[0]
        if len(demo_data['rgb']) < 30 : continue
        noise_data = ep[1]
        end_pos = demo_data['position'][-1]
        end_quat = demo_data['rotation'][-1]

        norms = np.linalg.norm(noise_data['position'] - end_pos, axis=1)
        min_dist_idx = np.argmin(norms)

        rots = arr_quat_to_rots(noise_data['rotation'])
        end_rots = arr_quat_to_rots(end_quat)

        end_rot_norms = rots - end_rots
        end_rot_norms[np.where(end_rot_norms > 3.14)] -= 6.28
        end_rot_norms[np.where(end_rot_norms < -3.14)] += 6.28

        if len(noise_data['rgb']) > max_follow_length : continue
        if norms[min_dist_idx] < 0.4 and abs(end_rot_norms[min_dist_idx]) < 0.35:
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
            save_list.append(result)

    joblib.dump(save_list, file_name)
    print('{} saved'.format(file_name))
    return [file_name]




class data_loader():
    def __init__(self,batch_size=4):
        self.batch_size = batch_size
        self.maximum_size = 500

    def load_data_from_files(self, file_list): # TODO make Queue structure to efficient file mangee
        self.data_memory = []
        for f in file_list[::-1]: # load data files from back (FIFO order)
            self.data_memory.extend(joblib.load(f))
            if len(self.data_memory) > self.maximum_size : break
        print('total num of eps : {}'.format(len(self.data_memory)))

        self.num_batches = int(len(self.data_memory) / self.batch_size)
        self.data_memory = self.data_memory[:self.num_batches*self.batch_size]
        self.batch_pointer = 0


    def get_next_batch(self):
        #[ref_rgb, ref_action, noise_rgb, noise_action, action_mask] = self.data_memory[self.batch_pointer*self.batch_size : (self.batch_pointer+1)*self.batch_size]
        curr_batch = np.array(self.data_memory[self.batch_pointer * self.batch_size: (self.batch_pointer + 1) * self.batch_size])


        print(curr_batch[:,0][0].shape)
        print(curr_batch[:, 0][1].shape)
        print(curr_batch[:, 0][2].shape)
        print(curr_batch[:, 0][3].shape)
        print('\n')

        demo_seqs_list = np.stack(curr_batch[:,0])
        demo_acts_list = np.stack(curr_batch[:,1])
        inp_rgb_list = np.stack(curr_batch[:,2])
        inp_act_list = np.stack(curr_batch[:,3])
        act_mask_list = np.stack(curr_batch[:,4])

        ret = [demo_seqs_list, demo_acts_list,inp_rgb_list, inp_act_list, act_mask_list]

        self.batch_pointer = (self.batch_pointer+1)%self.num_batches

        return ret

    def drop_all_the_data(self):
        self.data_memory = []

