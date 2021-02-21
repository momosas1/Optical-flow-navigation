import os
import numpy as np
import cv2
import torch

import gibson2
import flow_vis

from gibson2.envs.locomotor_env import NavigateRandomEnv
from action_classifier.liteflownet.model import Network, estimate


class G:
    flow_network = None

def get_optical_flow(pre, last):
    '''
    use liteflownet to get flow
    input:pre is the last rgb picture;last is the current rgb picture
    output:dense flow picture;need flow_vis to convert flow array to rgb picture
    '''


    with torch.no_grad():
        first = torch.FloatTensor(np.ascontiguousarray(pre))
        second = torch.FloatTensor(np.ascontiguousarray(last))
        flow = estimate(first.permute(2, 0, 1), second.permute(2, 0, 1), G.flow_network).detach()
        #flow to rgb
        flow = flow_vis.flow_to_color(flow.numpy().transpose(1, 2, 0), convert_to_bgr=True)

    return flow / 255.0

class Env(NavigateRandomEnv):

    # New env
    def __init__(self,
                 config_file,
                 model_id=None,
                 mode='headless',
                 action_timestep=1 / 5.0,
                 physics_timestep=1 / 240.0,
                 automatic_reset=False,
                 device_idx=0,
                 render_to_tensor=False,
                 ):
        super(Env, self).__init__(config_file,
                                    model_id=model_id,
                                    mode=mode,
                                    action_timestep=action_timestep,
                                    physics_timestep=physics_timestep,
                                    automatic_reset=automatic_reset,
                                    device_idx=device_idx,
                                    render_to_tensor=render_to_tensor)
        self.pre_state_rgb = None
        self.pre_state_flow = None


    def reset(self):
        #save the pre rgb and flow picture

        state = super(Env, self).reset()
        self.pre_state_rgb = state['rgb']

        #first flow is white
        state['flow'] = np.ones((256, 256, 3), dtype=np.float32)*255.0
        self.pre_state_flow = state['flow']

        return state


def main():
    config_filename = os.path.join(os.path.dirname(gibson2.__file__), '../test/test_flow.yaml')
    nav_env = Env(config_file=config_filename, mode='headless')

    #initial liteflownet
    G.flow_network = Network().cuda().eval()
    for param in G.flow_network.parameters():
        param.requires_grad = False

    for j in range(4):
        nav_env.reset()
        count = 0

        for i in range(500):
            #sample action and step get new rgb
            action = nav_env.action_space.sample()
            state, reward, done, info = nav_env.step(action)
            tmp = nav_env.collision_step

            #calculate the flow of the sample action
            pre = nav_env.pre_state_rgb
            next = state['rgb']
            flow = get_optical_flow(pre, next)*255.0

            #update pre_state
            nav_env.pre_state_rgb = next
            tmp_flow = nav_env.pre_state_flow
            nav_env.pre_state_flow = flow


            #if collision don't save the picture and reset the env
            if tmp > count:
                print("collision")
                if tmp>10:
                    nav_env.reset()
                    count = 0
                count = tmp
                continue

            #write back flow
            cv2.imwrite(os.path.join("flow_p","action{}_{}.png".format(action,j*10000+i)),tmp_flow)
            cv2.imwrite(os.path.join("flow_c","action{}_{}.png".format(action,j*10000+i)),flow)

if __name__ == "__main__":
    main()