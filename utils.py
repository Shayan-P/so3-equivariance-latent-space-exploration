import os
import torch


def save_model(model, name):
    checkpoint_path = f'{get_checkpoints_dir()}/checkpoint_{name}.pt'
    torch.save(model.state_dict(), checkpoint_path)


def load_model(model, name):
    checkpoint_path = f'{get_checkpoints_dir()}/checkpoint_{name}.pt'
    checkpoint = torch.load(checkpoint_path)
    model.load_state_dict(checkpoint)


def get_checkpoints_dir():
    dir = os.path.join(os.path.dirname(__file__), 'checkpoints')
    if not os.path.exists(dir):
        os.makedirs(dir)
    return dir



class CustomLRScheduler:
    def __init__(self, optimizer, my_rl):
        self.optimizer = optimizer
        self.my_rl = my_rl
        self.step()

    def set_rl(self, my_rl):
        self.my_rl = my_rl

    def step(self):
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = self.my_rl