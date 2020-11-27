import torch
from torch import nn
import torch.nn.init as init
import os
from CNN-F import image_net


class GEN(torch.nn.Module):
    def __init__(self, image_dim, text_dim, hidden_dim, output_dim, class_dim, pretrain_model=None):
        super(GEN, self).__init__()
        self.module_name = 'GEN_module'
        self.output_dim = output_dim
        self.cnn_f = image_net(pretrain_model)
        self.image_module = nn.Sequential(
            nn.Linear(image_dim, hidden_dim//2, bias=True),
            nn.ReLU(True),
            nn.Linear(hidden_dim//2, hidden_dim // 4, bias=True),
            nn.ReLU(True)
        )

        self.text_module = nn.Sequential(
            nn.Linear(text_dim, hidden_dim, bias=True),
            nn.ReLU(True),
            nn.Linear(hidden_dim, hidden_dim // 2, bias=True),
            nn.ReLU(True),
            nn.Linear(hidden_dim // 2, hidden_dim // 4, bias=True),
            nn.ReLU(True)
        )

        self.hash_module = nn.ModuleDict({
            'image': nn.Sequential(
            nn.Linear(hidden_dim // 4, output_dim, bias=True),
            nn.Tanh()),
            'text': nn.Sequential(
            nn.Linear(hidden_dim // 4, output_dim, bias=True),
            nn.Tanh()),
        })

        self.classifier = nn.ModuleDict({
            'image': nn.Sequential(
                nn.Linear(hidden_dim // 4, class_dim, bias=True),
                nn.Sigmoid()
            ),
            'text': nn.Sequential(
                nn.Linear(hidden_dim // 4, class_dim, bias=True),
                nn.Sigmoid()
            ),
        })
        self.weight_init()

    def weight_init(self):
        initializer = self.kaiming_init
        for block in self._modules:
            for m in self._modules[block]:
                initializer(m)

    def kaiming_init(self, m):
        if isinstance(m, (nn.Linear, nn.Conv2d)):
            init.kaiming_normal_(m.weight)
            if m.bias is not None:
                m.bias.data.fill_(0)
        elif isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d)):
            m.weight.data.fill_(1)
            if m.bias is not None:
                m.bias.data.fill_(0)

    def forward(self, x, y):
        f_x = self.cnn_f(x).squeeze()
        f_x = self.image_module(x).squeeze()
        f_y = self.text_module(y)

        # normalization
        f_x = f_x / torch.sqrt(torch.sum(f_x.detach() ** 2))
        f_y = f_y / torch.sqrt(torch.sum(f_y.detach() ** 2))

        x_class = self.classifier['image'](f_x).squeeze()
        y_class = self.classifier['text'](f_y).squeeze()
        x_code = self.hash_module['image'](f_x).reshape(-1, self.output_dim)
        y_code = self.hash_module['text'](f_y).reshape(-1, self.output_dim)
        return x_code, y_code, f_x.squeeze(), f_y.squeeze(), x_class, y_class

    def generate_img_code(self, i):
        f_i = self.cnn_f(i).squeeze()
        f_i = self.image_module(i).squeeze()
        f_i = f_i / torch.sqrt(torch.sum(f_i.detach() ** 2))

        code = self.hash_module['image'](f_i.detach()).reshape(-1, self.output_dim)
        return code

    def generate_txt_code(self, t):
        f_t = self.text_module(t)
        f_t = f_t / torch.sqrt(torch.sum(f_t.detach() ** 2))

        code = self.hash_module['text'](f_t.detach()).reshape(-1, self.output_dim)
        return code

    def load(self, path, use_gpu=False):
        if not use_gpu:
            self.load_state_dict(torch.load(path, map_location=lambda storage, loc: storage))
        else:
            self.load_state_dict(torch.load(path))

    def save(self, name=None, path='./checkpoints', cuda_device=None):
        if not os.path.exists(path):
            os.makedirs(path)
        if cuda_device is not None:
            with torch.cuda.device(cuda_device):
                torch.save(self.state_dict(), os.path.join(path, name))
        else:
            torch.save(self.state_dict(), os.path.join(path, name))
        return name
