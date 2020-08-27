import torch
import module.units.arcface_module as arcface_module
import module.units.cosface_module as cosface_module
from facenet_pytorch import InceptionResnetV1, fixed_image_standardization


class ArcFace(object):
    def __init__(self, device, classnum, pretrained_path=None):
        self.arcface = arcface_module.ArcFace(classnum=classnum).to(device)
        self.arcface.load_state_dict(torch.load(pretrained_path))
        self.arcface.eval()

    def forward(self, input):
        # costrain input \in [0, 1]
        input = 2.0 * input - 1.0
        logit = self.arcface(input)
        return logit


class CosFace(object):
    def __init__(self, device, classnum, pretrained_path=None):
        self.cosface = cosface_module.CosFace(classnum=classnum).to(device)
        self.cosface.load_state_dict(torch.load(pretrained_path))
        self.cosface.eval()

    def forward(self, input):
        # costrain input \in [0, 1]
        input = 2.0 * input - 1.0
        logit = self.cosface(input)
        return logit


class FaceNet(object):

    def __init__(self, device, classnum, pretrained_path=None):
        self.resnet = InceptionResnetV1(classify=True, num_classes=classnum).to(device)
        self.resnet.load_state_dict(torch.load(pretrained_path))
        self.resnet.eval()

    def forward(self, input):
        # contrain input \in [0, 1]
        input = input * 255.0
        source = fixed_image_standardization(input)
        logit = self.resnet(source)
        return logit
