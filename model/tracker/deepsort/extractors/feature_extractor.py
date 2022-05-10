import cv2
import numpy as np
import torch
import torchvision.transforms as transforms

class DefaultExtractor(object):

    def __init__(self, checkpoint_path, mult=None, pretrained='market1501'):
        from model.tracker.deepsort.extractors .model import Net

        self.net = Net(reid=True)
        self.device = torch.device(device='cpu')
        state_dict = torch.load(
            checkpoint_path,
            map_location=lambda storage, loc: storage)['net_dict']
        self.net.load_state_dict(state_dict)
        self.size = (64, 128)
        self.norm = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ])

    def to(self, device):
        self.device = torch.device(device=device)
        self.net.to(self.device)

    def _preprocess(self, im_crops):
        """
        TODO:
            1. to float with scale from 0 to 1
            2. resize to (64, 128) as Market1501 dataset did
            3. concatenate to a numpy array
            3. to torch Tensor
            4. normalize
        """

        def _resize(im, size):
            return cv2.resize(im.astype(np.float32) / 255., size)

        im_batch = torch.cat([
            self.norm(_resize(im, self.size)).unsqueeze(0) for im in im_crops
        ],
                             dim=0).float()
        return im_batch

    def __call__(self, im_crops):
        im_batch = self._preprocess(im_crops)
        with torch.no_grad():
            im_batch = im_batch.to(self.device)
            features = self.net(im_batch)
        return features.cpu().numpy()

class MobilenetV2Extractor(object):

    def __init__(self, checkpoint_path, mult=1.0, pretrained='market1501'):

        if mult == 1.0:
            from .mobilenet import mobilenetv2_x1_0
            self.net = mobilenetv2_x1_0(
                pretrained=True,
                pretrained_dataset = pretrained
            )

        elif mult == 1.4:
            from .mobilenet import mobilenetv2_x1_4
            self.net = mobilenetv2_x1_4(
                pretrained=True,
                pretrained_dataset = pretrained
            )

        self.device = torch.device(device='cpu')

        self.size = (64, 128)
        self.norm = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ])

    def to(self, device):
        self.device = torch.device(device=device)
        self.net.to(self.device)

    def _preprocess(self, im_crops):
        """
        TODO:
            1. to float with scale from 0 to 1
            2. resize to (64, 128) as Market1501 dataset did
            3. concatenate to a numpy array
            3. to torch Tensor
            4. normalize
        """

        def _resize(im, size):
            return cv2.resize(im.astype(np.float32) / 255., size)

        im_batch = torch.cat([
            self.norm(_resize(im, self.size)).unsqueeze(0) for im in im_crops
        ],
                             dim=0).float()
        return im_batch

    def __call__(self, im_crops):
        im_batch = self._preprocess(im_crops)
        with torch.no_grad():
            im_batch = im_batch.to(self.device)
            features = self.net(im_batch)
        return features.cpu().numpy()

class OSNetExtractor(object):
    def __init__(self, checkpoint_path, mult=0.5, pretrained='market1501'):

        if mult == 0.25:
            from .osnet import osnet_x0_25
            self.net = osnet_x0_25(
                pretrained=True, 
                pretrained_dataset = pretrained,
                num_classes = 512
                )
        elif mult == 0.5:
            from .osnet import osnet_x0_5
            self.net = osnet_x0_5(
                pretrained=True,
                pretrained_dataset = pretrained
                )
        elif mult == 0.75:
            from .osnet import osnet_x0_75
            self.net = osnet_x0_75(
                pretrained=True,
                pretrained_dataset = pretrained
                )
        elif mult == 1.0:
            from .osnet import osnet_x1_0
            self.net = osnet_x1_0(
                pretrained=True,
                pretrained_dataset = pretrained
            )
        self.net.training=False
        self.device = torch.device(device='cpu')

        self.size = (64, 128)
        self.norm = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ])

    def to(self, device):
        self.device = torch.device(device=device)
        self.net.to(self.device)

    def _preprocess(self, im_crops):
        """
        TODO:
            1. to float with scale from 0 to 1
            2. resize to (64, 128) as Market1501 dataset did
            3. concatenate to a numpy array
            3. to torch Tensor
            4. normalize
        """

        def _resize(im, size):
            return cv2.resize(im.astype(np.float32) / 255., size)

        im_batch = torch.cat([
            self.norm(_resize(im, self.size)).unsqueeze(0) for im in im_crops
        ],
                             dim=0).float()
        return im_batch

    def __call__(self, im_crops):
        im_batch = self._preprocess(im_crops)
        with torch.no_grad():
            im_batch = im_batch.to(self.device)
            features = self.net(im_batch)
        return features.cpu().numpy()
