from C3D import C3D
import torch
import torch.nn as nn
from gdn import GDN
from torch.autograd import Function


class ReverseLayerF(Function):

    @staticmethod
    def forward(ctx, x, alpha):
        ctx.alpha = alpha

        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        output = grad_output.neg() * ctx.alpha

        return output, None


class Model(nn.Module):

    def __init__(self, out_size=128, pretrained=False):
        super(Model, self).__init__()
        self.out_size = out_size
        self.pretrained = pretrained
        self.device = torch.device("cuda")

        self.C3D = C3D(self.pretrained)

        self.class_classifier = nn.Sequential()
        self.class_classifier.add_module('c_fc1', nn.Linear(self.out_size, 100))
        # self.class_classifier.add_module('c_gdn1', GDN(100, self.device))
        self.class_classifier.add_module('c_bn1', nn.BatchNorm1d(100))
        self.class_classifier.add_module('c_relu1', nn.ReLU(True))
        self.class_classifier.add_module('c_drop1', nn.Dropout2d())
        self.class_classifier.add_module('c_fc2', nn.Linear(100, 100))
        self.class_classifier.add_module('c_bn2', nn.BatchNorm1d(100))
        self.class_classifier.add_module('c_relu2', nn.ReLU(True))
        self.class_classifier.add_module('c_fc3', nn.Linear(100, 5))
        self.class_classifier.add_module('c_softmax', nn.Softmax(dim=1))

        self.domain_classifier = nn.Sequential()
        self.domain_classifier.add_module('d_fc1', nn.Linear(self.out_size, 100))
        self.domain_classifier.add_module('d_bn1', nn.BatchNorm1d(100))
        self.domain_classifier.add_module('d_relu1', nn.ReLU(True))
        self.domain_classifier.add_module('d_fc2', nn.Linear(100, 2))
        self.domain_classifier.add_module('d_softmax', nn.LogSoftmax(dim=1))

    def forward(self, input_data, alpha=1.0):
        feature = self.C3D.forward(input_data)
        reverse_feature = ReverseLayerF.apply(feature, alpha)
        class_output = self.class_classifier(feature)
        domain_output = self.domain_classifier(reverse_feature)

        return class_output, domain_output


if __name__ == "__main__":
    inputs = torch.rand(1, 3, 20, 224, 224)
    model = Model()
    model.eval()
    class_output, domain_output = model.forward(inputs)

    print(class_output)
    print(domain_output)
