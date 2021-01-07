import torchvision
import torch
import torch.nn.functional as F

def mobilenet_v2(pretrained=True):
    # get mobilenet_v2 from torchvision models
    model = torchvision.models.mobilenet_v2(pretrained=pretrained)
    return model


def densenet(pretrained=True):
    # get mobilenet_v2 from torchvision models
    model = torchvision.models.densenet161(pretrained=pretrained)
    return model

class TransferredNetworkD(torch.nn.Module):
    def __init__(self, pre_trained_model, num_classes):
        super(TransferredNetworkD, self).__init__()
        self.pretrained_model = pre_trained_model

        self.classification_layer = torch.nn.Linear(in_features=pre_trained_model.classifier.in_features,
                                                     out_features=num_classes)


    def forward(self, x):
        # extract the features from the model
        x = self.pretrained_model.features(x)
        x = F.relu(x, inplace=True)
        pooled_features = F.adaptive_avg_pool2d(x, (1, 1)).view(x.size(0), -1)
        x = self.classification_layer(pooled_features)
        # to extract the embedding to visualize the tsne on for classes
        return pooled_features, x


class TransferredNetworkM(torch.nn.Module):
    def __init__(self, pre_trained_model, num_classes):
        super(TransferredNetworkM, self).__init__()
        self.pretrained_model = pre_trained_model

        # just a linear linear is enough since nn.CrossEntropy combines Softmax and NLLLoss for efficiency
        self.classification_layer = torch.nn.Linear(in_features=pre_trained_model.classifier[1].in_features,
                                                     out_features=num_classes)

    def forward(self, x):
        x = self.pretrained_model.features(x)
        pooled_features = x.mean([2, 3])
        x = self.classification_layer(pooled_features)
        return pooled_features, x

def make(model_name, num_classes):

    if model_name == 'densenet':
        model = densenet(pretrained=True)
        model = TransferredNetworkD(pre_trained_model=model,
                                    num_classes=num_classes
                                    )
        return model
    elif model_name == 'mobilenet':
        model = mobilenet_v2(pretrained=True)
        model = TransferredNetworkM(pre_trained_model=model,
                                    num_classes=num_classes
                                    )
        return model
