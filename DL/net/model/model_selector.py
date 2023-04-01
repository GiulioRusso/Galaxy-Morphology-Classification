from torch import nn
from torchvision import models
from torchvision.models import ResNet18_Weights
from torchvision.models import ResNet34_Weights
from torchvision.models import ResNet50_Weights
from torchvision.models import ResNet101_Weights
from torchvision.models import ResNet152_Weights
from torchvision.models import VGG11_Weights
from torchvision.models import VGG13_Weights
from torchvision.models import VGG16_Weights
from torchvision.models import VGG19_Weights
from torchvision.models import DenseNet121_Weights
from torchvision.models import DenseNet161_Weights
from torchvision.models import DenseNet169_Weights
from torchvision.models import DenseNet201_Weights


def model_selector(model_name, class_number):
    """
    :param model_name: string to select the desired network model
    :param class_number: number of class to classify
    :return: network model pretrained
    """

    if model_name == "resnet18":
        # get the pre-trained weights
        model = models.resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
        # change the last layer into 10 output neurons for out 10 classes problem
        model.fc = nn.Linear(in_features=model.fc.in_features, out_features=class_number)
        # apply a softmax activation to the output of the new fully connected layer to have a probability distribution
        model = nn.Sequential(model, nn.Softmax(dim=1))

    if model_name == "resnet34":
        # get the pre-trained weights
        model = models.resnet34(weights=ResNet34_Weights.IMAGENET1K_V1)
        # change the last layer into 10 output neurons for out 10 classes problem
        model.fc = nn.Linear(in_features=model.fc.in_features, out_features=class_number)
        # apply a softmax activation to the output of the new fully connected layer to have a probability distribution
        model = nn.Sequential(model, nn.Softmax(dim=1))

    if model_name == "resnet50":
        # get the pre-trained weights
        model = models.resnet50(weights=ResNet50_Weights.IMAGENET1K_V1)
        # change the last layer into 10 output neurons for out 10 classes problem
        model.fc = nn.Linear(in_features=model.fc.in_features, out_features=class_number)
        # apply a softmax activation to the output of the new fully connected layer to have a probability distribution
        model = nn.Sequential(model, nn.Softmax(dim=1))

    if model_name == "resnet101":
        # get the pre-trained weights
        model = models.resnet101(weights=ResNet101_Weights.IMAGENET1K_V1)
        # change the last layer into 10 output neurons for out 10 classes problem
        model.fc = nn.Linear(in_features=model.fc.in_features, out_features=class_number)
        # apply a softmax activation to the output of the new fully connected layer to have a probability distribution
        model = nn.Sequential(model, nn.Softmax(dim=1))

    if model_name == "resnet152":
        # get the pre-trained weights
        model = models.resnet152(weights=ResNet152_Weights.IMAGENET1K_V1)
        # change the last layer into 10 output neurons for out 10 classes problem
        model.fc = nn.Linear(in_features=model.fc.in_features, out_features=class_number)
        # apply a softmax activation to the output of the new fully connected layer to have a probability distribution
        model = nn.Sequential(model, nn.Softmax(dim=1))

    if model_name == "vgg11":
        # get the pre-trained weights
        model = models.vgg11(weights=VGG11_Weights.IMAGENET1K_V1)
        # change the last layer into 10 output neurons for out 10 classes problem
        model.classifier[6] = nn.Linear(in_features=model.classifier[6].in_features, out_features=class_number)
        # apply a softmax activation to the output of the new fully connected layer to have a probability distribution
        model = nn.Sequential(model, nn.Softmax(dim=1))

    if model_name == "vgg13":
        # get the pre-trained weights
        model = models.vgg13(weights=VGG13_Weights.IMAGENET1K_V1)
        # change the last layer into 10 output neurons for out 10 classes problem
        model.classifier[6] = nn.Linear(in_features=model.classifier[6].in_features, out_features=class_number)
        # apply a softmax activation to the output of the new fully connected layer to have a probability distribution
        model = nn.Sequential(model, nn.Softmax(dim=1))

    if model_name == "vgg16":
        # get the pre-trained weights
        model = models.vgg16(weights=VGG16_Weights.IMAGENET1K_V1)
        # change the last layer into 10 output neurons for out 10 classes problem
        model.classifier[6] = nn.Linear(in_features=model.classifier[6].in_features, out_features=class_number)
        # apply a softmax activation to the output of the new fully connected layer to have a probability distribution
        model = nn.Sequential(model, nn.Softmax(dim=1))

    if model_name == "vgg19":
        # get the pre-trained weights
        model = models.vgg19(weights=VGG19_Weights.IMAGENET1K_V1)
        # change the last layer into 10 output neurons for out 10 classes problem
        model.classifier[6] = nn.Linear(in_features=model.classifier[6].in_features, out_features=class_number)
        # apply a softmax activation to the output of the new fully connected layer to have a probability distribution
        model = nn.Sequential(model, nn.Softmax(dim=1))

    if model_name == "densenet121":
        # get the pre-trained weights
        model = models.densenet121(weights=DenseNet121_Weights.IMAGENET1K_V1)
        # change the last layer into 10 output neurons for out 10 classes problem
        model.classifier = nn.Linear(in_features=model.classifier.in_features, out_features=class_number)
        # apply a softmax activation to the output of the new fully connected layer to have a probability distribution
        model = nn.Sequential(model, nn.Softmax(dim=1))

    if model_name == "densenet161":
        # get the pre-trained weights
        model = models.densenet161(weights=DenseNet161_Weights.IMAGENET1K_V1)
        # change the last layer into 10 output neurons for out 10 classes problem
        model.classifier = nn.Linear(in_features=model.classifier.in_features, out_features=class_number)
        # apply a softmax activation to the output of the new fully connected layer to have a probability distribution
        model = nn.Sequential(model, nn.Softmax(dim=1))

    if model_name == "densenet169":
        # get the pre-trained weights
        model = models.densenet169(weights=DenseNet169_Weights.IMAGENET1K_V1)
        # change the last layer into 10 output neurons for out 10 classes problem
        model.classifier = nn.Linear(in_features=model.classifier.in_features, out_features=class_number)
        # apply a softmax activation to the output of the new fully connected layer to have a probability distribution
        model = nn.Sequential(model, nn.Softmax(dim=1))

    if model_name == "densenet201":
        # get the pre-trained weights
        model = models.densenet201(weights=DenseNet201_Weights.IMAGENET1K_V1)
        # change the last layer into 10 output neurons for out 10 classes problem
        model.classifier = nn.Linear(in_features=model.classifier.in_features, out_features=class_number)
        # apply a softmax activation to the output of the new fully connected layer to have a probability distribution
        model = nn.Sequential(model, nn.Softmax(dim=1))

    return model
