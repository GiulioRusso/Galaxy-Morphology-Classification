import timm


def build(name, num_classes=10, pretrained=True):
    """Any timm backbone with a fresh num_classes head.

    Zoobot encoders load the same way: name="hf_hub:mwalmsley/zoobot-encoder-convnext_tiny".
    """
    return timm.create_model(name, pretrained=pretrained, num_classes=num_classes)
