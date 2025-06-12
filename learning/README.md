# OldenborgModel

Train and perform inference on Oldenborg datasets.

## Architectures

Our flawed method for choosing architectures:

- See [Papers With Code benchmarks](https://paperswithcode.com/task/image-classification)
- Go to the [Hugging Face models listing](https://huggingface.co/models)
- Set the task (eg, "Image Classification")
- Sort by "Trending" and by "Most Downloads"
- Look at model sizes and accuracies
- Set the library (eg, "timm") and search for the model name

`timm` models for these experiments:

| Full Name                                  | Params | Nice Name      |
| ------------------------------------------ | -----: | -------------- |
| resnet18.a1_in1k                           |  11.7M | ResNet18       |
| mobilenetv4_conv_small.e2400_r224_in1k     |   3.8M | MobileNetV4    |
| efficientnet_b3.ra2_in1k                   |  12.2M | EfficientNet   |
| convnextv2_atto.fcmae                      |   3.4M | ConvNextV2Atto |
| convnextv2_base.fcmae_ft_in22k_in1k        |  88.7M | ConvNextV2Base |
| vit_base_patch16_224.augreg2_in21k_ft_in1k |  86.6M | ViT            |
