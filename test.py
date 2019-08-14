import models
import torch
num_classes = 18
inputs = torch.rand([1,3,224,224])
test = models.resnet34(num_classes=num_classes, pretrained='imagenet')
assert test(inputs).size()[1] == num_classes
print('ok')
test = models.resnet50(num_classes=num_classes, pretrained='imagenet')
assert test(inputs).size()[1] == num_classes
print('ok')
test = models.resnet101(num_classes=num_classes, pretrained='imagenet')
assert test(inputs).size()[1] == num_classes
print('ok')
test = models.resnet152(num_classes=num_classes, pretrained='imagenet')
assert test(inputs).size()[1] == num_classes
print('ok')
test = models.alexnet(num_classes=num_classes, pretrained='imagenet')
assert test(inputs).size()[1] == num_classes
print('ok')
test = models.densenet121(num_classes=num_classes, pretrained='imagenet')
assert test(inputs).size()[1] == num_classes
print('ok')
test = models.densenet169(num_classes=num_classes, pretrained='imagenet')
assert test(inputs).size()[1] == num_classes
print('ok')
test = models.densenet201(num_classes=num_classes, pretrained='imagenet')
assert test(inputs).size()[1] == num_classes
print('ok')
test = models.densenet201(num_classes=num_classes, pretrained='imagenet')
assert test(inputs).size()[1] == num_classes
print('ok')

test = models.inceptionv3(num_classes=num_classes, pretrained='imagenet')
assert test(torch.rand([2, 3, 299, 299]))[0].size()[1] == num_classes
print('ok')
assert test(torch.rand([2, 3, 299, 299]))[1].size()[1] == num_classes
print('ok')

test = models.vgg16(num_classes=num_classes, pretrained='imagenet')
assert test(torch.rand([1,3,224,224])).size()[1]  == num_classes
print('ok')
test = models.vgg16_bn(num_classes=num_classes, pretrained=None)
assert test(torch.rand([1,3,224,224])).size()[1]  == num_classes
print('ok')
