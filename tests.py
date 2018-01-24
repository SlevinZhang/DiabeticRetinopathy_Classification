from vgg16 import VGG_DR
from resnet50 import RESNET_DR
"""
Set of tests to run. Note different kwargs for each test
"""

#Training for base benchmarks
#model = RESNET_DR('resnet50_lastlayer', **{'freeze':-2})
#model.train()
#
#model = RESNET_DR('resnet50_last2layers', **{'freeze':-3})
#model.train()
#
#model = RESNET_DR('resnet50_alllayers', **{'freeze':0})
#model.train()
#
#
## Data augmentation benchmarks
#model = RESNET_DR('resnet50_lastlayer_flipaug', **{'freeze':-2, 'vertical_flip': True,
#                                                   'horizontal_flip': True})
#model.train()
#
#model = RESNET_DR('resnet50_last2layers_flipaug', **{'freeze':-3, 'vertical_flip': True,
#                                                   'horizontal_flip': True})
#model.train()
#
#model = RESNET_DR('resnet50_alllayer_flipaug', **{'freeze':0, 'vertical_flip': True,
#                                                   'horizontal_flip': True})
#model.train()
#
## Standardized data
#model = RESNET_DR('resnet50_lastlayer_flipaug_std', **{'freeze':-2, 'vertical_flip': True,
#                                                   'horizontal_flip': True,
#                                                   'featurewise_center':True,
#                                                   'featurewise_std_normalization':True})
#model.train()
#
model = RESNET_DR('resnet50_last2layers_flipaug_std', **{'freeze':-3, 'vertical_flip': True,
                                                   'horizontal_flip': True,
                                                   'featurewise_center':True,
                                                   'featurewise_std_normalization':True})
model.train()

model = RESNET_DR('resnet50_alllayer_flipaug_std', **{'freeze':0, 'vertical_flip': True,
                                                   'horizontal_flip': True,
                                                   'featurewise_center':True,
                                                   'featurewise_std_normalization':True})
model.train()


# Testing trained models
#print("Running tests on Resnet50 Models")
#
#model = RESNET_DR('resnet50_lastlayer', **{'freeze':-2, 'load_model':True})
#model.test()
#
#model = RESNET_DR('resnet50_last2layers', **{'freeze':-3, 'load_model':True})
#model.test()
#
#model = RESNET_DR('resnet50_alllayers', **{'freeze':0, 'load_model':True})
#model.test()
#
#model = RESNET_DR('resnet50_lastlayer_flipaug', **{'freeze':-2, 'vertical_flip': True,
#                                                   'horizontal_flip': True, 'load_model':True})
#model.test()
#
#model = RESNET_DR('resnet50_last2layers_flipaug', **{'freeze':-3, 'vertical_flip': True,
#                                                   'horizontal_flip': True,
#                                                   'load_model':True})
#model.test()
#
#model = RESNET_DR('resnet50_alllayer_flipaug', **{'freeze':0, 'vertical_flip': True,
#                                                   'horizontal_flip': True,
#                                                   'load_model':True})
#model.test()
#
#model = RESNET_DR('resnet50_lastlayer_flipaug_std', **{'freeze':-2, 'vertical_flip': True,
#                                                   'horizontal_flip': True,
#                                                   'featurewise_center':True,
#                                                   'featurewise_std_normalization':True,
#                                                   'load_model':True})
#model.test()
