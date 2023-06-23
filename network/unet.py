import torch
import torch.nn as nn
import torch.nn.functional as F
#
# __all__ = ["UNet2D"]
# class InitWeights_He(object):
#     def __init__(self, neg_slope=1e-2):
#         self.neg_slope = neg_slope
#
#     def __call__(self, module):
#         if isinstance(module, nn.Conv3d) or isinstance(module, nn.Conv2d) or isinstance(module, nn.ConvTranspose2d)\
#             or isinstance(module, nn.ConvTranspose3d):
#             module.weight = nn.init.kaiming_normal_(module.weight, a=self.neg_slope)
#             if module.bias is not None:
#                 module.bias = nn.init.constant_(module.bias, 0)
#
#
# class encoder(nn.Module):
#     def __init__(self, in_channels, initial_filter_size, kernel_size, do_instancenorm, droprate):
#         super().__init__()
#         self.contr_1_1 = self.contract(in_channels, initial_filter_size, kernel_size, instancenorm=do_instancenorm)
#         self.contr_1_2 = self.contract(initial_filter_size, initial_filter_size, kernel_size,
#                                        instancenorm=do_instancenorm)
#         self.pool = nn.MaxPool2d(2, stride=2)
#         self.dropout = nn.Dropout(p=droprate)
#
#         self.contr_2_1 = self.contract(initial_filter_size, initial_filter_size * 2, kernel_size,
#                                        instancenorm=do_instancenorm)
#         self.contr_2_2 = self.contract(initial_filter_size * 2, initial_filter_size * 2, kernel_size,
#                                        instancenorm=do_instancenorm)
#
#         self.contr_3_1 = self.contract(initial_filter_size * 2, initial_filter_size * 2 ** 2, kernel_size,
#                                        instancenorm=do_instancenorm)
#         self.contr_3_2 = self.contract(initial_filter_size * 2 ** 2, initial_filter_size * 2 ** 2, kernel_size,
#                                        instancenorm=do_instancenorm)
#
#         self.contr_4_1 = self.contract(initial_filter_size * 2 ** 2, initial_filter_size * 2 ** 3, kernel_size,
#                                        instancenorm=do_instancenorm)
#         self.contr_4_2 = self.contract(initial_filter_size * 2 ** 3, initial_filter_size * 2 ** 3, kernel_size,
#                                        instancenorm=do_instancenorm)
#         self.center = nn.Sequential(
#             nn.Conv2d(initial_filter_size * 2 ** 3, initial_filter_size * 2 ** 4, 3, padding=1),
#             nn.ReLU(inplace=True),
#             nn.Conv2d(initial_filter_size * 2 ** 4, initial_filter_size * 2 ** 4, 3, padding=1),
#             nn.ReLU(inplace=True)
#         )
#
#     def forward(self, x):
#         contr_1 = self.contr_1_2(self.contr_1_1(x))
#         pool = self.pool(contr_1)
#         pool = self.dropout(pool)
#
#         contr_2 = self.contr_2_2(self.contr_2_1(pool))
#         pool = self.pool(contr_2)
#         pool = self.dropout(pool)
#
#         contr_3 = self.contr_3_2(self.contr_3_1(pool))
#         pool = self.pool(contr_3)
#         pool = self.dropout(pool)
#
#         contr_4 = self.contr_4_2(self.contr_4_1(pool))
#         pool = self.pool(contr_4)
#         pool = self.dropout(pool)
#
#         out = self.center(pool)
#         return out, contr_4, contr_3, contr_2, contr_1
#
#     @staticmethod
#     def contract(in_channels, out_channels, kernel_size=3, instancenorm=True):
#         if instancenorm:
#             layer = nn.Sequential(
#                 nn.Conv2d(in_channels, out_channels, kernel_size, padding=1),
#                 nn.BatchNorm2d(out_channels),
#                 nn.LeakyReLU(inplace=True))
#         else:
#             layer = nn.Sequential(
#                 nn.Conv2d(in_channels, out_channels, kernel_size, padding=1),
#                 nn.LeakyReLU(inplace=True))
#         return layer
#
#
# class decoder(nn.Module):
#     def __init__(self, initial_filter_size, classes, droprate):
#         super().__init__()
#         self.dropout = nn.Dropout(p=droprate)
#         # self.concat_weight = torch.nn.Parameter(torch.FloatTensor(1), requires_grad=True)
#         self.upscale5 = nn.ConvTranspose2d(initial_filter_size * 2 ** 4, initial_filter_size * 2 ** 3, kernel_size=2,
#                                            stride=2)
#         self.expand_4_1 = self.expand(initial_filter_size * 2 ** 4, initial_filter_size * 2 ** 3)
#         self.expand_4_2 = self.expand(initial_filter_size * 2 ** 3, initial_filter_size * 2 ** 3)
#         self.upscale4 = nn.ConvTranspose2d(initial_filter_size * 2 ** 3, initial_filter_size * 2 ** 2, kernel_size=2,
#                                            stride=2)
#
#         self.expand_3_1 = self.expand(initial_filter_size * 2 ** 3, initial_filter_size * 2 ** 2)
#         self.expand_3_2 = self.expand(initial_filter_size * 2 ** 2, initial_filter_size * 2 ** 2)
#         self.upscale3 = nn.ConvTranspose2d(initial_filter_size * 2 ** 2, initial_filter_size * 2, 2, stride=2)
#
#         self.expand_2_1 = self.expand(initial_filter_size * 2 ** 2, initial_filter_size * 2)
#         self.expand_2_2 = self.expand(initial_filter_size * 2, initial_filter_size * 2)
#         self.upscale2 = nn.ConvTranspose2d(initial_filter_size * 2, initial_filter_size, 2, stride=2)
#
#         self.expand_1_1 = self.expand(initial_filter_size * 2, initial_filter_size)
#         self.expand_1_2 = self.expand(initial_filter_size, initial_filter_size)
#         self.head = nn.Sequential(
#             nn.Conv2d(initial_filter_size, classes, kernel_size=1,
#                       stride=1, bias=False))
#
#     def forward(self, x, contr_4, contr_3, contr_2, contr_1):
#         concat_weight = 1
#
#         upscale = self.upscale5(x)
#         upscale = self.dropout(upscale)
#         crop = self.center_crop(contr_4, upscale.size()[2], upscale.size()[3])
#         concat = torch.cat([upscale, crop * concat_weight], 1)
#         expand = self.expand_4_2(self.expand_4_1(concat))
#
#         upscale = self.upscale4(expand)
#         upscale = self.dropout(upscale)
#         crop = self.center_crop(contr_3, upscale.size()[2], upscale.size()[3])
#         concat = torch.cat([upscale, crop * concat_weight], 1)
#         expand = self.expand_3_2(self.expand_3_1(concat))
#
#         upscale = self.upscale3(expand)
#         upscale = self.dropout(upscale)
#         crop = self.center_crop(contr_2, upscale.size()[2], upscale.size()[3])
#         concat = torch.cat([upscale, crop * concat_weight], 1)
#         expand = self.expand_2_2(self.expand_2_1(concat))
#
#         upscale = self.upscale2(expand)
#         upscale = self.dropout(upscale)
#         crop = self.center_crop(contr_1, upscale.size()[2], upscale.size()[3])
#         concat = torch.cat([upscale, crop * concat_weight], 1)
#         expand = self.expand_1_2(self.expand_1_1(concat))
#
#         out = self.head(expand)
#         return out
#
#     @staticmethod
#     def center_crop(layer, target_width, target_height):
#         batch_size, n_channels, layer_width, layer_height = layer.size()
#         xy1 = (layer_width - target_width) // 2
#         xy2 = (layer_height - target_height) // 2
#         return layer[:, :, xy1:(xy1 + target_width), xy2:(xy2 + target_height)]
#
#     @staticmethod
#     def expand(in_channels, out_channels, kernel_size=3):
#         layer = nn.Sequential(
#             nn.Conv2d(in_channels, out_channels, kernel_size, padding=1),
#             nn.BatchNorm2d(out_channels),
#             nn.LeakyReLU(inplace=True),
#         )
#         return layer
#
#
# class UNet2D(nn.Module):
#     def __init__(self, in_channels=1, initial_filter_size=64, kernel_size=3, classes=16, do_instancenorm=True,
#                  droprate=0.1):
#         super().__init__()
#
#         self.encoder = encoder(in_channels, initial_filter_size, kernel_size, do_instancenorm, droprate)
#         self.decoder = decoder(initial_filter_size, classes, droprate)
#
#         self.apply(InitWeights_He(1e-2))
#
#     def forward(self, x):
#         x_1, contr_4, contr_3, contr_2, contr_1 = self.encoder(x)
#         out = self.decoder(x_1, contr_4, contr_3, contr_2, contr_1)
#
#         return out
#
#
# class UNet2D_classification(nn.Module):
#     def __init__(self, in_channels=1, initial_filter_size=64, kernel_size=3, classes=2, do_instancenorm=True,
#                  droprate=0.1):
#         super().__init__()
#
#         self.encoder = encoder(in_channels, initial_filter_size, kernel_size, do_instancenorm, droprate)
#
#         self.head = nn.Sequential(
#             nn.AdaptiveAvgPool2d((1, 1)),
#             nn.Flatten(),
#             nn.Linear(initial_filter_size * 2 ** 4, initial_filter_size * 2 ** 4),
#             nn.ReLU(inplace=True),
#             nn.Linear(initial_filter_size * 2 ** 4, classes)
#         )
#
#         self.apply(InitWeights_He(1e-2))
#
#     def forward(self, x):
#         x_1, _, _, _, _ = self.encoder(x)
#         out = self.head(x_1)
#         # print(x_1.shape)
#         return out
#
# if __name__ == '__main__':
#     model = UNet2D(in_channels=1, initial_filter_size=64, kernel_size=3, classes=16, do_instancenorm=True, droprate=0.1)
#     input = torch.randn(20,1,256,256)
#     out = model(input)
#     print('out shape:{}'.format(out.shape))
#
#     model_classification = UNet2D_classification(in_channels=1, initial_filter_size=64, kernel_size=3, classes=64, do_instancenorm=True, droprate=0.1)
#     out_classification = model_classification(input)
#     print('out_classification shape:{}'.format(out_classification.shape))
#     print('out_classification.unsqueeze.shape:{}'.format(out_classification.unsqueeze(1).shape))
#


__all__ = ["UNET2D", "UNET3D"]


def get_default_config(dim=2, droprate=None, nonlin="LeakyReLU", norm_type="bn"):
    props = {}
    if dim == 2:
        props["dim"] = 2
        props["conv_op"] = nn.Conv2d
        props["convtrans_op"] = nn.ConvTranspose2d
        props["maxpool_op"] = nn.MaxPool2d
        props["adaptpool_op"] = nn.AdaptiveAvgPool2d
        props["dropout_op"] = nn.Dropout2d
    elif dim == 3:
        props["dim"] = 3
        props["conv_op"] = nn.Conv3d
        props["convtrans_op"] = nn.ConvTranspose3d
        props["maxpool_op"] = nn.MaxPool3d
        props["adaptpool_op"] = nn.AdaptiveAvgPool3d
        props["dropout_op"] = nn.Dropout3d
    else:
        raise NotImplementedError

    props["conv_op_kwargs"] = {"kernel_size": 3, "stride": 1, "padding": 1, "dilation": 1, "bias": True}
    props["convtrans_op_kwargs"] = {"kernel_size": 2, "stride": 2}
    props["maxpool_op_kwargs"] = {"kernel_size": 2, "stride": 2}
    props["adaptpool_op_kwargs"] = {"output_size": 1}


    if droprate is None:
        props["dropout_op"] = None
        props["dropout_op_kwargs"] = {"p": 0, "inplace": True}
    else:
        props["dropout_op_kwargs"] = {"p": droprate, "inplace": True}

    if nonlin == "LeakyReLU":
        props["nonlin_op"] = nn.LeakyReLU
        props["nonlin_op_kwargs"] = {"negative_slope": 1e-2, "inplace": True}
    elif nonlin == "ReLu":
        props["nonlin_op"] = nn.ReLU
        props["nonlin_op_kwargs"] = {"inplace": True}
    else:
        raise ValueError

    if norm_type == "bn":
        if dim == 2:
            props["norm_op"] = nn.BatchNorm2d
        elif dim == 3:
            props["norm_op"] = nn.BatchNorm3d
        props["norm_op_kwargs"] = {"eps": 1e-5, "affine": True}
    elif norm_type == "in":
        if dim == 2:
            props["norm_op"] = nn.InstanceNorm2d
        elif dim == 3:
            props["norm_op"] = nn.InstanceNorm3d
        props["norm_op_kwargs"] = {"eps": 1e-5, "affine": True}
    elif norm_type is None:
        props["norm_op"] = None
    else:
        raise NotImplementedError

    return props


class InitWeights_He(object):
    def __init__(self, neg_slope=1e-2):
        self.neg_slope = neg_slope

    def __call__(self, module):
        if isinstance(module, nn.Conv3d) or isinstance(module, nn.Conv2d) or isinstance(module, nn.ConvTranspose2d) \
                or isinstance(module, nn.ConvTranspose3d):
            module.weight = nn.init.kaiming_normal_(module.weight, a=self.neg_slope)
            if module.bias is not None:
                module.bias = nn.init.constant_(module.bias, 0)


class encoder(nn.Module):
    def __init__(self, input_feature=1, base_feature=64, props=None):
        super().__init__()
        # input_feature = input_feature
        # base_feature = base_feature
        self.props = props
        #print(self.props)
        self.do_norm = self.props["norm_op"]

        self.block1 = self.block(input_feature, base_feature)
        self.block2 = self.block(base_feature, base_feature*2)
        self.block3 = self.block(base_feature*2, base_feature*2**2)
        self.block4 = self.block(base_feature*2**2, base_feature*2**3)
        self.block5 = self.block(base_feature*2**3, base_feature*2**4)
        self.maxpool = self.props["maxpool_op"](**self.props["maxpool_op_kwargs"])
        self.dropout = self.props["dropout_op"](**self.props["dropout_op_kwargs"])

    def forward(self, x):
        skips = []
        for block in [self.block1, self.block2, self.block3, self.block4]:
            #print(block)
            x = block(x)
            skips.append(x)
            x = self.maxpool(x)
            x = self.dropout(x)
        x = self.block5(x)
        return x, skips[::-1]

    def block(self, in_channels, out_channels):
        if self.do_norm:
            block = nn.Sequential(
                self.props["conv_op"](in_channels=in_channels, out_channels=out_channels, **self.props["conv_op_kwargs"]),
                self.props["norm_op"](num_features=out_channels),
                self.props["nonlin_op"](**self.props["nonlin_op_kwargs"]),
                self.props["conv_op"](in_channels=out_channels, out_channels=out_channels, **self.props["conv_op_kwargs"]),
                self.props["norm_op"](num_features=out_channels),
                self.props["nonlin_op"](**self.props["nonlin_op_kwargs"])
            )
        else:
            block = nn.Sequential(
                self.props["conv_op"](in_channels=in_channels, out_channels=out_channels, **self.props["conv_op_kwargs"]),
                self.props["nonlin_op"](**self.props["nonlin_op_kwargs"]),
                self.props["conv_op"](in_channels=out_channels, out_channels=out_channels, **self.props["conv_op_kwargs"]),
                self.props["nonlin_op"](**self.props["nonlin_op_kwargs"])
            )

        return block


class decoder(nn.Module):
    def __init__(self, base_feature=64, num_classes=16, props=None):
        super().__init__()
        # base_feature = base_feature
        # num_classes = num_classes
        self.props = props
        self.do_norm = self.props["norm_op"]
        self.up1 = self.props["convtrans_op"](base_feature*2**4, base_feature*2**3, **self.props["convtrans_op_kwargs"])
        self.block1 = self.block(base_feature*2**4, base_feature*2**3)
        self.up2 = self.props["convtrans_op"](base_feature*2**3, base_feature*2**2, **self.props["convtrans_op_kwargs"])
        self.block2 = self.block(base_feature*2**3, base_feature*2**2)
        self.up3 = self.props["convtrans_op"](base_feature*2**2, base_feature*2, **self.props["convtrans_op_kwargs"])
        self.block3 = self.block(base_feature*2**2, base_feature*2)
        self.up4 = self.props["convtrans_op"](base_feature*2, base_feature, **self.props["convtrans_op_kwargs"])
        self.block4 = self.block(base_feature*2, base_feature)
        self.out = nn.Sequential(
            self.props["conv_op"](base_feature, num_classes, kernel_size=1, stride=1, bias=False),
            self.props["nonlin_op"](**self.props["nonlin_op_kwargs"])
        )
        self.dropout = self.props["dropout_op"](**self.props["dropout_op_kwargs"])

    def forward(self, x, skips):
        for (up, skip, block) in zip([self.up1, self.up2, self.up3, self.up4], skips, [self.block1, self.block2, \
                                                                                       self.block3, self.block4]):
            x = up(x)
            x = self.dropout(x)
            skip = self.crop(skip, x)
            x = torch.cat([skip, x], 1)
            x = block(x)
        x = self.out(x)
        return x

    def crop(self, old_img, target_img):
        if self.props["dim"] == 2:
            b1, c1, w1, h1 = old_img.shape
            b2, c2, w2, h2 = target_img.shape
            wd = (w1 - w2) // 2
            hd = (h1 - h2) // 2
            img = old_img[:, :, wd:w2+wd, hd:h2+hd]
        elif self.props["dim"] == 3:
            b1, c1, w1, h1, t1 = old_img.shape
            b2, c2, w2, h2, t2 = target_img.shape
            wd = (w1 - w2) // 2
            hd = (h1 - h2) // 2
            td = (t1 - t2) // 2
            img = old_img[:, :, wd:w2+wd, hd:h2+hd, td:t2+td]
        else:
            raise NotImplementedError
        return img

    def block(self, in_channels, out_channels):
        if self.do_norm:
            block = nn.Sequential(
                self.props["conv_op"](in_channels=in_channels, out_channels=out_channels, **self.props["conv_op_kwargs"]),
                self.props["norm_op"](num_features=out_channels),
                self.props["nonlin_op"](**self.props["nonlin_op_kwargs"]),
                self.props["conv_op"](in_channels=out_channels, out_channels=out_channels, **self.props["conv_op_kwargs"]),
                self.props["norm_op"](num_features=out_channels),
                self.props["nonlin_op"](**self.props["nonlin_op_kwargs"])
            )
        else:
            block = nn.Sequential(
                self.props["conv_op"](in_channels=in_channels, out_channels=out_channels, **self.props["conv_op_kwargs"]),
                self.props["nonlin_op"](**self.props["nonlin_op_kwargs"]),
                self.props["conv_op"](in_channels=out_channels, out_channels=out_channels, **self.props["conv_op_kwargs"]),
                self.props["nonlin_op"](**self.props["nonlin_op_kwargs"])
            )

        return block


class UNET_classification(nn.Module):
    def __init__(self, input_feature=1, base_feature=64, vector_feature=64, props=None):
        super().__init__()
        self.encoder = encoder(input_feature, base_feature, props)
        self.mlp = nn.Sequential(
            props["adaptpool_op"](**props["adaptpool_op_kwargs"]),
            nn.Flatten(),
            nn.Linear(base_feature*2**4, base_feature*2**4),
            props["nonlin_op"](**props["nonlin_op_kwargs"]),
            nn.Linear(base_feature*2**4, vector_feature)
        )
        self.apply(InitWeights_He(1e-2))

    def forward(self, x):
        x, skips = self.encoder(x)
        x = self.mlp(x)
        return x


class UNET(nn.Module):
    def __init__(self, input_feature=1, base_feature=64, num_classes=16, props=None):
        super().__init__()
        self.encoder = encoder(input_feature, base_feature, props)
        self.decoder = decoder(base_feature, num_classes, props)
        self.apply(InitWeights_He(1e-2))

    def forward(self, x):
        x, skips = self.encoder(x)
        x = self.decoder(x, skips)
        return x


if __name__ == '__main__':
    props = get_default_config(dim=3, droprate=0.1, nonlin="LeakyReLU", norm_type="bn")
    model3d_classification = UNET_classification(input_feature=1, base_feature=64, vector_feature=64, props=props)
    input3d = torch.randn(5,1,256,256,256)
    out3d_classification = model3d_classification(input3d)
    print('out3d_classification.shape:{}'.format(out3d_classification.shape))

    model_3d = UNET(input_feature=1, base_feature=64, num_classes=16, props=props)
    out_3d = model_3d(input3d)
    print('out_3d.shape:{}'.format(out_3d.shape))
    print('out_3d.unsqueeze.shape:{}'.format(out_3d.unsqueeze(1).shape))

