import torch
import timm

print(torch.cuda.is_available())
model = timm.create_model("vit_large_patch14_reg4_dinov2.lvd142m", num_classes=6, pretrained=True, in_chans=1).cuda()
print('Number of parameters: {}'.format(sum([p.data.nelement() for p in model.parameters()])))
x = torch.rand((2, 1, 518, 518)).cuda()
y = model(x)
# y = y.mean(dim=1)
print(y.shape)

print(model)