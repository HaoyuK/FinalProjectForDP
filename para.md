# CNN
* parameters: 43485992
```
model:
  type: 'cnn'
  in_channels: 3
  img_size: 224
  num_conv_layers: 3
  num_filters: [32, 64, 108]
  linear_dim: 768
  kernel_size: 3
  num_classes: 100
```

# Vit
* parameters: 43348324
```
model:
  type: 'transformer'  # 改为 'image_transformer'
  in_channels: 3 # 输入图片的通道
  patch_size: 16  # 设定 patch 尺寸，例如 4x4
  emb_size: 768 # 嵌入维数
  img_size: 224  # 设定图像尺寸，例如 32x32
  depth: 6 # transformer层数
  num_heads: 8 # 头数
  num_classes: 100 # 分类数
```

