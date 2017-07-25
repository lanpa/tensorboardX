from tensorboard.embedding import add_embedding
import keyword
import torch
meta = []
while len(meta)<100:
    meta = meta+keyword.kwlist
meta = meta[:100]

for i, v in enumerate(meta):
    meta[i] = v+str(i)

label_img = torch.rand(100, 3, 10, 32)
for i in range(100):
    label_img[i]*=i/100

add_embedding(torch.randn(100, 5), save_path='embedding1', metadata=meta, label_img=label_img)
add_embedding(torch.randn(100, 5), save_path='embedding2', label_img=label_img)
add_embedding(torch.randn(100, 5), save_path='embedding3', metadata=meta)

#tensorboard --logdir embedding1