import math
import torch
from tensorboardX import SummaryWriter

def main():
    degrees = torch.arange(3600).resize_(3600, 1) * math.pi / 180.0
    labels = ["%d" % (i) for i in range(0, 3600)]

    with SummaryWriter() as writer:
        # Maybe make a bunch of data that's always shifted in some
        # way, and that will be hard for PCA to turn into a sphere?

        for epoch in range(0, 16):
            shift = epoch * 2 * math.pi / 16.0
            mat = torch.cat([ \
                torch.sin(shift + degrees * 2 * math.pi / 180.0), \
                torch.sin(shift + degrees * 3 * math.pi / 180.0), \
                torch.sin(shift + degrees * 5 * math.pi / 180.0), \
                torch.sin(shift + degrees * 7 * math.pi / 180.0), \
                torch.sin(shift + degrees * 11 * math.pi / 180.0) \
            ], dim=1)
            writer.add_embedding(mat=mat, metadata=labels, tag="sin", global_step=epoch)

            mat = torch.cat([ \
                torch.cos(shift + degrees * 2 * math.pi / 180.0), \
                torch.cos(shift + degrees * 3 * math.pi / 180.0), \
                torch.cos(shift + degrees * 5 * math.pi / 180.0), \
                torch.cos(shift + degrees * 7 * math.pi / 180.0), \
                torch.cos(shift + degrees * 11 * math.pi / 180.0) \
            ], dim=1)
            writer.add_embedding(mat=mat, metadata=labels, tag="cos", global_step=epoch)

            mat = torch.cat([ \
                torch.tan(shift + degrees * 2 * math.pi / 180.0), \
                torch.tan(shift + degrees * 3 * math.pi / 180.0), \
                torch.tan(shift + degrees * 5 * math.pi / 180.0), \
                torch.tan(shift + degrees * 7 * math.pi / 180.0), \
                torch.tan(shift + degrees * 11 * math.pi / 180.0) \
            ], dim=1)
            writer.add_embedding(mat=mat, metadata=labels, tag="tan", global_step=epoch)

if __name__ == "__main__":
    main()

# tensorboard --logdir runs
# Under "Projection, you should see
#  48 tensor found named
#     cos:cos-00000 to cos:cos-00016
#     sin:sin-00000 to sin:sin-00016
#     tan:tan-00000 to tan:tan-00016
