import numpy as np
from tensorboardX import SummaryWriter
from numpy.random import randint


def draw_fusilli(turns, radius, omega):
    points = []
    faces = []
    colors = []
    for t in range(turns):
        for theta in np.linspace(0, 2 * np.pi, 100, endpoint=False):
            z = (theta + 2 * np.pi * t) * omega
            end_point = radius * np.cos(theta), radius * np.sin(theta), z
            center_point = 0, 0, z
            points.append(center_point)
            points.append(end_point)


    # The frontend stays silent even if you assigned 
    # non-existing points, be careful.
    for n in range(0, len(points)-3, 2):
        faces.append((n, n+1, n+3))

    for _ in range(len(points)):
        colors.append((randint(100, 200),
                       randint(100, 200),
                       randint(100, 200)))

    return np.array([points]), np.array([colors]), np.array([faces])


with SummaryWriter() as w: 
    points, colors, faces = draw_fusilli(5, 1, 0.1)
    w.add_mesh("my_mesh1", points, colors, faces, global_step=0)
   
    for i in range(1, 10):
        points, colors, faces = draw_fusilli(randint(4, 7), 1, 0.1*randint(1,3))
        points += randint(-5, 5)
        w.add_mesh("my_mesh1", points, colors, faces, global_step=i)
