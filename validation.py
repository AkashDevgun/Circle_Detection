import numpy as np
from shapely.geometry.point import Point
from skimage.draw import circle_perimeter_aa
import torch
import torch.nn.parallel
import torch.optim
import torch.utils.data
import torch.utils.data.distributed
import torchvision.transforms as transforms
from network import CircleNet

#
model = CircleNet()
# #model = Net()
# checkpoint = torch.load('/media/HDD_2TB.1/saved_models/model_checkpoint.pth_r2.tar')
checkpoint = torch.load('/media/HDD_2TB.1/model.pth.tar')
model.load_state_dict(checkpoint)
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def draw_circle(img, row, col, rad):
    rr, cc, val = circle_perimeter_aa(row, col, rad)
    valid = (
            (rr >= 0) &
            (rr < img.shape[0]) &
            (cc >= 0) &
            (cc < img.shape[1])
    )
    img[rr[valid], cc[valid]] = val[valid]


def noisy_circle(size, radius, noise):
    img = np.zeros((size, size), dtype=np.float)

    # Circle
    row = np.random.randint(size)
    col = np.random.randint(size)
    rad = np.random.randint(10, max(10, radius))
    draw_circle(img, row, col, rad)

    # Noise
    img += noise * np.random.rand(*img.shape)
    return (row, col, rad), img


def find_circle(model, img):
    model.eval()

    with torch.no_grad():
        image = np.expand_dims(np.asarray(img), axis=0)
        image = torch.from_numpy(np.array(image, dtype=np.float32))
        normalize = transforms.Compose([
            transforms.Normalize([0.5], [0.5]),
        ])
        image = normalize(image)
        image = image.unsqueeze(0)
        image = image.to(device)
        output = model(image)

    return [round(i) for i in (200 * output).tolist()[0]]


def iou(params0, params1):
    row0, col0, rad0 = params0
    row1, col1, rad1 = params1

    shape0 = Point(row0, col0).buffer(rad0)
    shape1 = Point(row1, col1).buffer(rad1)

    return (
            shape0.intersection(shape1).area /
            shape0.union(shape1).area
    )


def validating(model):
    results = []
    for _ in range(1000):
        params, img = noisy_circle(200, 50, 2)
        detected = find_circle(model, img)
        results.append(iou(params, detected))
    results = np.array(results)
    return (results > 0.7).mean()
