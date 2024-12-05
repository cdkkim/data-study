
import requests
import pdb
from torchvision import datasets, transforms


transform=transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
])
dataset = datasets.MNIST('./data', train=False, transform=transform)

data_list = []

for i in range(len(dataset)):
    if i > 10:
        continue

    image, label = dataset[i]

    # Convert the image to a flattened list (1D array of 784 elements)
    image_array = image.view(-1).numpy().tolist()

    res = requests.post('http://localhost:8000/mnist', json={'image': image_array}).json()
    print('Correct:', label, '\t', 'Predicted:', res)

pdb.set_trace()

