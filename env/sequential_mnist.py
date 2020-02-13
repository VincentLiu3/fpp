from torchvision import datasets, transforms
import torch


class MnistDataset(torch.utils.data.Dataset):
    """
    inherit torch.utils.data.Dataset (https://pytorch.org/docs/stable/_modules/torch/utils/data/dataset.html#Dataset)
    """
    def __init__(self, dataset_folder):
        self.mnist_dataset = datasets.MNIST(dataset_folder, train=True, download=True,
                                            transform=transforms.Compose([
                                                transforms.ToTensor(),
                                                transforms.Normalize((0.5,), (0.5,)),
                                                transforms.Lambda(lambda x: x.view(28, 28))
                                            ]))

    def __getitem__(self, index):
        """
        :return
        x: a row of image with shape (1, 1, 28)
        y: a label with shape (1)
        """
        photo_id = index // 28
        row_id = index % 28
        # print(photo_id)
        # print(row_id)
        x, y = self.mnist_dataset[photo_id]
        x = x[row_id].view(1, 1, 28)
        y = torch.tensor([y])
        return x, y

    def get_image(self, index):
        """
        :return
        x: an image with shape (28, 1, 28)
        y: a label with shape (28)
        """
        photo_id = index // 28
        x, y = self.mnist_dataset[photo_id]
        x = x.view(28, 1, 28)
        # y = torch.tensor([y])
        y = torch.ones([28]).long() * y
        return x, y


if __name__ == '__main__':
    mnist_dataset = MnistDataset(dataset_folder='../data')

    for i in range(30):
        x, y = mnist_dataset[i]
        print(x.size())
        print(x)
        print(y)
