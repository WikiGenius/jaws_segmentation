import torch
import matplotlib.pyplot as plt


def plot_test_sample(nets_dict, plane, test_loaders):
    '''
    plane: str = 'axial' | 'coronal' | 'sagittal'
    '''
    net = nets_dict[plane]
    net.eval()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    sample = iter(test_loaders[plane]).next()

    index = torch.randint(0, len(sample), (1,)).item()
    sample_img = sample['image'][index: index+1]
    sample_true = sample['mask'][index: index+1]
    with torch.no_grad():
        sample_pred = torch.argmax(net(sample_img.to(device)).cpu(), dim=1)

    plt.figure(figsize=(15, 4))

    plt.subplot(1, 3, 1)
    plt.title('img')
    plt.axis('off')
    plt.imshow(sample_img.squeeze(), cmap='bone')

    plt.subplot(1, 3, 2)
    plt.title('true label')
    plt.axis('off')
    plt.imshow(sample_true.squeeze())

    plt.subplot(1, 3, 3)
    plt.title('pred label')
    plt.axis('off')
    plt.imshow(sample_pred.detach().squeeze())

    plt.tight_layout()
