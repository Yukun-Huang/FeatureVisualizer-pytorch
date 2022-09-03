import PIL.Image
from torchvision.models import resnet50
from torchvision.transforms.functional import to_tensor, normalize

from feat_viz import FeatureVisualizer


def extract_features(x):
    net = resnet50(pretrained=True).eval()
    x = net.conv1(x)
    x = net.bn1(x)
    x = net.relu(x)
    x0 = net.maxpool(x)
    x1 = net.layer1(x0)
    x2 = net.layer2(x1)
    x3 = net.layer3(x2)
    x4 = net.layer4(x3)
    return x0, x1, x2, x3, x4


if __name__ == '__main__':
    # Load Data
    image_path = './demo/cat.jpg'
    img = PIL.Image.open(image_path)

    # Pre-process
    img = normalize(to_tensor(img), mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

    # Extract Features
    feats = extract_features(img.unsqueeze(dim=0))

    # Visualization
    V = FeatureVisualizer()

    for i, f in enumerate(feats):
        feat1_path = f'./demo/feat1{i}.jpg'
        feat2_path = f'./demo/feat2_{i}.jpg'
        mix_path = f'./demo/mix{i}.jpg'
        V.viz(f, size=img.shape[1:], save_path=feat1_path)
        V.viz(f, size=img.shape[1:], save_path=feat2_path, cmap='jet')
        V.mix(image_path, feat2_path, mix_path)
