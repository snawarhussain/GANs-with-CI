import torch
from Network.Generator import Generator, get_noise
from utils.viz import show_tensor_images

DEVICE = 'cpu'
PATH = 'generator.pt'


def load_gen_model(Path=None, device='cpu'):
    gen = Generator(64)
    print(f"{'=' * 4} loading model state dict {'=' * 4}")
    model_state_dict = torch.load(Path, map_location=torch.device(device))
    print(f" {'=' * 4} loading model from checkpoint {'=' * 4}")
    gen.load_state_dict(model_state_dict)
    print('model loaded successfully !')
    return gen


def inference(model, device, z_dim=64, batch_size=25, ):
    fake_noise = get_noise(batch_size, z_dim, device=device)
    pred = model(fake_noise)
    return pred


def predict(path, device):
    generator = load_gen_model(path, device)
    prediction = inference(generator, device)
    show_tensor_images(prediction, show=True, save=True)


if __name__ == '__main__':
    predict(PATH, DEVICE)
