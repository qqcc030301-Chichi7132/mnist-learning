"""Use the trained PyTorch MNIST model to classify a drawing from draw-digit.html."""

import argparse
import sys
from pathlib import Path

import torch
from PIL import Image, ImageChops, ImageOps
from torchvision.transforms.functional import to_tensor


PROJECT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT))
from main import Net  # noqa: E402


def prepare(path: Path) -> torch.Tensor:
    image = Image.open(path).convert("L")
    # The board produces white ink on black, like MNIST. Also accept a dark
    # digit on white paper by flipping it when the corners are mostly light.
    corners = [image.getpixel(p) for p in [(0, 0), (image.width - 1, 0),
                                           (0, image.height - 1),
                                           (image.width - 1, image.height - 1)]]
    if sum(corners) / len(corners) > 127:
        image = ImageOps.invert(image)
    box = image.point(lambda value: 255 if value > 24 else 0).getbbox()
    if box is None:
        raise ValueError("图片里没有找到数字；请在画板上画好后重新保存。")
    image = image.crop(box)
    width, height = image.size
    scale = 20 / max(width, height)
    image = image.resize((max(1, round(width * scale)),
                          max(1, round(height * scale))), Image.Resampling.LANCZOS)
    canvas = Image.new("L", (28, 28), 0)
    canvas.paste(image, ((28 - image.width) // 2, (28 - image.height) // 2))
    # Center by ink mass to more closely match the MNIST preprocessing.
    bounds = canvas.getbbox()
    if bounds is not None:
        weights = list(canvas.get_flattened_data())
        mass = sum(weights)
        x_center = sum((index % 28) * weight for index, weight in enumerate(weights)) / mass
        y_center = sum((index // 28) * weight for index, weight in enumerate(weights)) / mass
        canvas = ImageChops.offset(canvas, round(13.5 - x_center), round(13.5 - y_center))
    return (to_tensor(canvas) - 0.1307) / 0.3081


def main() -> None:
    parser = argparse.ArgumentParser(description="识别一张手写数字图片")
    parser.add_argument("image", nargs="?", type=Path,
                        default=Path.home() / "Downloads" / "my-digit.png")
    args = parser.parse_args()
    model_file = PROJECT / "mnist_cnn.pt"
    if not model_file.is_file():
        parser.error(f"找不到训练好的模型：{model_file}")
    if not args.image.is_file():
        parser.error(f"找不到图片：{args.image}")
    model = Net()
    model.load_state_dict(torch.load(model_file, map_location="cpu", weights_only=True))
    model.eval()
    with torch.no_grad():
        scores = model(prepare(args.image).unsqueeze(0))
        probabilities = scores.exp().squeeze(0)
    print(f"识别结果：{probabilities.argmax().item()}")
    print("最可能的三个数字：")
    for probability, digit in zip(*torch.topk(probabilities, 3)):
        print(f"  {digit.item()}：{probability.item():.1%}")


if __name__ == "__main__":
    main()
