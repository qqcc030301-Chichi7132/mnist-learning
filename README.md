# mnist-learning

这是我学习 PyTorch 和 MNIST 手写数字识别的练习仓库。

训练代码 `main.py` 和依赖清单来自 [PyTorch 官方 examples/mnist](https://github.com/pytorch/examples/tree/main/mnist)；原项目的 BSD 3-Clause 许可见 `LICENSE`。我在 Mac 上训练 1 轮后，测试准确率为 9836/10000（约 98%）。


## 在 Mac 上运行

下载仓库后，在终端进入这个项目文件夹，依次运行：

```bash
/Library/Frameworks/Python.framework/Versions/3.13/bin/python3.13 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

用浏览器打开 `draw-digit.html`，画一个数字并点“保存图片”。然后运行：

```bash
python predict_digit.py "$HOME/Downloads/my-digit.png"
```

如果想重新训练模型，运行：

```bash
python main.py --no-accel --epochs 1 --save-model
```
