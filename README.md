# aura-analyzer

aura analyzer is a project designed to perform seasonal color analysis and match users to their ideal color palette and a recommended outfit!

## installation

before running the demo, make sure to install the following dependencies:

1. Python 3.6
2. PyTorch
3. Torchvision
4. CUDA Toolkit
5. TensorBoard (optional)
6. NumPy
7. Pillow
8. Future
9. tqdm
10. Matplotlib
11. SciPy
12. scikit-learn
13. OpenCV-Python

install these dependencies using `pip`:

```bash
pip install torch==1.5.0 torchvision==0.6.0 cudatoolkit numpy pillow future tqdm matplotlib scipy scikit-learn opencv-python tensorboard
```

## run demo

1. navigate to combined-demo/scripts
```bash
cd combined-demo/scripts
```
2. run the following command, replacing <image-name.jpg> with an image of your choice (ensure that you add it to the input-imgs folder first)
```bash
./run_demo.sh image-name.jpg
```
3. for a sample image, run the following:
```bash
./run_demo.sh IMG_6119.jpg
```