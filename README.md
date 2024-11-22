# aura-analyzer

aura analyzer is a project designed to perform seasonal color analysis and match users to their ideal color palette and a recommended outfit!

## installation (for windows)

it is highly recommended that you create a new virtual environment before installing dependencies. you can do this with conda, venv, or any other alternatives you see fit. here is an example of how to create and activate a conda environment:
```bash
conda create --name aura-analyzer
```
```bash
conda activate aura-analyzer
```

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
14. Pylette
15. mediapipe

install these dependencies using `pip`:

```bash
pip install -r requirements.txt
```

## run demo

1. navigate to combined_demo/scripts
```bash
cd combined_demo/scripts
```
2. run the following command, replacing <image-name.jpg> with an image of your choice (ensure that you add it to the input-imgs folder first)
```bash
./run_demo.sh image-name.jpg
```
3. for a sample image, run the following:
```bash
./run_demo.sh IMG_6119.jpg
```

## installation (for mac)

1. remove CUDA Toolkit from requirements.txt
2. run the following commands in aura-analyzer:
```bash
conda create -n aura-analyzer python=3.9 

source activate aura-analyzer

pip install -r requirements.txt

conda install -c nvidia cuda-python

cd combined_demo/scripts

chmod +x ./run_demo.sh

./run_demo.sh image-name.jpg
```