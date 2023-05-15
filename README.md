# Mean Shift Image Segmentation

This script performs image segmentation using the Mean Shift algorithm. It takes an input image and applies the Mean Shift algorithm to segment the image into distinct regions based on color similarity.

## Requirements

- Python 3.x
- OpenCV
- NumPy

## Usage

Run the script using the following command:

```
python mean_shift.py --image path/to/input/image.jpg --radius 30 -c 4 --features 3 --scale 0.3 --threshold 0.01 --save True
```
All the initial experiments and tests have been implemented in ```mean_shift_tests.ipynb```, if you want to have a look.

### Arguments

- `--image`: Path to the input image (required)
- `--radius`: Window radius for the Mean Shift algorithm (default: 30)
- `-c`: Constant value for speedup (default: 4)
- `--features`: Number of dimensions in the feature vector (3 or 5) (default: 3)
- `--scale`: Scale factor for resizing the image (default: 0.3)
- `--threshold`: Threshold for merging peaks (default: 0.01)
- `--save`: Whether to save the segmented image (default: False)

## Output

The script will display the segmented image based on the Mean Shift algorithm. If the `--save` argument is set to `True`, the segmented image will also be saved to disk. If it is set to `False`, then it will show the images to you. I have run these experiments through the terminal where I can then press a button to close the image window.

_Note: Please create an `output/` folder for saving the output image. The filename will be created automatically._

Feel free to adjust the parameters to experiment with different segmentation results. Enjoy!