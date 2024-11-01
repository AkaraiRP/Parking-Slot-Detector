# Parking-Slot-Detector
## Dependencies
This application requires Python 3.12 and the following dependencies to run:
```python
opencv-python
scikit-learn
pandas
Pillow
scikit-image
```

## Installation
Install Dependencies with
```cmd
pip install -r requirements.txt
```

## Running the Application
Run the Parking Detector application with the follow command:
```cmd
python main.py video=<VideoCapture> mask=[Optional: MaskFile]
```
    
## Documentation
### File arguments
| Argument | Alias      | Description | Example |
|----------|-------     |-------------|---------|
|video=    | `-v, --video`| The video capture stream to use. Either from a file or direct feed from a camera. See [using capture device for VideoCapture.](https://docs.opencv.org/4.x/dd/d43/tutorial_py_video_display.html) | `-v data/parking_sample_loop.mp4, -v 0` |
|mask=     | `-m, --mask` | [OPTIONAL!] The mask file to use. If no mask is supplied, the user should define their own parking slots using mouse clicks. See [Manual Masking](#manual-masking) for more information. | `-m data/parking_mask.mp4` |

### Manual Masking
You can manually define parking spots if a mask is not supplied or you need to add a new spot. To do this, simply left click on the 4 corners of the parking spot you want to define and press `Enter`.
![Manual Masking Example](https://media0.giphy.com/media/v1.Y2lkPTc5MGI3NjExbXB3bTRnN2syYW14eTJkYjFtZWM3Ymp2bGUyZ2h5azZiNGM1YnVuNiZlcD12MV9pbnRlcm5hbF9naWZfYnlfaWQmY3Q9Zw/95eUbbkp9ejCFt1F9G/giphy.webp)

Additionally, if you want to remove a masked slot, simply press `Backspace` to delete the most recent slot. If using a mask file, the most recent slot may not be what you expect as it uses OpenCV's component mask to get the parking spots.

To save the current parking spots, press `S` to save to the `mask` folder in the main file's directory.



## Authors

- [Emil Christoffer Briones](https://github.com/AkaraiRP)
- [Bryan Lewis Cabarroguis](https://github.com/bryyc)
- [Andrea Nixie Manansala](https://github.com/dreanaa)
- [Sage Vilaga](https://github.com/ManThisSucks)

