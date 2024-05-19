<h1 align="center">
	IconMatch
</h1>

<p align="center">
	<i>Easily select icons on the screen in any environment.</i>
</p>

<p align="center">
  <a href="https://luiszugasti.me">
    <img src="https://raw.githubusercontent.com/luiszugasti/IconMatch/main/images/screenshot.png" alt="Showcasing bounding boxes and original image"/>
  </a>
  <a href="https://luiszugasti.me">
    <img src="https://raw.githubusercontent.com/luiszugasti/IconMatch/main/images/nearest_box.gif" alt="Showcasing candidate boxes functionality"/>
  </a>
  <a href="https://luiszugasti.me">
    <img src="https://github.com/NativeSensors/IconMatch/assets/40773550/ebc5aa2e-50b3-464a-a033-7c54b7615eeb" alt="Showcasing realtime demo"/>
  </a>
</p>

  
This is part of the Hands Free Computing project. Built with [OpenCV 3.12](https://opencv.org) and [Python 3.8](https://python.org).

## Table of Contents


  - [Installation](#installation)
  - [Usage](#usage)
  - [API](#api)
  - [Roadmap](#roadmap)
  - [Contributing](#contributing)
  - [License](#license)
  - [Contact](#contact)


## Installation 

1. Clone the repo and open it locally:
```
$ git clone https://github.com/luiszugasti/IconMatch/
$ cd IconMatch
```

2. Install the [requirements](https://github.com/luiszugasti/IconMatch/blob/main/requirements.txt):
```
$ pip install -r requirements.txt
```

## Usage

You can use the functions as shown in [demo.py](https://github.com/luiszugasti/IconMatch/blob/main/icondetection/demo/demo.py) as a default entry point.

In the below example, the main set of functions is called within a callback function, as this allows the threshold value
to be controlled from a GUI in OpenCV.

Image Scanner:

```python
import cv2 as cv

import IconMatch.IconMatch from ImageScanner

src = cv.imread("source to your image file")
scanner = ImageScanner(thersh = 100)

detected_rectangles = scanner.scan(src)
# list of [(x,y,w,h),(x,y,w,h), ... , (x,y,w,h)]

```

Screen Scanner:

```python
import cv2 as cv

import IconMatch.IconMatch from ScreenScanner

src = cv.imread("source to your image file")
scanner = ScreenScanner(thersh = 100)

detected_rectangles = scanner.scan(bbox = (x,y,w,h))
# list of [(x,y,w,h),(x,y,w,h), ... , (x,y,w,h)]

```

RealTime demo:
```bash
python rt_demo.py
```

## Key Features

- Detection of areas with a high likelihood of being clickable icons.
- Detection of closest rectangle to point of interest (be it gaze, or mouse as in the examples)

## API

The current available APIs encompass what your image processing pipeline should contain. Both APIs are 
currently still experimental as I learn more about OpenCV and optimize code.

### ImageScanner
> Performs Canny detection on passed images and group overlapping rectangles 

### ScreenScanner
> Scans your display, take screnshoots and call ImageScanners

## Roadmap

- [x] Detect regions of interest with moderate accuracy
- [x] Detect candidate region based on proximity
- [x] Detect icon-like objects on the screen
- [?] Context provision into regions of interest


## Contributing

Contributions are what make the open source community such an amazing place to be learn, inspire, and create. Any contributions you make are **genuinely appreciated**.

1. Fork the Project
2. Create your Feature Branch (`git checkout -b feature/AmazingFeature`)
3. Commit your Changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the Branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## License

Distributed under the MIT License. See `LICENSE` for more information.

## Contact

Original Creator: Luis Zugasti - [@luis\_\_zugasti](https://twitter.com/luis__zugasti)

Current Maintainer: Piotr Walas - [@Piotr\_\_Walas](https://twitter.com/PW4ltz)

Project Link: [https://github.com/NativeSensors/IconMatch](https://github.com/NativeSensors/IconMatch)
