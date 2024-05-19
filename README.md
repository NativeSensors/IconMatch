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


## Key Features

- Detection of areas with a high likelihood of being clickable icons.
- Detection of closest rectangle to point of interest (be it gaze, or mouse as in the examples)

## API

The current available APIs encompass what your image processing pipeline should contain. Both APIs are 
currently still experimental as I learn more about OpenCV and optimize code.

### canny_detection(gray_scale_image, min_threshold)
> Performs canny detection when given a gray scale image and a minimum threshold for hysteresis. Returns bounding rectangles of points of interest.

### group_rects(bound_rectangles, initial_scanning_range, final_scanning_range)
> Groups rectangles that are overlapping in two-dimensional space and returns their conglomerate components.

## Roadmap

- [x] Detect regions of interest with moderate accuracy
- [x] Detect candidate region based on proximity
- [ ] Detect icon-like objects on the screen
- [ ] Context provision into regions of interest


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

Luis Zugasti - [@luis\_\_zugasti](https://twitter.com/luis__zugasti)

Project Link: [https://github.com/luiszugasti/IconMatch](https://github.com/luiszugasti/IconMatch)

[contributors-shield]: https://img.shields.io/github/contributors/luiszugasti/IconMatch.svg?style=flat-square
[contributors-url]: https://github.com/luiszugasti/IconMatch/graphs/contributors
[forks-shield]: https://img.shields.io/github/forks/luiszugasti/IconMatch.svg?style=flat-square
[forks-url]: https://github.com/luiszugasti/IconMatch/network/members
[stars-shield]: https://img.shields.io/github/stars/luiszugasti/IconMatch.svg?style=flat-square
[stars-url]: https://github.com/luiszugasti/IconMatch/stargazers
[issues-shield]: https://img.shields.io/github/issues/luiszugasti/IconMatch.svg?style=flat-square
[issues-url]: https://github.com/luiszugasti/IconMatch/issues
[license-shield]: https://img.shields.io/github/license/luiszugasti/IconMatch.svg?style=flat-square
[license-url]: https://github.com/luiszugasti/IconMatch/blob/main/LICENSE.txt
[linkedin-shield]: https://img.shields.io/badge/-LinkedIn-black.svg?style=flat-square&logo=linkedin&colorB=555
[linkedin-url]: https://linkedin.com/in/luiszugasti
[product-screenshot1]: https://i.imgur.com/Q4Rm7M6.png
[product-screenshot2]: https://i.imgur.com/8NZGOa7.gif
