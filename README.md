# Half-Circle-Detection [![forthebadge made-with-python](http://ForTheBadge.com/images/badges/made-with-python.svg)](https://www.python.org/)
## Introduction
This work done by me to path a drone in parkour system.Parkour system contain red half-circle pasteboards and traffic signs(like turn righ-left).The script is like:
```
1. Drone look for objects
2. Drone detect both objects(traffic sign and half circle)
3. Drone compares which object is closer
4. If object is circle, Drone pass throguht circle
5. If object is traffic sign, Drone execute command(like turn right and look for new object)
```
## Dependicies
```
- Opencv
- Numpy
- Scikit-Image
- Sklearn
```
## Examples
- `webcam_h_c_detector.py :` This script find red colored half-circle object.You can tune the contour mask with "Trackbars" to find the other colored circles.
- `wbcm_trafficsign_hc.py :` This script both can find red-circle and traffic signs.In first situation it detects just red half circle.There is a flag inside.If you change flag "0" to "1" it detect traffic signs.
- `webcam_circle_cnn.py   :` This script detect both traffic sign and half circle in the same frame and find which one is closer.You have to download [this](https://github.com/ZhouJiaHuan/traffic-sign-detection/blob/master/svm_hog_classification/svm_model.pkl) file and put it to the "svm_hog_classification" folder to run script.
-This section will be updated soon!!!

## Credit
Thanks to him, I used [ZhouJiaHuan's](https://github.com/ZhouJiaHuan/traffic-sign-detection) implemention of traffic sign detection.
