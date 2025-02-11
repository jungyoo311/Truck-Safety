## Overview
TPD system
2024-25 SDP Truck Safety TPD system
1. Forward Collision Warning
2. Lane Merge Warning
 
## Instructions to set-up the environment
## Installation
1. Environmental Configuration
   Ensure the environment is set up correctly by sourcing the provided script. This script is required and activates the Hailo virtual environment.
```mardown
source setup_env.sh
```
    As Version Confliction reported, use this cmd line to use certain versions.
```mardown
sudo apt install hailo-tappas-core=3.30.0-1 hailo-dkms=4.19.0-1 hailort=4.19.0-3
```

2. Requirment Installation
    Under the activated virtual environment, install the necessary Python requirement packages.
```markdown
pip install -r requirements.txt
```
3. Resource download
    download the required resoures by running
```markdown
./download_resources.sh
```
## Run the code
To run the code, you should

```markdown
python basic_pipeline/TPD_detection.py
```
## options for running cmd
1. --input rpi : for camera input
2. --input /path/to/file : input video/image
3. --show-fps : print fps on the screen
4. --hef : your hef model path

## example running code
```markdown
python basic_pipelines/TPD_detection.py --hef resources/yolov10s.hef  --input resources/test01.mp4 --show-fps
```
