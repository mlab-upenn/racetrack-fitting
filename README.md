# racetrack-fitting
Finds a feasible centerline, walls, and a signed-distance field from SLAM output

![image](https://user-images.githubusercontent.com/8052622/62798513-6699af80-baac-11e9-97cb-69573af57a86.png)

## Prerequisites
Get track image and info files from [Cartographer](https://github.com/googlecartographer) or some other SLAM tool. The image file should be in pgm format and the info file should be in yaml format. Here is an example track pgm and yaml combination:
##### track.pgm
![image](https://user-images.githubusercontent.com/8052622/62554222-bd01b680-b83e-11e9-8084-12e16ce1e749.png)
##### track.yaml
```
image: track.pgm
resolution: 0.050000
origin: [-5.659398, -4.766974, 0.000000]
negate: 0
occupied_thresh: 0.65
free_thresh: 0.196
```
## Usage
Clone the repository and run the center-line-fitting script:
```
git clone https://github.com/mlab-upenn/center-line-fitting.git
cd center-line-fitting
python src/fit_centerline.py
```

### Plots
To control the plots that are displayed, use
```
python center-line-fitting.py --plot_mode <0, 1, or 2>
```
0 shows no plots, 1 (default) shows basic plots, and 2 shows all plots

### I/O
You can specify the input file paths and output directory using the command line:
```
python center-line-fitting.py --pgm_path <path to track image> --yaml_path <path to track info> --out_dir <path to output directory>
```

### Subsampling Period
You can change the subsampling period as follows:
```
python center-line-fitting.py --subsample_period 20
```
This changes how sparesly the points are sampled from the center-line path
##### subsample_period = 6
![Subsampled6](https://user-images.githubusercontent.com/8052622/62798875-5930f500-baad-11e9-9c9b-fa2d5834daed.png)
##### subsample_period = 20
![Subsampled20](https://user-images.githubusercontent.com/8052622/62798874-58985e80-baad-11e9-9152-5e0a14f99eb3.png)

## Examples
![Singh](https://user-images.githubusercontent.com/8052622/62799073-d6f50080-baad-11e9-9d68-e094af89ac89.png)
