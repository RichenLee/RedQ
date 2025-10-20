# RedQ (RedchanelQuantification) - Fluorescence Quantification Software
## RedQ：a YOLOv5-based software for tube fluorescence signals quantification.
### Overview
RedQ (Red Channel Quantification) is a specialized software tool designed for extracting quantitative red channel intensity values from fluorescence tube images. Leveraging computer vision and deep learning technologies, this software enables precise quantification of fluorescence signals for research applications.
<img width="1594" height="724" alt="image" src="https://github.com/user-attachments/assets/9c144a64-f26d-4b34-960c-e6387686cc80" />

### Environment Setup
#### Create Conda Environment
```
conda create -n redq_env python=3.8.5
conda activate redq_env
```

#### Install Dependencies
```
# Install PyTorch packages
conda install pytorch torchvision torchaudio

# Install additional dependencies
conda install reportlab
pip install -r requirements.txt
pip install pymupdf
```
#### Font Configuration
```
# Copy font files to ReportLab fonts directory
# Replace {anaconda_env_path} with your actual Anaconda environment path
cp arial.ttf simsun.ttc {anaconda_env_path}/lib/python3.8/site-packages/reportlab/fonts/
```

### Usage
#### Basic Command
```
python detect.py \
  --source path/to/your/pictures \
  --weights ./trained_model/weights/best.pt \
  --save-txt \
  --name [output_dir] \
  --device cpu \
  --hide-conf
```

### Output Results
Upon completion, the software generates:
+ Annotated detection images
+ Detection data in text format
+ Comprehensive analysis report: `Report.pdf`

### Train
```
# Data for trainning were restored in data_for_train
python train.py \
--data tube_data.yaml \
--cfg tube_yolov5s.yaml \
--weights yolov5s.pt \
--batch-size 4 \
--epoch 100 \
--device 0
```
