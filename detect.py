# YOLOv5 🚀 by Ultralytics, GPL-3.0 license
"""
Run inference on images, videos, directories, streams, etc.

Usage:
    $ python path/to/detect.py --weights yolov5s.pt --source 0  # webcam
                                                             img.jpg  # image
                                                             vid.mp4  # video
                                                             path/  # directory
                                                             path/*.jpg  # glob
                                                             'https://youtu.be/Zgi9g1ksQHc'  # YouTube
                                                             'rtsp://example.com/media.mp4'  # RTSP, RTMP, HTTP stream
"""

import argparse
import os
import sys
from pathlib import Path
from report import *
import cv2
import torch
import torch.backends.cudnn as cudnn
import numpy as np
import math
import fitz
from PIL import Image as PI
import shutil
import json
FILE = Path(__file__).resolve()
ROOT = FILE.parents[0]  # YOLOv5 root directory
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  # add ROOT to PATH
ROOT = Path(os.path.relpath(ROOT, Path.cwd()))  # relative

from models.common import DetectMultiBackend
from utils.datasets import IMG_FORMATS, VID_FORMATS, LoadImages, LoadStreams
from utils.general import (LOGGER, check_file, check_img_size, check_imshow, check_requirements, colorstr,
                           increment_path, non_max_suppression, print_args, scale_coords, strip_optimizer, xyxy2xywh)
from utils.plots import Annotator, colors, save_one_box
from utils.torch_utils import select_device, time_sync
import matplotlib.pyplot as plt
from reportlab.graphics.shapes import Image as DrawingImage

@torch.no_grad()

def mkdir(path):
    folder = os.path.exists(path)
    if not folder:                   
        os.makedirs(path)            
    else:
        pass
        
def cv_show(name,image):
    cv2.imshow(name,image)
    cv2.waitKey(0)
    

def Trans_label(x,y,w,h,W,H):
    x=x*W
    y=y*H
    w=w*W
    h=h*H
    x_range=[round(x-0.5*w),round(x+0.5*w)]
    y_range=[round(y-0.5*h),round(y+0.5*h)]
    return x_range,y_range

def Get_core(x,y):
    x=round(x/2)
    y=round(y/2)
    x_range=[x-5,x+5]
    y_range=[y-5,y+5]
    return x_range,y_range

def Get_Red(img,out,exp,task_name,yolo_img):
    image = cv2.imread(img)
    size = image.shape
    w = size[1] #宽度
    h = size[0] #高度
    x_range,y_range=Trans_label(exp[0],exp[1],exp[2],exp[3],w,h)
    project='_'.join(out.split('_')[:-1])
    area=image[y_range[0]:y_range[1],x_range[0]:x_range[1]]
    draw_number(yolo_img,x_range,y_range,out.split('_')[-1])
    (B,G,R) = cv2.split(area)
    
    mkdir(os.path.join(task_name,out))
    cv2.imwrite('./%s/%s/%s.jpg'%(task_name,out,out),area)

    R = area.copy()
    R[:,:,0] = 0
    R[:,:,1] = 0
    cv2.imwrite('./%s/%s/%s_Red.jpg'%(task_name,out,out),R)
    G = area.copy()
    G[:,:,0] = 0
    G[:,:,2] = 0
    cv2.imwrite('./%s/%s/%s_Green.jpg'%(task_name,out,out),G)
    B = area.copy()
    B[:,:,1] = 0
    B[:,:,2] = 0
    cv2.imwrite('./%s/%s/%s_Blue.jpg'%(task_name,out,out),B)

    core_x_range,core_y_range=Get_core(len(R),len(R[0]))
    core=R[core_x_range[0]:core_x_range[1],core_y_range[0]:core_y_range[1]]
    
    result=[]
    for i in core:
        for j in i:
            v=j[2]
            result.append(v)
    cv2.imwrite('./%s/%s/%s_core.jpg'%(task_name,out,out),core)
    return result

def sort_key(elem):
    return elem[1][0]

def draw_number(yolo_img,x_range,y_range,n):
    x=int(x_range[0])
    y=int(y_range[1])+200
    #print(x,y)
    cv2.putText(yolo_img,n, (x,y), cv2.FONT_HERSHEY_SIMPLEX, 6, (0, 0, 255), 6)

def work(file,label_file,task_name,yolo_file):
    flag=True
    img=file
    prefix='.'.join(file.split('.')[:-1])
    task=[]
    if os.path.exists(label_file):
        ff=open(label_file)
        f=ff.readlines()
        ff.close()
        f=[i.strip().split()[0] for i in f]
        if '0' not in f:
            flag=False
    else:
        flag=False

    if flag:
        with open(label_file) as ff:
            n=0
            for line in ff:
                sp=line.strip().split()
                if sp[0]=='0':
                    n+=1
                    cood=[float(i) for i in sp[1:]]
                    out='tube'+'_'+str(n)
                    task.append([out,cood])

        task.sort(key=sort_key)
        sorted_task=[]
        n=0
        for tmp in task:
            n+=1
            sorted_task.append([tmp[0].split('_')[0]+'_'+str(n),tmp[1]])
        project='_'.join(out.split('_')[:-1])
        mkdir(task_name)
        report=open('./%s/Report.txt'%(task_name),'w')
        result_list=[('编号', '荧光强度','红光通道')]
        x=[]
        y=[]
        img_list=[]
        yolo_img=cv2.imread(yolo_file)
        for arg in sorted_task:
            result=Get_Red(img,arg[0],arg[1],task_name,yolo_img)
            img_list.append('./%s/%s/%s_Red.jpg'%(task_name,arg[0],arg[0]))
            #print('./%s/%s/%s_Red.jpg'%(task_name,arg[0],arg[0]))
            img_ = Image('./%s/%s/%s_Red.jpg'%(task_name,arg[0],arg[0]))
            img_.drawWidth = 30
            img_.drawHeight = 30
            ave=sum(result)/len(result)
            report.write(arg[0]+'\t'+str(ave)+'\n')
            result_list.append((arg[0].split('_')[1],str(ave),img_))
            x.append((arg[0].split('_')[1]))
            y.append(ave)
        report.close()
        cv2.imwrite('./%s/number.jpg'%(task_name),yolo_img)
        img_name=file.split('/')[-1]
        fig, ax = plt.subplots()
        ax.spines['right'].set_visible(False)
        ax.spines['top'].set_visible(False)
        p1=plt.bar(x, y)
        plt.bar_label(p1, label_type='edge')
        plt.savefig('./%s/Report.png'%(task_name))
        # 创建内容对应的空列表
        content = list()
        # 添加标题
        content.append(Graphs.draw_title('CRISPR核酸检测报告'))

        # 添加图片
        content.append(Graphs.draw_little_title(f'项目名称：{task_name}'))
        content.append(Spacer(1, 10))
        content.append(Graphs.draw_little_title(f'上传图片：{img_name}'))
        content.append(Spacer(1, 10))
        time_=str(datetime.datetime.now()).split('.')[0]
        content.append(Graphs.draw_little_title(f'检测时间：{time_}'))
        content.append(Spacer(1, 10))
        content.append(Graphs.draw_little_title(f'1.照片与样品编号：'))
        content.append(Spacer(1, 10))
        content.append(Graphs.draw_img('./%s/number.jpg'%(task_name)))
        content.append(Spacer(1, 10))


        # 添加段落文字
        # 添加小标题
        content.append(Graphs.draw_little_title('2.荧光强度分析'))
        content.append(Spacer(1, 10))
        # 添加表格
        content.append(Graphs.draw_table(*result_list))
        content.append(Spacer(1, 10))
        img = Image('./%s/Report.png'%(task_name))       # 读取指定路径下的图片
        img.drawWidth = 16*cm        # 设置图片的宽度
        img.drawHeight = 12*cm       # 设置图片的高度
        content.append(img)
        content.append(Spacer(1, 10))

        #img = Image() 
        #content.append()
        # 生成图表

        # 生成pdf文件
        doc = SimpleDocTemplate('./%s/Report.pdf'%(task_name), pagesize=letter)
        doc.build(content)
    else:
        mkdir(task_name)
        img_name=file.split('/')[-1]
        content = list()
        # 添加标题
        content.append(Graphs.draw_title('CRISPR核酸检测报告'))

        # 添加图片
        content.append(Graphs.draw_little_title(f'项目名称：{task_name}'))
        content.append(Spacer(1, 10))
        content.append(Graphs.draw_little_title(f'上传图片：{img_name}'))
        content.append(Spacer(1, 10))
        time_=str(datetime.datetime.now()).split('.')[0]
        content.append(Graphs.draw_little_title(f'检测时间：{time_}'))
        content.append(Spacer(1, 10))
        content.append(Graphs.draw_little_title(f'未检测到有效目标，请检查是否按要求使用软件！'))
        doc = SimpleDocTemplate('./%s/Report.pdf'%(task_name), pagesize=letter)
        doc.build(content)    

def pdf2png(report,name):
    mkdir('./pdf2png/')
    pdf=report
    doc = fitz.open(pdf)
    for pg in range(doc.page_count):
        page = doc[pg]
        rotate = int(0)
        # 每个尺寸的缩放系数为2，这将为我们生成分辨率提高四倍的图像。
        zoom_x = 2.0
        zoom_y = 2.0
        trans = fitz.Matrix(zoom_x, zoom_y).prerotate(rotate)
        pm = page.get_pixmap(matrix=trans, alpha=False)
        pm.pil_save('./pdf2png/%s.png' % pg)
    lst=[]
    for pg in range(doc.page_count):
        img = PI.open('./pdf2png/%s.png' % pg) # 打开图片
        img = np.array(img) # 转化为ndarray对象
        lst.append(img)
    img = np.concatenate(lst, axis = 1)   
    img = PI.fromarray(img)
    img.save('%s_result.png'%name)

def run(weights=ROOT / 'yolov5s.pt',  # model.pt path(s)
        source_path=ROOT / 'data/images',  # file/dir/URL/glob, 0 for webcam
        imgsz=640,  # inference size (pixels)
        conf_thres=0.25,  # confidence threshold
        iou_thres=0.45,  # NMS IOU threshold
        max_det=1000,  # maximum detections per image
        device='',  # cuda device, i.e. 0 or 0,1,2,3 or cpu
        view_img=False,  # show results
        save_txt=False,  # save results to *.txt
        save_conf=False,  # save confidences in --save-txt labels
        save_crop=False,  # save cropped prediction boxes
        nosave=False,  # do not save images/videos
        classes=None,  # filter by class: --class 0, or --class 0 2 3
        agnostic_nms=False,  # class-agnostic NMS
        augment=False,  # augmented inference
        visualize=False,  # visualize features
        update=False,  # update all models
        project=ROOT / 'runs/detect',  # save results to project/name
        name='RAVI-CRISPR-Detection',  # save results to project/name
        exist_ok=False,  # existing project/name ok, do not increment
        line_thickness=3,  # bounding box thickness (pixels)
        hide_labels=False,  # hide labels
        hide_conf=False,  # hide confidences
        half=False,  # use FP16 half-precision inference
        dnn=False,  # use OpenCV DNN for ONNX inference
        ):

    source=source_path
    save_img = not nosave and not source.endswith('.txt')  # save inference images
    is_file = Path(source).suffix[1:] in (IMG_FORMATS + VID_FORMATS)
    is_url = source.lower().startswith(('rtsp://', 'rtmp://', 'http://', 'https://'))
    webcam = source.isnumeric() or source.endswith('.txt') or (is_url and not is_file)
    if is_url and is_file:
        source = check_file(source)  # download

    # Directories
    save_dir = increment_path(Path(project) / name, exist_ok=exist_ok)  # increment run
    (save_dir / 'labels' if save_txt else save_dir).mkdir(parents=True, exist_ok=True)  # make dir

    # Load model
    device = select_device(device)
    model = DetectMultiBackend(weights, device=device, dnn=dnn)
    stride, names, pt, jit, onnx = model.stride, model.names, model.pt, model.jit, model.onnx
    imgsz = check_img_size(imgsz, s=stride)  # check image size

    # Half
    half &= pt and device.type != 'cpu'  # half precision only supported by PyTorch on CUDA
    if pt:
        model.model.half() if half else model.model.float()

    # Dataloader
    if webcam:
        view_img = check_imshow()
        cudnn.benchmark = True  # set True to speed up constant image size inference
        dataset = LoadStreams(source, img_size=imgsz, stride=stride, auto=pt and not jit)
        bs = len(dataset)  # batch_size
    else:
        dataset = LoadImages(source, img_size=imgsz, stride=stride, auto=pt and not jit)
        bs = 1  # batch_size
    vid_path, vid_writer = [None] * bs, [None] * bs

    # Run inference
    if pt and device.type != 'cpu':
        model(torch.zeros(1, 3, *imgsz).to(device).type_as(next(model.model.parameters())))  # warmup
    dt, seen = [0.0, 0.0, 0.0], 0
    for path, im, im0s, vid_cap, s in dataset:
        t1 = time_sync()
        im = torch.from_numpy(im).to(device)
        im = im.half() if half else im.float()  # uint8 to fp16/32
        im /= 255  # 0 - 255 to 0.0 - 1.0
        if len(im.shape) == 3:
            im = im[None]  # expand for batch dim
        t2 = time_sync()
        dt[0] += t2 - t1

        # Inference
        visualize = increment_path(save_dir / Path(path).stem, mkdir=True) if visualize else False
        pred = model(im, augment=augment, visualize=visualize)
        t3 = time_sync()
        dt[1] += t3 - t2

        # NMS
        pred = non_max_suppression(pred, conf_thres, iou_thres, classes, agnostic_nms, max_det=max_det)
        dt[2] += time_sync() - t3

        # Second-stage classifier (optional)
        # pred = utils.general.apply_classifier(pred, classifier_model, im, im0s)

        # Process predictions
        for i, det in enumerate(pred):  # per image
            seen += 1
            if webcam:  # batch_size >= 1
                p, im0, frame = path[i], im0s[i].copy(), dataset.count
                s += f'{i}: '
            else:
                p, im0, frame = path, im0s.copy(), getattr(dataset, 'frame', 0)

            p = Path(p)  # to Path
            save_path = str(save_dir / p.name)  # im.jpg
            txt_path = str(save_dir / 'labels' / p.stem) + ('' if dataset.mode == 'image' else f'_{frame}')  # im.txt
            s += '%gx%g ' % im.shape[2:]  # print string
            gn = torch.tensor(im0.shape)[[1, 0, 1, 0]]  # normalization gain whwh
            imc = im0.copy() if save_crop else im0  # for save_crop
            annotator = Annotator(im0, line_width=line_thickness, example=str(names))
            if len(det):
                # Rescale boxes from img_size to im0 size
                det[:, :4] = scale_coords(im.shape[2:], det[:, :4], im0.shape).round()

                # Print results
                for c in det[:, -1].unique():
                    n = (det[:, -1] == c).sum()  # detections per class
                    s += f"{n} {names[int(c)]}{'s' * (n > 1)}, "  # add to string

                # Write results
                for *xyxy, conf, cls in reversed(det):
                    if save_txt:  # Write to file
                        xywh = (xyxy2xywh(torch.tensor(xyxy).view(1, 4)) / gn).view(-1).tolist()  # normalized xywh
                        line = (cls, *xywh, conf) if save_conf else (cls, *xywh)  # label format
                        with open(txt_path + '.txt', 'a') as f:
                            f.write(('%g ' * len(line)).rstrip() % line + '\n')

                    if save_img or save_crop or view_img:  # Add bbox to image
                        c = int(cls)  # integer class
                        label = None if hide_labels else (names[c] if hide_conf else f'{names[c]} {conf:.2f}')
                        annotator.box_label(xyxy, label, color=colors(c, True))
                        if save_crop:
                            save_one_box(xyxy, imc, file=save_dir / 'crops' / names[c] / f'{p.stem}.jpg', BGR=True)

            # Print time (inference-only)
            LOGGER.info(f'{s}Done. ({t3 - t2:.3f}s)')

            # Stream results
            im0 = annotator.result()
            if view_img:
                cv2.imshow(str(p), im0)
                cv2.waitKey(1)  # 1 millisecond

            # Save results (image with detections)
            if save_img:
                if dataset.mode == 'image':
                    #print(source,save_path,'./%s/Report.pdf'%(name),source.split('/')[-1].split('.')[0],source.split('/')[-1].split('.')[0]+'_result.png',name)
                    cv2.imwrite(save_path, im0)
                    work(source,txt_path+'.txt',name,save_path)
                    pdf2png('./%s/Report.pdf'%(name),source.split('/')[-1].split('.')[0])
                    result_name=source.split('/')[-1].split('.')[0]+'_result.png'
                    #shutil.rmtree('./%s/'%name)                        
                    if os.path.exists(os.path.join(project,result_name)):
                        os.remove(os.path.join(project,result_name))
                    shutil.move(result_name,name)
                else:  # 'video' or 'stream'
                    if vid_path[i] != save_path:  # new video
                        vid_path[i] = save_path
                        if isinstance(vid_writer[i], cv2.VideoWriter):
                            vid_writer[i].release()  # release previous video writer
                        if vid_cap:  # video
                            fps = vid_cap.get(cv2.CAP_PROP_FPS)
                            w = int(vid_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                            h = int(vid_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                        else:  # stream
                            fps, w, h = 30, im0.shape[1], im0.shape[0]
                            save_path += '.mp4'
                        vid_writer[i] = cv2.VideoWriter(save_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (w, h))
                    vid_writer[i].write(im0)

    # Print results
    t = tuple(x / seen * 1E3 for x in dt)  # speeds per image
    LOGGER.info(f'Speed: %.1fms pre-process, %.1fms inference, %.1fms NMS per image at shape {(1, 3, *imgsz)}' % t)
    if save_txt or save_img:
        s = f"\n{len(list(save_dir.glob('labels/*.txt')))} labels saved to {save_dir / 'labels'}" if save_txt else ''
        LOGGER.info(f"Results saved to {colorstr('bold', save_dir)}{s}")
    if update:
        strip_optimizer(weights)  # update model (to fix SourceChangeWarning)
    shutil.rmtree(save_dir)
    shutil.rmtree('./pdf2png/')
    shutil.rmtree('./runs/')

def parse_opt():
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', nargs='+', type=str, default=ROOT / 'runs/train/exp7/weights/best.pt', help='model path(s)')
    parser.add_argument('--source_path', type=str, default=ROOT / 'data/images', help='file/dir/URL/glob, 0 for webcam')
    parser.add_argument('--imgsz', '--img', '--img-size', nargs='+', type=int, default=[640], help='inference size h,w')
    parser.add_argument('--conf-thres', type=float, default=0.5, help='confidence threshold')
    parser.add_argument('--iou-thres', type=float, default=0.45, help='NMS IoU threshold')
    parser.add_argument('--max-det', type=int, default=1000, help='maximum detections per image')
    parser.add_argument('--device', default='cpu', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--view-img', action='store_true', help='show results')
    parser.add_argument('--save-txt', action='store_true', default=True,help='save results to *.txt')
    parser.add_argument('--save-conf', action='store_true', help='save confidences in --save-txt labels')
    parser.add_argument('--save-crop', action='store_true', help='save cropped prediction boxes')
    parser.add_argument('--nosave', action='store_true', help='do not save images/videos')
    parser.add_argument('--classes', nargs='+', type=int, help='filter by class: --classes 0, or --classes 0 2 3')
    parser.add_argument('--agnostic-nms', action='store_true', help='class-agnostic NMS')
    parser.add_argument('--augment', action='store_true', help='augmented inference')
    parser.add_argument('--visualize', action='store_true', help='visualize features')
    parser.add_argument('--update', action='store_true', help='update all models')
    parser.add_argument('--project', default=ROOT / 'runs/detect', help='save results to project/name')
    parser.add_argument('--name', default='exp', help='save results to project/name')
    parser.add_argument('--exist-ok', action='store_true', help='existing project/name ok, do not increment')
    parser.add_argument('--line-thickness', default=3, type=int, help='bounding box thickness (pixels)')
    parser.add_argument('--hide-labels', default=False, action='store_true', help='hide labels')
    parser.add_argument('--hide-conf', default=True, action='store_true', help='hide confidences')
    parser.add_argument('--half', action='store_true', help='use FP16 half-precision inference')
    parser.add_argument('--dnn', action='store_true', help='use OpenCV DNN for ONNX inference')
    opt = parser.parse_args()
    opt.imgsz *= 2 if len(opt.imgsz) == 1 else 1  # expand
    print_args(FILE.stem, opt)
    return opt


def main(opt):
    check_requirements(exclude=('tensorboard', 'thop'))
    run(**vars(opt))


# 命令使用
# python detect.py --weights runs/train/exp_yolov5s/weights/best.pt --source  data/images/fishman.jpg # webcam
if __name__ == "__main__":
    opt = parse_opt()
    main(opt)
