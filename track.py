import os
import sys
import argparse
import platform
import shutil
import time
from pathlib import Path
import cv2
import torch
import torch.backends.cudnn as cudnn
import requests  
import threading
sys.path.insert(0, './yolov5')

from yolov5.models.experimental import attempt_load
from yolov5.utils.datasets import LoadImages, LoadStreams
from yolov5.utils.general import (LOGGER, check_img_size, non_max_suppression, scale_coords, 
                                  check_imshow, xyxy2xywh, increment_path)
from yolov5.utils.torch_utils import select_device, time_sync
from yolov5.utils.plots import Annotator, colors
from deep_sort.utils.parser import get_config
from deep_sort.deep_sort import DeepSort
from yolov5.models.common import DetectMultiBackend

FILE = Path(__file__).resolve()
ROOT = FILE.parents[0]  # yolov5 deepsort root directory
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  # add ROOT to PATH
ROOT = Path(os.path.relpath(ROOT, Path.cwd()))  # relative

SERVER_URL = 'http://127.0.0.1:5000/person'  

# 전역 변수 초기화
count = 0
passed_objects = {}  # 선을 넘은 객체를 기록하는 사전
areaName = '안양대학교 아리관 602호'
def send_to_server_async():
    global areaName
    def send_request():
        try:
            requests.post(SERVER_URL, json={"areaName": areaName})
            print(f"{areaName} 지점의 카운트를 서버에 성공적으로 전송했습니다.")
        except Exception as e:
            print(f"서버와 통신 중 오류가 발생했습니다: {e}")
    
    # 새로운 스레드 생성 후 시작
    thread = threading.Thread(target=send_request)
    thread.start()

def count_obj(box, w, h, obj_id, class_name):
    global count, passed_objects
    # 객체의 중앙 좌표 계산
    center_coordinates = (int(box[0] + (box[2] - box[0]) / 2), int(box[1] + (box[3] - box[1]) / 2))
    line_position = w - 300  # 화면 너비에서 200px 떨어진 위치에 선을 긋습니다.

    # 객체가 선을 넘었는지 확인하고, 이미 넘은 객체는 무시
    if center_coordinates[0] > line_position and obj_id not in passed_objects:
        count += 1
        passed_objects[obj_id] = True  # 객체가 선을 넘었음을 기록
        print("사람이 선을 넘었습니다.")        
        # 서버로 상품 이름을 전송
        send_to_server_async()


def detect(opt):
    out, source, yolo_model, deep_sort_model, show_vid, save_vid, save_txt, imgsz, evaluate, half, project, name, exist_ok = \
        opt.output, opt.source, opt.yolo_model, opt.deep_sort_model, opt.show_vid, opt.save_vid, \
        opt.save_txt, opt.imgsz, opt.evaluate, opt.half, opt.project, opt.name, opt.exist_ok
    webcam = source == '0' or source.startswith('rtsp') or source.startswith('http') or source.endswith('.txt')

    # initialize deepsort
    cfg = get_config()
    cfg.merge_from_file(opt.config_deepsort)
    deepsort = DeepSort(deep_sort_model,
                        max_dist=cfg.DEEPSORT.MAX_DIST,
                        max_iou_distance=cfg.DEEPSORT.MAX_IOU_DISTANCE,
                        max_age=cfg.DEEPSORT.MAX_AGE, n_init=cfg.DEEPSORT.N_INIT, nn_budget=cfg.DEEPSORT.NN_BUDGET,
                        use_cuda=True)

    # Initialize
    device = select_device(opt.device)
    half &= device.type != 'cpu'  # half precision only supported on CUDA

    # The MOT16 evaluation runs multiple inference streams in parallel, each one writing to
    # its own .txt file. Hence, in that case, the output folder is not restored
    if not evaluate:
        if os.path.exists(out):
            pass
            shutil.rmtree(out)  # delete output folder
        os.makedirs(out)  # make new output folder

    # Directories
    save_dir = increment_path(Path(project) / name, exist_ok=exist_ok)  # increment run
    save_dir.mkdir(parents=True, exist_ok=True)  # make dir

    # Load model
    device = select_device(device)
    model = DetectMultiBackend(yolo_model, device=device, dnn=opt.dnn)
    stride, names, pt, jit, _ = model.stride, model.names, model.pt, model.jit, model.onnx
    imgsz = check_img_size(imgsz, s=stride)  # check image size

    # Half
    half &= pt and device.type != 'cpu'  # half precision only supported by PyTorch on CUDA
    if pt:
        model.model.half() if half else model.model.float()

    # Set Dataloader
    vid_path, vid_writer = None, None
    if show_vid:
        show_vid = check_imshow()

    if webcam:
        show_vid = check_imshow()
        cudnn.benchmark = True  # set True to speed up constant image size inference
        dataset = LoadStreams(source, img_size=imgsz, stride=stride, auto=pt and not jit)
        bs = len(dataset)  # batch_size
    else:
        dataset = LoadImages(source, img_size=imgsz, stride=stride, auto=pt and not jit)
        bs = 1  # batch_size
    vid_path, vid_writer = [None] * bs, [None] * bs

    names = model.module.names if hasattr(model, 'module') else model.names
    if pt and device.type != 'cpu':
        model(torch.zeros(1, 3, *imgsz).to(device).type_as(next(model.model.parameters())))  # warmup

    dt, seen = [0.0, 0.0, 0.0, 0.0], 0
    for frame_idx, (path, img, im0s, vid_cap, s) in enumerate(dataset):
        t1 = time_sync()
        img = torch.from_numpy(img).to(device)
        img = img.half() if half else img.float()  # uint8 to fp16/32
        img /= 255.0  # 0 - 255 to 0.0 - 1.0
        if img.ndimension() == 3:
            img = img.unsqueeze(0)
        t2 = time_sync()
        dt[0] += t2 - t1

        # Inference
        visualize = increment_path(save_dir / Path(path).stem, mkdir=True) if opt.visualize else False
        pred = model(img, augment=opt.augment, visualize=visualize)
        t3 = time_sync()
        dt[1] += t3 - t2

        # Apply NMS
        pred = non_max_suppression(pred, opt.conf_thres, opt.iou_thres, opt.classes, opt.agnostic_nms, max_det=opt.max_det)
        dt[2] += time_sync() - t3

        # Process detections
        for i, det in enumerate(pred):
            seen += 1
            if webcam:
                p, im0, _ = path[i], im0s[i].copy(), dataset.count
                s += f'{i}: '
            else:
                p, im0, _ = path, im0s.copy(), getattr(dataset, 'frame', 0)

            p = Path(p)
            save_path = str(save_dir / p.name)
            s += '%gx%g ' % img.shape[2:]

            annotator = Annotator(im0, line_width=1, pil=not ascii)
            w, h = im0.shape[1], im0.shape[0]
            if det is not None and len(det):
                # Rescale boxes from img_size to im0 size
                det[:, :4] = scale_coords(img.shape[2:], det[:, :4], im0.shape).round()

                for c in det[:, -1].unique():
                    n = (det[:, -1] == c).sum()  # detections per class
                    s += f"{n} {names[int(c)]}{'s' * (n > 1)}, "

                xywhs = xyxy2xywh(det[:, 0:4])
                confs = det[:, 4]
                clss = det[:, 5]

                # Pass detections to deepsort
                outputs = deepsort.update(xywhs.cpu(), confs.cpu(), clss.cpu(), im0)

                if len(outputs) > 0:
                    for output in outputs:
                        bboxes = output[0:4]
                        id = output[4]
                        cls = output[5]
                        label = f'{id} {names[int(cls)]}'
                        annotator.box_label(bboxes, label, color=colors(int(cls), True))

                        # 객체가 선을 넘으면 서버로 상품 번호를 전송합니다.
                        count_obj(bboxes, w, h, id, names[int(cls)])

                LOGGER.info(f'{s}Done.')

            else:
                deepsort.increment_ages()

            # Stream results and draw the line
            im0 = annotator.result()

            # 화면에 선을 그립니다 (화면 오른쪽에서 200px 떨어진 위치)
            line_position = w - 300
            color = (0, 255, 0)  # 초록색 선
            thickness = 2
            cv2.line(im0, (line_position, 0), (line_position, h), color, thickness)

            #해상도 조절
            im0_resized = cv2.resize(im0,(640,640))
            if show_vid:
                cv2.imshow(str(p), im0_resized)
                if cv2.waitKey(1) == ord('q'):
                    raise StopIteration

    LOGGER.info(f"Speed: {t3 - t2:.1f}ms inference, {t2 - t1:.1f}ms NMS per image")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--yolo_model', nargs='+', type=str, default='yolov5m.pt', help='model.pt path(s)')
    parser.add_argument('--deep_sort_model', type=str, default='osnet_x0_25')
    parser.add_argument('--source', type=str, default='0', help='source')  # file/folder, 0 for webcam
    parser.add_argument('--output', type=str, default='inference/output', help='output folder')  # output folder
    parser.add_argument('--imgsz', '--img', '--img-size', nargs='+', type=int, default=[640], help='inference size h,w')
    parser.add_argument('--conf-thres', type=float, default=0.30, help='object confidence threshold')
    parser.add_argument('--iou-thres', type=float, default=0.50, help='IOU threshold for NMS')
    parser.add_argument('--fourcc', type=str, default='mp4v', help='output video codec (verify ffmpeg support)')
    parser.add_argument('--device', default='', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--show-vid', action='store_true', help='display tracking video results')
    parser.add_argument('--save-vid', action='store_true', help='save video tracking results')
    parser.add_argument('--save-txt', action='store_true', help='save MOT compliant results to *.txt')
    parser.add_argument('--classes', nargs='+', default= [0],type=int, help='filter by class: --class 0, or --class 16 17')
    parser.add_argument('--agnostic-nms', action='store_true', help='class-agnostic NMS')
    parser.add_argument('--augment', action='store_true', help='augmented inference')
    parser.add_argument('--evaluate', action='store_true', help='augmented inference')
    parser.add_argument("--config_deepsort", type=str, default="deep_sort/configs/deep_sort.yaml")
    parser.add_argument("--half", action="store_true", help="use FP16 half-precision inference")
    parser.add_argument('--visualize', action='store_true', help='visualize features')
    parser.add_argument('--max-det', type=int, default=1000, help='maximum detection per image')
    parser.add_argument('--dnn', action='store_true', help='use OpenCV DNN for ONNX inference')
    parser.add_argument('--project', default=ROOT / 'runs/track', help='save results to project/name')
    parser.add_argument('--name', default='exp', help='save results to project/name')
    parser.add_argument('--exist-ok', action='store_true', help='existing project/name ok, do not increment')
    opt = parser.parse_args()
    opt.imgsz *= 2 if len(opt.imgsz) == 1 else 1  # expand

    with torch.no_grad():
        detect(opt)
print('asd')