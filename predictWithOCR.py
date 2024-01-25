# Ultralytics YOLO 🚀, GPL-3.0 license

import hydra
import torch
import easyocr
import cv2
from ultralytics.yolo.engine.predictor import BasePredictor
from ultralytics.yolo.utils import DEFAULT_CONFIG, ROOT, ops
from ultralytics.yolo.utils.checks import check_imgsz
from ultralytics.yolo.utils.plotting import Annotator, colors, save_one_box
import datetime 
from azure.storage.blob import BlobServiceClient

def getOCR(im, coors):
    x,y,w, h = int(coors[0]), int(coors[1]), int(coors[2]),int(coors[3])
    im = im[y:h,x:w]
    conf = 0.2

    gray = cv2.cvtColor(im , cv2.COLOR_RGB2GRAY)
    results = reader.readtext(gray)
    ocr = ""

    for result in results:
        if len(results) == 1:
            ocr = result[1]
        if len(results) >1 and len(results[1])>6 and results[2]> conf:
            ocr = result[1]
    
    return str(ocr)

class DetectionPredictor(BasePredictor):

    def get_annotator(self, img):
        return Annotator(img, line_width=self.args.line_thickness, example=str(self.model.names))

    def preprocess(self, img):
        img = torch.from_numpy(img).to(self.model.device)
        img = img.half() if self.model.fp16 else img.float()  # uint8 to fp16/32
        img /= 255  # 0 - 255 to 0.0 - 1.0
        return img

    def postprocess(self, preds, img, orig_img):
        preds = ops.non_max_suppression(preds,
                                        self.args.conf,
                                        self.args.iou,
                                        agnostic=self.args.agnostic_nms,
                                        max_det=self.args.max_det)

        for i, pred in enumerate(preds):
            shape = orig_img[i].shape if self.webcam else orig_img.shape
            pred[:, :4] = ops.scale_boxes(img.shape[2:], pred[:, :4], shape).round()

        return preds

    def write_results(self, idx, preds, batch):
        p, im, im0 = batch
        log_string = ""
        if len(im.shape) == 3:
            im = im[None]  # expand for batch dim
        self.seen += 1
        im0 = im0.copy()
        if self.webcam:  # batch_size >= 1
            log_string += f'{idx}: '
            frame = self.dataset.count
        else:
            frame = getattr(self.dataset, 'frame', 0)

        self.data_path = p
        # save_path = str(self.save_dir / p.name)  # im.jpg
        self.txt_path = str(self.save_dir / 'labels' / p.stem) + ('' if self.dataset.mode == 'image' else f'_{frame}')
        log_string += '%gx%g ' % im.shape[2:]  # print string
        self.annotator = self.get_annotator(im0)

        det = preds[idx]
        self.all_outputs.append(det)
        if len(det) == 0:
            return log_string
        for c in det[:, 5].unique():
            n = (det[:, 5] == c).sum()  # detections per class
            log_string += f"{n} {self.model.names[int(c)]}{'s' * (n > 1)}, "
        # write
        gn = torch.tensor(im0.shape)[[1, 0, 1, 0]]  # normalization gain whwh
        for *xyxy, conf, cls in reversed(det):
            if self.args.save_txt:  # Write to file-------------------
                xywh = (ops.xyxy2xywh(torch.tensor(xyxy).view(1, 4)) / gn).view(-1).tolist()  # normalized xywh
                line = (cls, *xywh, conf) if self.args.save_conf else (cls, *xywh)  # label format
                with open(f'{self.txt_path}.txt', 'a') as f:
                    f.write(('%g ' * len(line)).rstrip() % line + '\n')

            if self.args.save or self.args.save_crop or self.args.show:  # Add bbox to image
                c = int(cls)  # integer class
                label = None if self.args.hide_labels else (
                    self.model.names[c] if self.args.hide_conf else f'{self.model.names[c]} {conf:.2f}')
                ocr = getOCR(im0,xyxy)
                if ocr != "":
                    label = ocr
                    #new code to fill completly rectangle.
                color = colors(c, True)
                xyxy = [int(x) for x in xyxy]
                cv2.rectangle(im0, (xyxy[0], xyxy[1]), (xyxy[2], xyxy[3]), color, -1)
                # Convert the image to bytes
                is_success, im_buf_arr = cv2.imencode(".jpg", im0)
                byte_im = im_buf_arr.tobytes()
                # Generate a unique name for the blob
                now = datetime.datetime.now()
                blob_name = f"myimage_{now.strftime('%Y%m%d%H%M%S')}.jpg"
 
                # Upload to Azure Blob Storage
                my_connection_string='DefaultEndpointsProtocol=https;AccountName=aiworkspace4684782811;AccountKey=W7ibcV7jRMT/zTRbAZtTgq600jthxpQdWZ/MFAjgtpEULGFteQrgxtMp7XgG/y71Gi5E1QHoI2PB+AStCeppaQ==;EndpointSuffix=core.windows.net'
                try:
                    blob_service_client = BlobServiceClient.from_connection_string(my_connection_string)
                    blob_client = blob_service_client.get_blob_client("azureml", blob_name)
 
                    print("\nUploading to Azure Storage as blob:\n\t" + "myimage.jpg")
 
                    # Upload the created file
                    blob_client.upload_blob(byte_im)
 
                except Exception as ex:
                    print('Exception:')
                    print(ex)
                #self.annotator.box_label(xyxy, label, color=colors(c, True))  -- To display text.
            if self.args.save_crop:
                imc = im0.copy()
                save_one_box(xyxy,
                             imc,
                             file=self.save_dir / 'crops' / self.model.model.names[c] / f'{self.data_path.stem}.jpg',
                             BGR=True)

        return log_string


@hydra.main(version_base=None, config_path=str(DEFAULT_CONFIG.parent), config_name=DEFAULT_CONFIG.name)
def predict(cfg):
    cfg.model = cfg.model or "yolov8n.pt"
    cfg.imgsz = check_imgsz(cfg.imgsz, min_dim=2)  # check image size
    cfg.source = cfg.source if cfg.source is not None else ROOT / "assets"
    predictor = DetectionPredictor(cfg)
    return predictor()


if __name__ == "__main__":
    reader = easyocr.Reader(['en'])
    a=predict()
    print(a)
