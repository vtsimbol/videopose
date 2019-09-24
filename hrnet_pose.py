import cv2
import numpy as np
import time

import torch
from joints_detectors.hrnet.lib.config import cfg
from joints_detectors.hrnet.lib.detector.yolo.human_detector import main as yolo_det
from joints_detectors.hrnet.pose_estimation.video import getTwoModel
from joints_detectors.hrnet.pose_estimation.utilitys import PreProcess
from joints_detectors.hrnet.lib.core.inference import get_final_preds


class HrnetPose(object):
    def __init__(self):
        self.joint_pairs = [
                            # лицо
                            [0, 1, (0, 153, 255)],
                            [1, 3, (0, 153, 255)],
                            [0, 2, (0, 153, 255)],
                            [2, 4, (0, 153, 255)],
                            # левое плечо
                            [5, 7, (0, 0, 255)],
                            # левое предплечье
                            [7, 9, (0, 0, 255)],
                            # правое плечо
                            [6, 8, (0, 204, 0)],
                            # правое предплечье
                            [8, 10, (0, 204, 0)],
                            # плечи
                            [5, 6, (255, 102, 0)],
                            # левый бок
                            [5, 11, (255, 102, 0)],
                            # правый бок
                            [6, 12, (255, 102, 0)],
                            # таз
                            [11, 12, (255, 102, 0)],
                            # левое бедро
                            [11, 13, (0, 255, 255)],
                            # левая голень
                            [13, 15, (0, 255, 255)],
                            # правое бедро
                            [12, 14, (255, 204, 0)],
                            # правая голень
                            [14, 16, (255, 204, 0)]]
        self.bboxModel, self.poseModel = getTwoModel()

    def predict_bbox(self, image):
        return yolo_det(image, self.bboxModel)

    def predict_kpts(self, image, bboxs, scores):
        if len(bboxs) > 0:
            inputs, origin_img, center, scale = PreProcess(image, bboxs, scores, cfg)
            if len(inputs) > 0:
                with torch.no_grad():
                    inputs = inputs[:, [2, 1, 0]]
                    output = self.poseModel(inputs.cuda())
                    preds, maxvals = get_final_preds(cfg, output.clone().cpu().numpy(), np.asarray(center), np.asarray(scale))
                result = np.concatenate((preds[0], maxvals[0]), 1)
                return result
        return None

    def visualisation(self, image, keypoints):
        # keypoints : (17, 3)  3-->(x, y, score)
        for item in keypoints:
            score = item[-1]
            if score > 0.1:
                x, y = int(item[0]), int(item[1])
                cv2.circle(image, (x, y), 2, (255, 0, 38), 5)
        for pair in self.joint_pairs:
            j, j_parent, color = pair
            pt1 = (int(keypoints[j][0]), int(keypoints[j][1]))
            pt2 = (int(keypoints[j_parent][0]), int(keypoints[j_parent][1]))
            cv2.line(image, pt1, pt2, color, 5)
        return image


if __name__ == "__main__":
    hrnet = HrnetPose()
    cap = cv2.VideoCapture(0)
    fps_time = 0
    while True:
        ret, img = cap.read()
        if ret:
            keypoints = hrnet.predict(img)
            if keypoints is not None:
                img = hrnet.visualisation(img, keypoints)
            cv2.putText(img, "FPS: %f" % (1.0 / (time.time() - fps_time)), (10, 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
            cv2.imshow('camera', img)
            if cv2.waitKey(1) == 27:
                break
            fps_time = time.time()
        else:
            break
    cap.release()
    cv2.destroyAllWindows()
