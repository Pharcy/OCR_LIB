import layoutparser as lp
import cv2

model = lp.Detectron2LayoutModel(
        config_path="lp://NewspaperNavigator/faster_rcnn_R_50_FPN_3x/config",
        label_map={0: "Text", 1: "Title", 2: "List", 3: "Table", 4: "Figure"},
        score_thresh= 0.5
        )
        
layout = model.detect(image)