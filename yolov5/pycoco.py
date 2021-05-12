import pdb

import numpy as np
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval

cocoGt = COCO("/data/data/MSCOCO2017/annotations/instances_val2017.json")
cocoDt = cocoGt.loadRes("results.json")

imgIds = sorted(cocoGt.getImgIds())
imgIds = imgIds[0:100]
imgId = imgIds[np.random.randint(100)]

cocoEval = COCOeval(cocoGt, cocoDt, "bbox")
cocoEval.params.imgIds = imgIds
cocoEval.evaluate()
cocoEval.accumulate()
cocoEval.summarize()
