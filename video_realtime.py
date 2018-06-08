import os
import numpy as np
import coco
import model as modellib
import visualize
from model import log
import cv2
import time
from skimage.measure import find_contours
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib.lines as lines
from matplotlib.patches import Polygon
import IPython.display

ROOT_DIR = os.getcwd()
# Directory to save logs and trained model
MODEL_DIR = os.path.join(ROOT_DIR, "mylogs")
# Local path to trained weights file
COCO_MODEL_PATH = os.path.join(ROOT_DIR, "mask_rcnn_coco_humanpose.h5")
class InferenceConfig(coco.CocoConfig):
    GPU_COUNT = 1
    IMAGES_PER_GPU = 1
    KEYPOINT_MASK_POOL_SIZE = 7

inference_config = InferenceConfig()

# Recreate the model in inference mode
model = modellib.MaskRCNN(mode="inference",
                          config=inference_config,
                          model_dir=MODEL_DIR)

# Get path to saved weights

model_path = os.path.join(ROOT_DIR, "mask_rcnn_coco_humanpose.h5")
assert model_path != "", "Provide path to trained weights"
print("Loading weights from ", model_path)
model.load_weights(model_path, by_name=True)

class_names = ['BG', 'person']
def display_keypoints(image, boxes, keypoints, class_ids, class_names,
                      skeleton = [], scores=None, title="",
                      figsize=(16, 16), ax=None):
    """
    boxes: [num_persons, (y1, x1, y2, x2)] in image coordinates.
    keypoints: [num_persons, num_keypoint, 3]
    class_ids: [num_persons]
    class_names: list of class names of the dataset
    scores: (optional) confidence scores for each box
    figsize: (optional) the size of the image.
    """
    # Number of persons
    N = boxes.shape[0]
    keypoints = np.array(keypoints).astype(int)
    print("keypoint_shape:", np.shape(keypoints))
    if not N:
        print("\n*** No persons to display *** \n")
    else:
        assert boxes.shape[0] == keypoints.shape[0] == class_ids.shape[0]

    if not ax:
        _, ax = plt.subplots(1, figsize=figsize)

    # Generate random colors
    colors = visualize.random_colors(N)

    # Show area outside image boundaries.
    height, width = image.shape[:2]
    ax.set_ylim(height + 10, -10)
    ax.set_xlim(-10, width + 10)
    ax.axis('off')
    ax.set_title(title)
    skeleton_image = image.astype(np.float32).copy()
    for i in range(N):
        color = colors[i]

        # Bounding box
        if not np.any(boxes[i]):
            # Skip this instance. Has no bbox. Likely lost in image cropping.
            continue
        y1, x1, y2, x2 = boxes[i]
        p = patches.Rectangle((x1, y1), x2 - x1, y2 - y1, linewidth=2,
                              alpha=0.7, linestyle="dashed",
                              edgecolor=color, facecolor='none')
        ax.add_patch(p)

        # Label
        class_id = class_ids[i]
        score = scores[i] if scores is not None else None
        label = class_names[class_id]
        # x = random.randint(x1, (x1 + x2) // 2)
        caption = "{} {:.3f}".format(label, score) if score else label
        ax.text(x1, y1 + 8, caption,
                color='w', size=11, backgroundcolor="none")
        # Keypoints: num_person, num_keypoint, 3
        for Joint in keypoints[i]:
            if(Joint[2]!=0):
                circle = patches.Circle((Joint[0],Joint[1]),radius=1,edgecolor=color,facecolor='none')
                ax.add_patch(circle)

        # Skeleton: 11*2
        limb_colors = [[0, 0, 255], [0, 170, 255], [0, 255, 170], [0, 255, 0], [170, 255, 0],
                  [255, 170, 0], [255, 0, 0], [255, 0, 170], [170, 0, 255], [170,170,0],[170,0,170]]
        if(len(skeleton)):
            skeleton = np.reshape(skeleton,(-1,2))
            neck = np.array((keypoints[i, 5, :] + keypoints[i,6,:])/2).astype(int)
            if(keypoints[i, 5, 2] == 0 or keypoints[i,6,2] == 0):
                neck = [0,0,0]
            limb_index = -1
            for limb in skeleton:
                limb_index += 1
                start_index, end_index = limb  # connection joint index from 0 to 16
                if(start_index == -1):
                    Joint_start = neck
                else:
                    Joint_start = keypoints[i][start_index]
                if(end_index == -1):
                    Joint_end = neck
                else:
                    Joint_end = keypoints[i][end_index]
                # both are Annotated
                # Joint:(x,y,v)
                if ((Joint_start[2] != 0) & (Joint_end[2] != 0)):
                    # print(color)
                    cv2.line(skeleton_image, tuple(Joint_start[:2]), tuple(Joint_end[:2]), limb_colors[limb_index],5)
    ax.imshow(skeleton_image.astype(np.uint8))
    plt.show()




cap = cv2.VideoCapture(0)
while(1):
    # get a frame
    ret, frame = cap.read()
    "BGR->RGB"
    rgb_frame = frame[:,:,::-1]
    print(np.shape(frame))
    # Run detection
    t = time.time()
    results = model.detect_keypoint([rgb_frame], verbose=0)
    # show a frame
    t = time.time() - t
    print(1.0 / t)
    r = results[0]  # for one image
    log("rois", r['rois'])
    log("keypoints", r['keypoints'])
    log("class_ids", r['class_ids'])
    log("keypoints", r['keypoints'])
    log("masks", r['masks'])
    log("scores", r['scores'])
    result_image = display_keypoints(frame,r['rois'],r['keypoints'],r['class_ids'],class_names,skeleton = inference_config.LIMBS)

    cv2.imshow('Detect image', result_image)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
cap.release()
cv2.destroyAllWindows()
