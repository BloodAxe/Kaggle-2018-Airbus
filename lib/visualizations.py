import albumentations as A
import cv2
import numpy as np
from skimage.color import label2rgb
import warnings
from lib.common.rbox_util import rbbox2corners
from lib.common.torch_utils import to_numpy
from lib.dataset.common import ID_KEY, MASK_KEY, IMAGE_KEY


def visualize_bbox(img, bbox, color, thickness=2, **kwargs):
    x_min, y_min, x_max, y_max = bbox
    cv2.rectangle(img, (x_min, y_min), (x_max, y_max), color=color, thickness=thickness)
    return img


def visualize_rbbox(img, bbox, color, thickness=2, **kwargs):
    points = rbbox2corners(bbox)
    points = np.expand_dims(points, 1)
    points = points.astype(int)
    if thickness == cv2.FILLED:
        cv2.fillConvexPoly(img, points, color)
    else:
        cv2.polylines(img, [points], True, color, thickness=thickness, lineType=cv2.LINE_AA)


def visualize_titles(img, pt, title, box_color, text_color=(255, 255, 255), thickness=2, font_thickness=1, font_scale=0.5, **kwargs):
    x_min, y_min = pt
    x_min = int(x_min)
    y_min = int(y_min)
    ((text_width, text_height), _) = cv2.getTextSize(title, cv2.FONT_HERSHEY_SIMPLEX, font_scale, font_thickness)
    cv2.rectangle(img, (x_min, y_min - int(1.3 * text_height)), (x_min + text_width, y_min), box_color, -1)
    cv2.putText(img, title, (x_min, y_min - int(0.3 * text_height)), cv2.FONT_HERSHEY_SIMPLEX, font_scale, text_color,
                font_thickness, lineType=cv2.LINE_AA)
    return img


def vis_image_opencv(img, boxes, scores, color):
    '''Visualize a color image.

    Args:
      img: (PIL.Image/tensor) image to visualize.
      boxes: (tensor) bounding boxes, sized [#obj, 4].
      scores: (list) confidence scores.

    Reference:
      https://github.com/chainer/chainercv/blob/master/chainercv/visualizations/vis_bbox.py
      https://github.com/chainer/chainercv/blob/master/chainercv/visualizations/vis_image.py
    '''
    # Plot image

    img = img.copy()

    alpha = 0.5

    # Plot boxes
    # output = img.copy()

    for bbox, score in zip(boxes, scores):
        bbox = np.clip(np.array(bbox), a_min=-1147483648, a_max=1147483647)
        bbox = bbox.astype(int)

        x_min, y_min, x_max, y_max = bbox
        bbox = (int(x_min), int(y_min), int(x_max), int(y_max))

        color = tuple(np.array(color, dtype=float))
        caption = '{0:.2f}'.format(score)

        overlay = img.copy()
        visualize_bbox(overlay, bbox, color)

        cx, cy = (bbox[0] + bbox[2]) // 2, (bbox[1] + bbox[3]) // 2
        visualize_titles(overlay, (cx, cy), caption, color)

        img = cv2.addWeighted(img, alpha, overlay, 1 - alpha, 0)

    return img


def draw_rbboxes(img, boxes, scores, color, thickness=1, show_scores=True):
    '''Visualize a color image.

    Args:
      img: (PIL.Image/tensor) image to visualize.
      boxes: (tensor) bounding boxes, sized [#obj, 4].
      scores: (list) confidence scores.

    Reference:
      https://github.com/chainer/chainercv/blob/master/chainercv/visualizations/vis_bbox.py
      https://github.com/chainer/chainercv/blob/master/chainercv/visualizations/vis_image.py
    '''
    # Plot image
    img = img.copy()

    alpha = 0.5

    # Plot boxes
    # output = img.copy()

    color = tuple(np.array(color, dtype=float))
    for bbox, score in zip(boxes, scores):

        overlay = img.copy()
        visualize_rbbox(overlay, bbox, color, thickness=thickness)

        if show_scores:
            cx, cy = bbox[:2]
            caption = '{0:.2f}'.format(score)
            visualize_titles(overlay, (cx, cy), caption, color)

        img = cv2.addWeighted(img, alpha, overlay, 1 - alpha, 0)

    return img

#
# def visualize_ssd_predictions(data, normalize=A.Normalize(), show_groundtruth=True):
#     batch_images = to_numpy(data[IMAGE_KEY])
#     pred_ssd_bboxes = data['pred_ssd_bboxes']
#     true_ssd_bboxes = data['true_ssd_bboxes']
#     pred_ssd_labels = data['pred_ssd_labels']
#     true_ssd_labels = data['true_ssd_labels']
#
#     box_coder = FPNSSDBoxCoder()
#
#     images = []
#
#     font_scale = 1
#     font_thickness = 1
#     text_color = (255, 255, 255)
#
#     # Render pred bboxes
#     for image, bboxes, labels in zip(batch_images, pred_ssd_bboxes, pred_ssd_labels):
#         image = np.moveaxis(image, 0, -1)
#         image = (image * np.array(normalize.std) + np.array(normalize.mean)) * normalize.max_pixel_value
#         image = image.astype(np.uint8).copy()
#
#         title = str(id)
#         cv2.putText(image, title, (5, 5), cv2.FONT_HERSHEY_SIMPLEX, font_scale, text_color, font_thickness, lineType=cv2.LINE_AA)
#
#         bboxes, labels, probs = box_coder.decode(bboxes, labels)
#
#         bboxes = to_numpy(bboxes)
#         labels = to_numpy(labels)
#         probs = to_numpy(probs)
#         if len(labels):
#             image = vis_image_opencv(image, bboxes, probs, (255, 0, 0))
#
#         images.append(image)
#
#     # Render true bboxes
#     if show_groundtruth:
#         new_images = []
#         for image, bboxes, labels in zip(images, true_ssd_bboxes, true_ssd_labels):
#             # image = np.moveaxis(image, 0, -1)
#             # image = (image * np.array(normalize.std) + np.array(normalize.mean)) * normalize.max_pixel_value
#             # image = image.astype(np.uint8)
#
#             bboxes, labels, probs = box_coder.decode(bboxes, labels)
#
#             bboxes = to_numpy(bboxes)
#             labels = to_numpy(labels)
#             probs = to_numpy(probs)
#
#             if len(labels):
#                 image = vis_image_opencv(image, bboxes, probs, (0, 255, 0))
#
#                 new_images.append(image)
#         images = new_images
#
#     return images


def visualize_rssd_predictions(data, box_coder, normalize=A.Normalize(), show_groundtruth=True):
    batch_ids = data[ID_KEY]
    batch_images = to_numpy(data[IMAGE_KEY])
    pred_ssd_bboxes = to_numpy(data['pred_ssd_bboxes'])
    true_ssd_bboxes = to_numpy(data['true_ssd_bboxes'])
    pred_ssd_labels = to_numpy(data['pred_ssd_labels'])
    true_ssd_labels = to_numpy(data['true_ssd_labels'])

    images = []

    font_scale = 1
    font_thickness = 1
    text_color = (255, 255, 255)

    # Render pred bboxes
    for id, image, bboxes, labels in zip(batch_ids, batch_images, pred_ssd_bboxes, pred_ssd_labels):
        image = np.moveaxis(image, 0, -1)
        image = (image * np.array(normalize.std) + np.array(normalize.mean)) * normalize.max_pixel_value
        image = image.astype(np.uint8).copy()

        title = str(id)
        cv2.putText(image, title, (5, 25), cv2.FONT_HERSHEY_SIMPLEX, font_scale, text_color, font_thickness, lineType=cv2.LINE_AA)

        bboxes, probs = box_coder.decode(bboxes, labels)

        if len(probs) > 1024:
            warnings.warn(f'Too many {len(bboxes)} RSSD predictions. Will render first 1024')
            bboxes = bboxes[:1024]
            probs = probs[:1024]

        if len(probs):
            image = draw_rbboxes(image, bboxes, probs, (255, 0, 0), thickness=3)

        images.append(image)

    # Render true bboxes
    if show_groundtruth:
        new_images = []
        for image, bboxes, labels in zip(images, true_ssd_bboxes, true_ssd_labels):
            # image = np.moveaxis(image, 0, -1)
            # image = (image * np.array(normalize.std) + np.array(normalize.mean)) * normalize.max_pixel_value
            # image = image.astype(np.uint8)

            bboxes, probs = box_coder.decode(bboxes, labels)

            if len(labels):
                image = draw_rbboxes(image, bboxes, probs, (0, 255, 0), thickness=1, show_scores=False)

            new_images.append(image)
        images = new_images

    return images


def visualize_mask_predictions(data, normalize=A.Normalize(), threshold=0.5, show_groundtruth=True):
    batch_ids = data[ID_KEY]
    batch_images = to_numpy(data[IMAGE_KEY])
    pred_mask = to_numpy(data['pred_mask'])
    true_mask = to_numpy(data['true_mask'])

    images = []

    alpha = 0.5
    font_scale = 0.5
    font_thickness = 1
    text_color = (255, 255, 255)

    def prediction_to_rgb_mask(image, pred, ship_color=(0, 255, 0), edge_color=(255, 0, 0)):
        ship_mask = pred[0, ...] > threshold
        image[ship_mask] = ship_color

        if pred.shape[0] > 1:
            edge_mask = pred[1, ...] > threshold
            image[edge_mask] = edge_color
        return image

    # Render pred bboxes
    for id, image, p, t in zip(batch_ids, batch_images, pred_mask, true_mask):
        image = np.moveaxis(image, 0, -1)
        image = (image * np.array(normalize.std) + np.array(normalize.mean)) * normalize.max_pixel_value
        image = image.astype(np.uint8).copy()

        overlay = prediction_to_rgb_mask(image.copy(), p)
        overlay = cv2.addWeighted(image, alpha, overlay, 1 - alpha, 0)

        if show_groundtruth:
            overlay2 = prediction_to_rgb_mask(image.copy(), t)
            overlay2 = cv2.addWeighted(image, alpha, overlay2, 1 - alpha, 0)
            overlay = np.hstack((overlay, overlay2))

        title = str(id)
        cv2.putText(overlay, title, (5, 25), cv2.FONT_HERSHEY_SIMPLEX, font_scale, text_color, font_thickness, lineType=cv2.LINE_AA)

        images.append(overlay)

    return images


def visualize_cls_predictions(data, normalize=A.Normalize(), threshold=0.5, show_groundtruth=True):
    batch_ids = data[ID_KEY]
    batch_images = to_numpy(data[IMAGE_KEY])
    batch_masks = to_numpy(data[MASK_KEY])
    batch_pred_has_ship = to_numpy(data['pred_has_ship'])
    batch_true_has_ship = to_numpy(data['true_has_ship'])

    images = []

    alpha = 0.5
    font_scale = 0.5
    font_thickness = 1
    text_color = (255, 255, 255)

    def prediction_to_rgb_mask(image, pred, ship_color=(0, 255, 0), edge_color=(255, 0, 0)):
        ship_mask = pred[0, ...] > threshold
        image[ship_mask] = ship_color

        if pred.shape[0] > 1:
            edge_mask = pred[1, ...] > threshold
            image[edge_mask] = edge_color
        return image

    # Render pred bboxes
    for id, image, mask, p, t in zip(batch_ids, batch_images, batch_masks, batch_pred_has_ship, batch_true_has_ship):
        image = np.moveaxis(image, 0, -1)
        image = (image * np.array(normalize.std) + np.array(normalize.mean)) * normalize.max_pixel_value
        image = image.astype(np.uint8).copy()

        # Put ships overlay
        ships_rgb = (label2rgb(mask, bg_label=0) * 255).astype(np.uint8)
        image = cv2.addWeighted(image, alpha, ships_rgb, 1 - alpha, 0)

        overlay = image.copy()
        # Put image title
        title = str(id)
        cv2.putText(overlay, title, (5, 25), cv2.FONT_HERSHEY_SIMPLEX, font_scale, text_color, font_thickness, lineType=cv2.LINE_AA)

        # Put prediction confidence
        cv2.putText(overlay, '{0:.2f}'.format(float(p)), (10, image.shape[0] - 10), cv2.FONT_HERSHEY_SIMPLEX, font_scale, (255, 0, 0), font_thickness, lineType=cv2.LINE_AA)

        # Put true label
        if show_groundtruth:
            cv2.putText(overlay, '{0:.2f}'.format(float(t)), (image.shape[1] - 40, image.shape[0] - 10), cv2.FONT_HERSHEY_SIMPLEX, font_scale, (0, 255, 0), font_thickness, lineType=cv2.LINE_AA)

        overlay = cv2.addWeighted(image, alpha, overlay, 1 - alpha, 0)
        images.append(overlay)

    return images
