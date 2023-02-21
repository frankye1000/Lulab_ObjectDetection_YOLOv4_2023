import math

def get_iou(bbox_ai, bbox_gt):
    iou_x = max(bbox_ai[0], bbox_gt[0]) # x
    iou_y = max(bbox_ai[1], bbox_gt[1]) # y
    iou_w = min(bbox_ai[2]+bbox_ai[0], bbox_gt[2]+bbox_gt[0]) - iou_x # w
    iou_w = max(iou_w, 0)
    # print(f'{iou_w=}')
    iou_h = min(bbox_ai[3]+bbox_ai[1], bbox_gt[3]+bbox_gt[1]) - iou_y # h
    iou_h = max(iou_h, 0)
    # print(f'{iou_h=}')

    iou_area = iou_w * iou_h
    # print(f'{iou_area=}')
    all_area = bbox_ai[2]*bbox_ai[3] + bbox_gt[2]*bbox_gt[3] - iou_area
    # print(f'{all_area=}')

    if all_area == 0:  # 有可能雙方皆沒有重疊，或是一開始就沒有label
        return 0
    return max(iou_area/all_area, 0)


def calculate_ciou(box_1, box_2):
    """
    calculate ciou
    :param box_1: (x0, y0, x1, y1)
    :param box_2: (x0, y0, x1, y1)
    :return: value of ciou
    """
    # perfect
    if box_1==box_2:  
        return 1.0

    # calculate area of each box
    width_1  = box_1[2] - box_1[0]
    height_1 = box_1[3] - box_1[1]
    area_1   = width_1 * height_1

    width_2  = box_2[2] - box_2[0]
    height_2 = box_2[3] - box_2[1]
    area_2   = width_2 * height_2

    # calculate center point of each box
    center_x1 = (box_1[2] - box_1[0]) / 2
    center_y1 = (box_1[3] - box_1[1]) / 2
    center_x2 = (box_2[2] - box_2[0]) / 2
    center_y2 = (box_2[3] - box_2[1]) / 2

    # calculate square of center point distance
    p2 = (center_x2 - center_x1) ** 2 + (center_y2 - center_y1) ** 2

    # calculate square of the diagonal length
    width_c  = max(box_1[2], box_2[2]) - min(box_1[0], box_2[0])
    height_c = max(box_1[3], box_2[3]) - min(box_1[1], box_2[1])
    c2 = width_c ** 2 + height_c ** 2

    # find the edge of intersect box
    left   = max(box_1[0], box_2[0])
    top    = max(box_1[1], box_2[1])
    bottom = min(box_1[3], box_2[3])
    right  = min(box_1[2], box_2[2])

    # calculate the intersect area
    area_intersection = (right - left) * (bottom - top)

    # calculate the union area
    area_union = area_1 + area_2 - area_intersection

    # calculate iou
    iou = float(area_intersection) / area_union

    # calculate v
    arctan = math.atan(float(width_2) / height_2) - math.atan(float(width_1) / height_1)
    v = (4.0 / math.pi ** 2) * (arctan ** 2)

    # calculate alpha
    alpha = float(v) / (1 - iou + v)

    # calculate ciou(iou - p2 / c2 - alpha * v)
    ciou = iou - float(p2) / c2 - alpha * v

    return ciou