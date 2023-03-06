yolo_config = {
    # Basic
    'img_size': (416, 416, 3),
    'anchors': [12, 16, 19, 36, 40, 28, 36, 75, 76, 55, 72, 146, 142, 110, 192, 243, 459, 401],
    # 'anchors': [3091, 3181, 3648, 7416, 3804, 4187, 4094, 5435, 5291, 6229, 5621, 3649, 6129, 7902, 8033, 6997, 9207, 4878],
    'strides': [8, 16, 32],
    'xyscale': [1.2, 1.1, 1.05],

    # Training
    'iou_loss_thresh': 0.5,
    'batch_size': 8,
    'num_gpu': 1,  # 2,

    # Inference
    'max_boxes': 1,
    'iou_threshold': 0.0,
    'score_threshold': 0.0,
}
