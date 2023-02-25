yolo_config = {
    # Basic
    'img_size': (416, 416, 3),
    # 'anchors': [12, 16, 19, 36, 40, 28, 36, 75, 76, 55, 72, 146, 142, 110, 192, 243, 459, 401],
    'anchors': [130, 133, 150, 289, 157, 183, 198, 233, 226, 373, 228, 279, 235, 154, 285, 319, 364, 260],
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
