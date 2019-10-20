from mmdet.apis import init_detector, inference_detector, show_result
import cv2

# 首先下载模型文件https://s3.ap-northeast-2.amazonaws.com/open-mmlab/mmdetection/models/faster_rcnn_r50_fpn_1x_20181010-3d1b3351.pth
config_file = 'configs/cascade_rcnn_r50_fpn_1x.py'
checkpoint_file = 'work_dirs/cascade_rcnn_r50_fpn_1x/epoch_12.pth'


# # 初始化模型
# model = init_detector(config_file, checkpoint_file)
#
# # 测试一张图片
# img = 'data/coco/train2017/IMG_9372.MOV0035.jpg'
# result = inference_detector(model, img)
# show_result(img, result, model.CLASSES)

# 测试一系列图片
# imgs = ['demo/demo.jpg']
# for i, result in enumerate(inference_detector(model, imgs, device='cuda:0')):
#     show_result(imgs[i], result, model.CLASSES, out_file='result_{}.jpg'.format(i))
def main(show_one_pic=False):
    model = init_detector(config_file, checkpoint_file)
    if show_one_pic:
        img = 'data/coco/train2017/IMG_9372.MOV0035.jpg'
        result = inference_detector(model, img)
        print(result)
        show_result(img, result, model.CLASSES)
        return

    # fourcc = cv2.VideoWriter_fourcc(*'XVID')
    # out = cv2.VideoWriter('out.avi', fourcc, 25.0, (1280, 720))
    camera = cv2.VideoCapture('/home/rex/mmdetection/IMG_9372~1.avi')

    print('Press "Esc", "q" or "Q" to exit.')
    while True:
        ret_val, img = camera.read()
        result = inference_detector(model, img)

        ch = cv2.waitKey(1)
        if ch == 27 or ch == ord('q') or ch == ord('Q'):
            break

        # frame = show_result(img, result, ('wing', 'drumstick', 'pai', 'tart', 'none'), score_thr=0.35, wait_time=1,
        #                     show=False)
        show_result(img, result, model.CLASSES, score_thr=0.35, wait_time=1)
        # out.write(frame)


if __name__ == '__main__':
    main(True)
