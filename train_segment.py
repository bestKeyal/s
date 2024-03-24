import argparse

import segment

if __name__ == '__main__':
    args = argparse.Namespace(
        model_name='DR_UNet',
        dims=32,
        epochs=50,
        batch_size=16,
        lr=2e-4,
        height=256,
        width=256,
        channel=1,
        pred_height=1024,
        pred_width=1024,
        total_samples=2814,
        invalid_samples=272,
        regularize=False,

        record_dir='record',
        train_record_name='DataSet_Train',

        test_image_dir='test-images',

        invalid_record_name='DataSet_Valid',
        invalid_volume_dir='record',

        gt_mask_dir = 'masks_jpg_valid',
    )

    Seg = segment.Segmentation(args)
    # start training
    Seg.train()
