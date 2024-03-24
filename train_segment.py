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
        total_samples=75,
        invalid_samples=0,
        regularize=False,
        record_dir='',
        train_record_name='DataSet',
        test_image_dir='test-images',
        invalid_record_name='',
        gt_mask_dir='',
        invalid_volume_dir=''
    )

    Seg = segment.Segmentation(args)
    # start training
    Seg.train()
