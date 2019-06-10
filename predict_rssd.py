import argparse
import os

from torch.backends import cudnn

from lib.common.checkpoint import restore_checkpoint
from lib.common.fs import auto_file
from lib.common.random import set_manual_seed
from lib.dataset import rssd_dataset as D
from lib.dataset.common import all_test_ids, get_transform
from lib.models.models_factory import get_model
from lib.models.rssd.rsdd import RSSDBoxCoder


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    parser.add_argument('-dd', '--data-dir', type=str, default='d:\\datasets\\airbus', help='Data dir')
    parser.add_argument('-m', '--model', type=str, default='rssd512', help='')
    parser.add_argument('-b', '--batch-size', type=int, default=4, help='Batch Size during training, e.g. -b 64')
    parser.add_argument('-es', '--early-stopping', type=int, default=None,
                        help='Maximum number of epochs without improvement')
    parser.add_argument('-r', '--resume', type=str, default=None, help='Checkpoint filename to resume')
    parser.add_argument('-w', '--workers', default=4, type=int, help='Num workers')
    parser.add_argument('-p', '--patch-size', type=int, default=768, help='')

    args = parser.parse_args()
    set_manual_seed(args.seed)

    fname = auto_file(args.resume)
    exp_dir = os.path.dirname(fname)
    prefix = os.path.splitext(os.path.basename(fname))[0]

    model = get_model(args.model, num_classes=1).cuda()

    start_epoch, train_history, best_score = restore_checkpoint(fname, model)
    print(train_history)

    testset_full = D.RSSDDataset(sample_ids=all_test_ids(args.data_dir),
                                 test=True,
                                 data_dir=args.data_dir,
                                 transform=get_transform(training=False, width=args.patch_size, height=args.patch_size),
                                 box_coder=RSSDBoxCoder(args.patch_size, args.patch_size))
    test_predictions = model.predict_as_csv(testset_full, batch_size=args.batch_size, workers=args.workers)
    test_predictions.to_csv(os.path.join(exp_dir, f'{prefix}_test_predictions.csv'), index=None)
    print('Predictions saved')


if __name__ == '__main__':
    cudnn.benchmark = True
    main()
