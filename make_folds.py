import os

import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.utils import check_random_state

from lib.dataset import common as D


def _count_ships(args):
    image_id, groundtruth = args
    rle = D.get_rle_from_groundtruth(groundtruth, image_id)
    mask = D.rle2instancemask(rle)
    return mask.max()


def get_folds_by_big_image(ids, n_folds=4, random_state=None):
    big_images = pd.read_csv('lib/dataset/big-images-ids_v2.csv')
    big_images['ImageId'] = big_images['ImageId'].str.lower()
    big_images = big_images[big_images['ImageId'].isin(ids)]

    le = LabelEncoder()
    big_images['BigImageId'] = le.fit_transform(big_images['BigImageId'])

    unique_big_img_ids = np.unique(big_images['BigImageId'])
    n_big_images = len(unique_big_img_ids)
    folds = np.array(list(range(n_folds)) * n_big_images)[:n_big_images]

    sorted_indexes = list(range(n_big_images))
    rnd = check_random_state(random_state)
    rnd.shuffle(sorted_indexes)

    folds_for_big_images = folds[sorted_indexes]
    big_images['Fold'] = big_images['BigImageId'].apply(lambda x: folds_for_big_images[x])

    return big_images


if __name__ == '__main__':
    data_dir = 'd:\\datasets\\airbus'
    n_folds = 4

    groundtruth = 'train_ship_segmentations_v2.csv'
    groundtruth = pd.read_csv(os.path.join(data_dir, groundtruth))
    groundtruth['ImageId'] = groundtruth['ImageId'].str.lower()

    # big_images = pd.read_csv('lib/dataset/big-images-ids_v2.csv')
    # big_images['ImageId'] = big_images['ImageId'].str.lower()

    train_ids = groundtruth.ImageId.unique()
    train_ids_nonempty = groundtruth[~groundtruth.EncodedPixels.isna()].ImageId.unique()
    N = len(train_ids)
    print(len(train_ids), len(train_ids_nonempty))

    # num_ships = []
    # with Pool(8) as wp:
    #     for n in tqdm(wp.imap(_count_ships, zip(train_ids, [groundtruth] * N), chunksize=512), total=N):
    #         num_ships.append(n)
    #
    # num_ships = np.array(num_ships)
    #
    # ships_count_df = pd.DataFrame.from_dict({'ImageId': train_ids, 'Ships': num_ships})
    # ships_count_df.to_csv('lib/dataset/ship_counts.csv', index=None)

    folds = get_folds_by_big_image(train_ids, 4, random_state=1234)
    folds.to_csv('lib/dataset/folds.csv', index=None)

    folds = get_folds_by_big_image(train_ids_nonempty, 4, random_state=1234)
    folds.to_csv('lib/dataset/folds_nonempty.csv', index=None)

    # groundtruth.to_csv('lib/dataset/big-images-ids_v2.csv', index=None)

    # folds_by_rnd = D.get_folds_vector(train_ids, big_images, n_folds=4, random_state=42)
    # pd.DataFrame.from_dict({'id': train_ids, 'fold': folds_by_rnd}).to_csv('data/folds_by_rnd_42.csv', index=False)
    # pd.DataFrame.from_dict({'id': train_ids, 'fold': folds_by_rnd}).to_csv('data/folds_by_ships_non_empty.csv', index=False)
