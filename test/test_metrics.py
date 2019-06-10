from tensorboardX import SummaryWriter
import numpy as np
from tensorboardX.x2num import makenp

if __name__ == '__main__':
    pr1 = PRCurveMeter()
    pr2 = PRCurveMeter2()

    y_true = np.array([0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1,1,1])
    y_true = np.dstack([y_true,y_true,y_true])

    y_pred = np.array([0.9, 0.8, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1, 0.1])
    y_pred  = np.dstack([y_pred,y_pred,y_pred])

    pr1.update(y_pred, y_true)
    pr2.update(y_pred, y_true)

    with SummaryWriter(comment='_test_metrics') as summary_writer:
        summary_writer.add_image('y_true', y_true)
        summary_writer.add_image('y_pred', y_pred)

        summary_writer.add_pr_curve_raw('pr1',
                                        true_positive_counts=pr1.tp,
                                        true_negative_counts=pr1.tn,
                                        false_negative_counts=pr1.fn,
                                        false_positive_counts=pr1.fp,
                                        precision=pr1.precision(),
                                        recall=pr1.recall(),
                                        global_step=1)

        summary_writer.add_pr_curve_raw('pr2',
                                        true_positive_counts=pr2.tp,
                                        true_negative_counts=pr2.tn,
                                        false_negative_counts=pr2.fn,
                                        false_positive_counts=pr2.fp,
                                        precision=pr2.precision(),
                                        recall=pr2.recall(),
                                        global_step=1)

        summary_writer.add_pr_curve('pr3',
                                    labels=y_true,
                                    predictions=y_pred,
                                    num_thresholds=127,
                                    global_step=1)

        true_positive_counts = [75, 64, 21, 5, 0]
        false_positive_counts = [150, 105, 18, 0, 0]
        true_negative_counts = [0, 45, 132, 150, 150]
        false_negative_counts = [0, 11, 54, 70, 75]
        precision = [0.3333333, 0.3786982, 0.5384616, 1.0, 0.0]
        recall = [1.0, 0.8533334, 0.28, 0.0666667, 0.0]
        summary_writer.add_pr_curve_raw('prcurve with raw data', true_positive_counts,
                                false_positive_counts,
                                true_negative_counts,
                                false_negative_counts,
                                precision,
                                recall, 0)
    print('Done')