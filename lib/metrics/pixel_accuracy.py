
class PixelAccuracy:
    def __init__(self, threshold=0.5):
        self.scores_per_image = []
        self.threshold = threshold

    def reset(self):
        self.scores_per_image = []

    def update(self, y_pred: Tensor, y_true: Tensor):
        batch_size = y_true.size(0)

        y_pred = y_pred.detach().view(batch_size, -1) > self.threshold
        y_true = y_true.detach().view(batch_size, -1) > 0.5

        correct = (y_true == y_pred).float()
        accuracy = correct.sum(1) / y_true.size(1)

        self.scores_per_image.extend(accuracy.cpu().numpy())

    def __str__(self):
        return '%.4f' % self.value()

    def value(self):
        return np.mean(self.scores_per_image)

    def log_to_tensorboard(self, saver: SummaryWriter, prefix, step):
        saver.add_scalar(prefix + '/value', self.value(), step)
        saver.add_histogram(prefix + '/histogram', np.array(self.scores_per_image), step)
