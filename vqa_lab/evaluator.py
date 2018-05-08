import torch

class LossEvaluator(object):

    def __init__(self):
        super(LossEvaluator, self).__init__()
        self.reset()

    def reset(self):

        self.total_loss = 0.0
        self.total_step = 0.0

    def _add_batch(self, loss):

        self.total_loss += loss
        self.total_step += 1.0

    def add_batch(self, out, data = None):

        self._add_batch(out['loss'])

    def get_print_result(self, msg = 'Average Loss : '):

        print(msg, self.total_loss / self.total_step + 1e-9)
        return self.total_loss / self.total_step + 1e-9     

class AccuracyEvaluator(object):
    
    def __init__(self):
        super(AccuracyEvaluator, self).__init__()
        self.reset()

    def reset(self):

        self.total_eqs = 0.0
        self.total     = 0.0

    def _add_batch(self, eqs):

        self.total_eqs += eqs.sum()
        self.total     += eqs.size(0)

    def add_batch(self, out, data):

        self._add_batch(out['predicts'].max(1)[1].eq(data['answer']))

    def get_print_result(self, msg = 'Accuracy : '):

        print(msg, '{:.2f}% ({:d} / {:d})'.format(100.0 * self.total_eqs / (self.total + 1e-9), int(self.total_eqs), int(self.total)))
        return self.total_eqs / (self.total + 1e-9)

class VQAV2ScoreEvaluator(object):
    
    def __init__(self):
        super(VQAV2ScoreEvaluator, self).__init__()
        self.reset()

    def reset(self):

        self.total_score = 0.0
        self.total       = 0.0

    def _add_batch(self, predicts, answer_vecs):

        batch_size = predicts.size(0)
        eq = (predicts[:, None].expand_as(answer_vecs) == answer_vecs).sum(1)
        t_scores = torch.min(eq.float() / 3.0, torch.FloatTensor(batch_size).fill_(1.0))

        self.total_score += t_scores.sum()
        self.total       += batch_size

    def add_batch(self, out, data):

        self._add_batch(out['predicts'].max(1)[1], data['raw_answer'])

    def get_print_result(self, msg = 'VQAv2-Score : '):

        print(msg, '{:.2f}'.format(100.0 * self.total_score / (self.total + 1e-9)))
        return self.total_score / (self.total + 1e-9)
        