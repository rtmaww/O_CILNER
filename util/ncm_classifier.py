import torch
import torch.nn.functional as F
class NcmClassification():
    def __init__(self):
        super(NcmClassification, self).__init__()


    def ncm_classifier_dot(self, reps, support_reps, support_labels, exemplar_means):
        n_tags = torch.max(support_labels) + 1

        feature = reps.view(-1, reps.shape[-1])  # (batch_size, feature_size)
        for j in range(feature.size(0)):  # Normalize
            feature.data[j] = feature.data[j] / feature.data[j].norm()
        #feature = feature.unsqueeze(1)  # (batch_size, 1, feature_size)

        means = torch.stack([exemplar_means[cls] for cls in range(n_tags)])  # (n_classes, feature_size)
    
        dists = -torch.matmul(feature, means.T)  # (batch_size, n_classes)
        _, pred_label = dists.min(1)
        pred_label = pred_label.view(reps.shape[0], reps.shape[1])
        return pred_label

class NNClassification():
    def __init__(self):
        super(NNClassification, self).__init__()

    def nn_classifier_dot(self, reps, support_reps, support_tags):
        batch_size, sent_len, ndim = reps.shape
        # support_reps [bsz*squ_len*baz_num, ndim]
        scores = self._euclidean_metric_dot(reps.view(-1, ndim), support_reps, True)
        emissions = self.get_nn_emissions(scores, support_tags)
        tags = torch.argmax(emissions, 1)
        return tags.view(batch_size, sent_len)
    
    def nn_classifier_dot_score(self, reps, support_reps, support_tags):
        batch_size, sent_len, ndim = reps.shape
        # support_reps [bsz*squ_len*baz_num, ndim]
        scores = self._euclidean_metric_dot(reps.view(-1, ndim), support_reps, True)
        # tags = support_tags[torch.argmax(scores, 1)]
        emissions = self.get_nn_emissions(scores, support_tags)
        tags = torch.argmax(emissions, 1)
        scores_dists = self._euclidean_metric_dot_2(reps.view(-1, ndim), support_reps, True)
        dists = scores_dists.max(1)[0]
        return tags.view(batch_size, sent_len), dists.view(batch_size, sent_len, -1)

    def nn_classifier_dot_prototype(self, reps, support_reps, support_tags, exemplar_means):
        batch_size, sent_len, ndim = reps.shape

        n_tags = torch.max(support_tags) + 1
        feature = reps.view(-1, reps.shape[-1])  # (batch_size, ndim)
        for j in range(feature.size(0)):  # Normalize
            feature.data[j] = feature.data[j] / feature.data[j].norm()
        # feature = feature.unsqueeze(1)  # (batch_size, 1, feature_size)
        means = torch.stack([exemplar_means[cls] for cls in range(n_tags)])# (n_classes, ndim)
        dists = torch.matmul(feature, means.T)  # (batch_size, n_classes)
        # prediction tags and emissions
        dists[:, 0] = torch.zeros(1).to(reps.device)
        emissions, tags = dists.max(1)

        prototype_dists = []
        support_reps = support_reps.view(-1, support_reps.shape[-1])
        for j in range(support_reps.size(0)):  # Normalize
            support_reps.data[j] = support_reps.data[j] / support_reps.data[j].norm()
        support_reps = F.normalize(support_reps)
        for i in range(n_tags):
            support_reps_dists = torch.matmul(support_reps[support_tags==i], means[i].T)
            prototype_dists.append(support_reps_dists.min(-1)[0])
        prototype_dists = torch.stack(prototype_dists).to(reps.device)
        return tags.view(batch_size, sent_len), emissions.view(batch_size, sent_len, -1), prototype_dists


    def get_top_emissions(self, reps, reps_labels, top_k, largest):
        if top_k <= 0:
            return None
        device = (torch.device('cuda')
                  if reps.is_cuda
                  else torch.device('cpu'))
        batch_size, sent_len, ndim = reps.shape
        reps_labels = reps_labels.view(-1)
        scores = self._euclidean_metric_dot_2(reps.view(-1, ndim), reps.view(-1, ndim), True)
        # emissions = self.get_nn_emissions(scores, reps_tags)
        # print(scores.size())
        # mask diag and labels != 'O'
        if largest is True:
            scores = torch.where(reps_labels == 0, scores.double(), -100.)
            scores = torch.scatter(scores, 1,
                                   torch.arange(scores.shape[0]).view(-1, 1).to(device), -100.)
        else:

            scores = torch.where(reps_labels == 0, scores.double(), 100.)
            scores = torch.scatter(scores, 1,
                                   torch.arange(scores.shape[0]).view(-1, 1).to(device), 100.)
        top_emissions = torch.topk(scores, top_k, dim=1, largest=largest).indices
        return top_emissions.view(-1, top_k)
    def get_top_emissions_with_th(self, reps, reps_labels, th_dists):
        # if top_k <= 0:
        #     return None
        device = (torch.device('cuda')
                  if reps.is_cuda
                  else torch.device('cpu'))
        batch_size, sent_len, ndim = reps.shape
        reps_labels = reps_labels.view(-1)
        scores = self._euclidean_metric_dot_2(reps.view(-1, ndim), reps.view(-1, ndim), True)
        scores = torch.where(reps_labels == 0, scores.double(), -100.)
        scores = torch.scatter(scores, 1,
                               torch.arange(scores.shape[0]).view(-1, 1).to(device), -100.)

        top_emissions = scores > th_dists
        return top_emissions, scores

    def _euclidean_metric(self, a, b, normalize=False):
        if normalize:
            a = F.normalize(a)
            b = F.normalize(b)
        n = a.shape[0]
        m = b.shape[0]
        a = a.unsqueeze(1).expand(n, m, -1)
        b = b.unsqueeze(0).expand(n, m, -1)
        logits = -((a - b) ** 2).sum(dim=2)
        # logits = torch.matmul(a, b.T)
        # logits_max, _ = torch.max(logits, dim=-1, keepdim=True)
        # logits = logits-logits_max.detach()
        return logits
    def _euclidean_metric_dot(self, a, b, normalize=False):
        if normalize:
            a = F.normalize(a)
            b = F.normalize(b)
        logits = torch.matmul(a, b.T)
        logits_max, _ = torch.max(logits, dim=-1, keepdim=True)
        logits = logits-logits_max.detach()
        return logits
    def _euclidean_metric_dot_2(self, a, b, normalize=False):
        if normalize:
            a = F.normalize(a)
            b = F.normalize(b)
        logits = torch.matmul(a, b.T)
        return logits.detach()
    def get_nn_emissions(self, scores, tags):
        """
        Obtain emission scores from NNShot
        """
        n, m = scores.shape
        n_tags = torch.max(tags) + 1
        emissions = -100000. * torch.ones(n, n_tags).to(scores.device)
        for t in range(n_tags):
            mask = (tags == t).float().view(1, -1)
            masked = scores * mask
            # print(masked)
            masked = torch.where(masked < 0, masked, torch.tensor(-100000.).to(scores.device))
            emissions[:, t] = torch.max(masked, dim=1)[0]
        return emissions