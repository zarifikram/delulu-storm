import torch
import torch.nn as nn
import torch.nn.functional as F

@torch.no_grad()
def symlog(x):
    return torch.sign(x) * torch.log(1 + torch.abs(x))


@torch.no_grad()
def symexp(x):
    return torch.sign(x) * (torch.exp(torch.abs(x)) - 1)


class SymLogLoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, output, target):
        target = symlog(target)
        return 0.5*F.mse_loss(output, target)


class SymLogTwoHotLoss(nn.Module):
    def __init__(self, num_classes, lower_bound, upper_bound):
        super().__init__()
        self.num_classes = num_classes
        self.lower_bound = lower_bound
        self.upper_bound = upper_bound
        self.bin_length = (upper_bound - lower_bound) / (num_classes-1)

        # use register buffer so that bins move with .cuda() automatically
        self.bins: torch.Tensor
        self.register_buffer(
            'bins', torch.linspace(-20, 20, num_classes), persistent=False)

    def forward(self, output, target, rejection_mask=None):
        target = symlog(target)
        assert target.min() >= self.lower_bound and target.max() <= self.upper_bound

        index = torch.bucketize(target, self.bins)
        diff = target - self.bins[index-1]  # -1 to get the lower bound
        weight = diff / self.bin_length
        weight = torch.clamp(weight, 0, 1)
        weight = weight.unsqueeze(-1)

        target_prob = (1-weight)*F.one_hot(index-1, self.num_classes) + weight*F.one_hot(index, self.num_classes)

        loss = -target_prob * F.log_softmax(output, dim=-1)
        loss = loss.sum(dim=-1)
        if rejection_mask is not None:
            loss[rejection_mask.squeeze(-1)] = 0.0
        return (
            torch.sum(loss) / (loss != 0.0).sum()
            if (loss != 0.0).sum() > 0
            else loss.mean()
        )

    def decode(self, output):
        return symexp(F.softmax(output, dim=-1) @ self.bins)


class TemporalConsistancyLoss(torch.nn.Module):
    def __init__(self, config):
        super(TemporalConsistancyLoss, self).__init__()
        self.config = config
        self.latent_dim = config["latent_dim"]
        self.k = config["k"]
        self.tau = config["tau"]
        self.loss_type = config["loss_type"]
        self.delta = config["delta"]
        if self.loss_type == "predictive":
            self.predictor = torch.nn.Linear(self.latent_dim, 2*self.latent_dim) # mu and logvar
        self.mu, self.std = None, None

    def calculate_loss(self, features):
        # features should be of shape (batch_size, time, feature_dim)
        features = F.normalize(features, dim=-1)
        tc_loss = self._temporal_contrastive_loss(features) if self.loss_type == "contrastive" else self._temporal_predictive_loss(features)
        return tc_loss

    def _temporal_predictive_loss(self, x):
        # anchor and positive used loosely to allude to contrastive learning and not the actual meaning
        anchor = x[:, :-self.k].reshape(-1, self.latent_dim)  # Shape: (B*(T-k), D)
        positive = self._calculate_positive(x)  # Shape: (B*(T-k), k, D)
        predicted_positive = self._calculate_predicted_positive(anchor)  # Shape: (B*(T-k), k, D)
        return self._calculate_weighted_mse_loss(predicted_positive, positive)
    
    def _calculate_weighted_mse_loss(self, predicted_positive, positive):
        weights = torch.exp(-self.delta * torch.arange(self.k, device=predicted_positive.device))
        weights = weights / weights.sum()
        mse_loss = torch.nn.functional.mse_loss(predicted_positive, positive, reduction='none')
        weighted_mse_loss = (mse_loss * weights.unsqueeze(-1).unsqueeze(0)).sum(dim=1).mean(-1)
        self.mu, self.std = weighted_mse_loss.mean(), weighted_mse_loss.std()
        return weighted_mse_loss.mean()
    
    def _calculate_positive(self, x):
        B, T, D = x.shape
        positive_idx = self._calculate_next_k_indices(B, T, x.device)  # (B*(T-k), k)
        positive = x.reshape(-1, D)[positive_idx.reshape(-1)].reshape(-1, self.k, D)  # (B*(T-k), k, D)
        return positive

    def _calculate_predicted_positive(self, anchor):
        mu, logvar = self.predictor(anchor).chunk(2, dim=-1)
        mu = mu.unsqueeze(1).repeat(1, self.k, 1)  # Shape: (B*(T-k), k, D)
        logvar = logvar.unsqueeze(1).repeat(1, self.k, 1)
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std  # Shape: (B*(T-k), k, D)

    def _temporal_contrastive_loss(self, x):
        B, T, D = x.shape
        sim = self._calculate_cosine_similarity(x)  # (B*(T-k), B*T)
        
        postive_idx = self._calculate_next_k_indices(B, T, x.device)  # (B*(T-k), k)
        positive_similarities = sim.gather(dim=1, index=postive_idx)  # (B*(T-k), k)

        all_offset = torch.arange(0, B * T, device=x.device).view(B, T)
        all_idx = all_offset.unsqueeze(1).repeat(1, T - self.k, 1).reshape(-1, T)  # (B*(T-k), T)         
        all_similarities = sim.gather(dim=1, index=all_idx)  # (B*(T-k), T) # we care about the features _in_ the same sequence
                
        loss = torch.logsumexp(all_similarities, dim=1) - torch.logsumexp(positive_similarities, dim=1)
        self.mu, self.std = positive_similarities.mean(-1).mean(), positive_similarities.mean(-1).std()
        return loss.mean()
    
    def _calculate_cosine_similarity(self, x):
        B, T, D = x.shape
        anchors = x[:, :-self.k].reshape(-1, D)  # Shape: (B*(T-k), D)
        candidates = x.reshape(B * T, D) # Shape: (B*T, D)
        
        return torch.matmul(anchors, candidates.T) / self.tau  # Shape: (B*(T-k), B*T)
    
    def _calculate_next_k_indices(self, B, T, device):
        b_idx = torch.arange(B, device=device).unsqueeze(1).repeat(1, T - self.k)  # (B, T-k)
        t_idx = torch.arange(T - self.k, device=device).unsqueeze(0).repeat(B, 1)    # (B, T-k)
        
        # For each anchor, calculate positive indices in the flattened candidate space.
        # For an anchor at time t in batch b, its positive indices are: b * T + (t+1, ..., t+k).
        pos_offset = torch.arange(1, self.k + 1, device=device).view(1, 1, -1)  # (1, 1, k)
        pos_idx = b_idx.unsqueeze(-1) * T + (t_idx.unsqueeze(-1) + pos_offset)  # (B, T-k, k)
        pos_idx = pos_idx.reshape(-1, self.k)  # (B*(T-k), k)
        return pos_idx
    
    @torch.no_grad()
    def calculate_rejection_mask_and_distance_from_generated_outputs(
        self, context_feats, generated_feats
    ):
        # normalize the features
        s_t, s_t_plus_one = F.normalize(context_feats, dim=-1), F.normalize(generated_feats, dim=-1)
        if self.loss_type == "predictive":
            # calculate the distance
            rejection_mask, dist = self._calculate_rejection_mask_predictive(s_t, s_t_plus_one)
        elif self.loss_type == "contrastive":
            rejection_mask, dist = self._calculate_rejection_mask_contrastive(s_t, s_t_plus_one)
        return rejection_mask, dist

    def _calculate_rejection_mask_predictive(self, s_t, s_t_plus_one):
        deter_t, deter_t_plus_one = s_t[:, -self.latent_dim:], s_t_plus_one[:, -self.latent_dim:]
        # deter_t, deter_t_plus_one = s_t[:, :self.latent_dim], s_t_plus_one[:, :self.latent_dim]
        with torch.autocast(device_type='cuda', dtype=torch.bfloat16, enabled=True):
            predicted_deter_t_plus_one = self._calculate_predicted_positive(deter_t) # (B, k, D)
            prediction_error = (deter_t_plus_one.unsqueeze(1) - predicted_deter_t_plus_one).pow(2).mean(dim=[1, 2])
            prediction_error = (prediction_error - self.mu) / self.std
        return prediction_error > 1.645, prediction_error

    def _calculate_rejection_mask_contrastive(self, s_t, s_t_plus_one):
        # calculate the rejection mask based on the cosine similarity
        # both s_t and s_t plus one is (B, d). Calculate cosine similarity to get (B,)
        cosine_similarity = (torch.nn.functional.cosine_similarity(s_t, s_t_plus_one, dim=-1) / self.tau - self.mu) / self.std

        # reject ones with 5% of the distribution
        rejection_mask = cosine_similarity < -1.645
        return rejection_mask, cosine_similarity

