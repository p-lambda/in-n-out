import torch
from torch import nn
from torch import autograd
from torch import optim
from innout.load_utils import initialize


class WeightedSelfTrainLoss(nn.Module):
    '''
    Self training loss with labeled and unlabeled examples
    '''
    custom_input = True

    def __init__(self, model, unlabeled_weight=0.5, reduction='mean',
                 eval=False, is_binary=False):
        '''
        Args:
            model: nn.Module
                provided by main
            unlabeled_weight: float
                weight on the unlabeled loss
            reduction: str
                mean or sum
            eval: bool
                evaluation or not
        '''
        super().__init__()
        if reduction not in {'mean', 'sum'}:
            raise ValueError(f"Reduction {reduction} not supported")
        self.reduction = reduction
        self.unlabeled_weight = unlabeled_weight
        self.eval = eval
        self.model = model
        if is_binary:
            self.label_loss = nn.BCEWithLogitsLoss(reduction=reduction)
        else:
            self.label_loss = nn.CrossEntropyLoss(reduction=reduction)

    def forward(self, batch, epoch, optimizer, device):
        self.model.zero_grad()
        inputs, targets, labeled = batch['data'], batch['target'], batch['domain_label']['labeled']
        inputs, targets = inputs.to(device), targets.to(device)
        labeled = labeled.bool()

        labeled_input, labeled_target = inputs[labeled], targets[labeled]
        unlabeled_input, unlabeled_target = inputs[~labeled], targets[~labeled]

        labeled_loss = 0.0
        unlabeled_loss = 0.0
        if labeled_input.shape[0] > 0:
            labeled_out = self.model(labeled_input)
            labeled_loss = self.label_loss(labeled_out, labeled_target)
        if unlabeled_input.shape[0] > 0:
            unlabeled_out = self.model(unlabeled_input)
            unlabeled_loss = self.label_loss(unlabeled_out, unlabeled_target)
        loss = (1 - self.unlabeled_weight) * labeled_loss + self.unlabeled_weight * unlabeled_loss

        if not self.eval:
            loss.backward()
            optimizer.step()

        batch_metrics = {'epoch': epoch, 'loss': loss.item(), 'total': len(inputs)}
        return batch_metrics

class MaskedPretrainLoss(nn.Module):
    '''
    Use in conjunction with subclasses of MultitaskModel
    '''
    custom_input = True
    def __init__(self, model, loss_names, reduction='mean', eval=False):
        '''
        Args:
            model: nn.Module
                provided by main
            loss_names: List[str]
                list of strings, currently support 'cross_entropy' and 'mse_loss'
                as the elements of the list
            reduction: str
                mean or sum
            eval: bool
                evaluation or not
        '''
        super().__init__()
        if reduction not in {'mean', 'sum'}:
            raise ValueError(f"Reduction {reduction} not supported")
        self.reduction = reduction
        self.loss_names = loss_names
        self.ce_loss = nn.CrossEntropyLoss(reduction=reduction)
        self.binary_ce_loss = nn.BCEWithLogitsLoss(reduction=reduction)
        self.eval = eval
        self.model = model

    def forward(self, batch, epoch, optimizer, device):
        self.model.zero_grad()
        inputs, targets, use_idxs = batch['data'], batch['target'], batch['domain_label']['use_idx']
        inputs, targets = inputs.to(device), [t.to(device) for t in targets]

        # assumes that the input is already appropriately masked according
        # to the corresponding use_idx.
        # assumes that targets is a list of targets, where we will look
        # at target[use_idx] for every example

        shared_output = self.model.shared_layers(inputs)

        losses = []
        for use_idx in torch.unique(use_idxs):
            task_layer = self.model.task_layers[use_idx]
            use_idx_mask = (use_idxs == use_idx)
            curr_out = task_layer(shared_output[use_idx_mask])
            curr_targets = targets[use_idx][use_idx_mask]
            if self.loss_names[use_idx] == 'cross_entropy':
                curr_loss = self.ce_loss(curr_out, curr_targets.long())
            elif self.loss_names[use_idx] == 'mse_loss':
                curr_loss = mse_loss(curr_out, curr_targets.float(),
                                     reduction=self.reduction)
            elif self.loss_names[use_idx] == 'binary_cross_entropy':
                curr_loss = self.binary_ce_loss(curr_out, curr_targets.long())
            else:
                raise ValueError(f"Loss name not supported: {self.loss_names[use_idx]}")

            losses.append(curr_loss)

        if self.reduction == 'mean':
            loss = torch.mean(torch.stack(losses))
        else:
            loss = torch.sum(torch.stack(losses))

        if not self.eval:
            loss.backward()
            optimizer.step()

        batch_metrics = {'epoch': epoch, 'loss': loss.item(), 'total': len(inputs)}
        return batch_metrics


def mse_loss(out, targets, reduction='mean'):
    losses = (out - targets)**2
    reduce_dims = tuple(list(range(1, len(targets.shape))))
    losses = torch.mean(losses, dim=reduce_dims)

    if reduction == 'mean':
        loss = losses.mean()
    elif reduction == 'sum':
        loss = losses.sum()
    return loss


class MSELoss(nn.Module):
    custom_input = True

    def __init__(self, model, reduction='mean', eval=False, weight_decay=0):
        super().__init__()
        if reduction not in {'mean', 'sum'}:
            raise ValueError(f"Reduction {reduction} not supported")
        self.reduction = reduction
        self.eval = eval
        self.model = model
        self.weight_decay = weight_decay

    def forward(self, batch, epoch, optimizer, device):
        self.model.zero_grad()
        inputs, targets = batch['data'], batch['target']
        inputs, targets = inputs.to(device), targets.to(device)
        out = self.model(inputs)
        loss = mse_loss(out, targets, reduction=self.reduction)
        if self.weight_decay > 0:
            loss += self.weight_decay * l2_weight_norm(self.model)

        if not self.eval:
            loss.backward()
            optimizer.step()
            

        batch_metrics = {'epoch': epoch, 'loss': loss.item(), 'total': len(inputs)}
        return batch_metrics


class BCEMultiTaskLoss(nn.Module):
    custom_input = True

    def __init__(self, model, reduction='mean', eval=False, weight_decay=0):
        super().__init__()
        if reduction not in {'mean', 'sum'}:
            raise ValueError(f"Reduction {reduction} not supported")
        self.reduction = reduction
        self.eval = eval
        self.model = model
        self.weight_decay = weight_decay
        self.loss = torch.nn.BCEWithLogitsLoss(reduction=reduction)

    def forward(self, batch, epoch, optimizer, device):
        self.model.zero_grad()
        inputs, targets = batch['data'], batch['target']
        inputs, targets = inputs.to(device), targets.to(device)
        logits = self.model(inputs)
        if len(logits.shape) == 2 and logits.shape[1] == 1:
            logits = logits.squeeze()
        loss = self.loss(logits, targets)
        if self.weight_decay > 0:
            loss += self.weight_decay * l2_weight_norm(self.model)
        if not self.eval:
            loss.backward()
            optimizer.step()
        batch_metrics = {'epoch': epoch, 'loss': loss.item(), 'total': len(inputs)}
        return batch_metrics


def l2_weight_norm(model):
    weight_norm = torch.tensor(0.)
    is_cuda = list(model.parameters())[0].is_cuda
    if is_cuda:
        weight_norm = weight_norm.cuda()
    for name, w in model.named_parameters():
        if 'bias' not in name:
            weight_norm += w.norm().pow(2)
    return weight_norm


def irm_loss(task_loss, model, preds, targets, domains, lamda, weight_decay):
    domain_losses = []
    domain_gradnorms = []
    for domain_idx in range(int(domains.max().item()) + 1):
        domain_mask = (domains == domain_idx)
        num_in_domain = torch.sum(domain_mask).item()
        if num_in_domain == 0:
            continue

        domain_preds = preds[domain_mask]
        domain_targets = targets[domain_mask]

        loss_1 = task_loss(domain_preds, domain_targets)

        grads = autograd.grad(loss_1, list(model.end_model.parameters()), create_graph=True)
        grad_norms = [torch.sum(grad**2) for grad in grads]
        grad_norm = torch.sum(torch.stack(grad_norms))

        domain_losses.append(loss_1)
        domain_gradnorms.append(grad_norm)

    weight_norm = l2_weight_norm(model)

    loss = torch.stack(domain_losses).mean()
    loss += lamda * torch.stack(domain_gradnorms).mean()
    loss += weight_decay * weight_norm

    if lamda > 1.0:
        loss /= lamda
    return loss


class IRMLoss(nn.Module):
    requires_domain = True

    def __init__(self, model, lamda, weight_decay, binary=False):
        super().__init__()
        self.model = model
        self.lamda = lamda
        self.weight_decay = weight_decay

        if binary:
            self.task_loss = nn.BCEWithLogitsLoss()
        else:
            self.task_loss = nn.CrossEntropyLoss()

    def forward(self, preds, targets, domains):
        return irm_loss(self.task_loss, self.model, preds, targets, domains, self.lamda, self.weight_decay)


class DROLoss(nn.Module):
    requires_domain = True

    def __init__(self, binary=False):
        super().__init__()

        if binary:
            self.task_loss = nn.BCEWithLogitsLoss()
        else:
            self.task_loss = nn.CrossEntropyLoss()

    def forward(self, preds, targets, domains):
        domain_losses = []
        for domain_idx in range(int(domains.max().item()) + 1):
            domain_mask = (domains == domain_idx)
            num_in_domain = torch.sum(domain_mask).item()
            if num_in_domain == 0:
                continue

            domain_preds = preds[domain_mask]
            domain_targets = targets[domain_mask]

            loss_1 = self.task_loss(domain_preds, domain_targets)
            domain_losses.append(loss_1)
        loss = torch.stack(domain_losses).max()
        return loss


class DANNLoss(nn.Module):
    custom_input = True

    def __init__(self, model, domain_loss_weight,
                 domain_optimizer_name='adam', binary=False):
        super().__init__()
        self.model = model
        self.domain_loss_weight = domain_loss_weight

        if domain_optimizer_name == 'adam':
            self.domain_optimizer = optim.Adam(
                self.model.domain_model.parameters(), lr=1e-3)
        elif domain_optimizer_name == 'momentum':
            self.domain_optimizer = optim.SGD(
                self.model.domain_model.parameters(), lr=1e-3, momentum=0.9)
        else:
            raise ValueError('domain optimizer name not supported')

        if binary:
            self.task_loss = nn.BCEWithLogitsLoss()
        else:
            self.task_loss = nn.CrossEntropyLoss()

    def forward(self, batch, epoch, optimizer, device):
        inputs, targets, domains = batch['data'], batch['target'], batch['domain_label']
        inputs, targets, domains = inputs.to(device), targets.to(device), domains.to(device)
        # two stages: optimize the domain classifier internally, then compute the loss
        preds, feats = self.model(inputs, with_feats=True)
        pred_domain = self.model.domain_model(feats)
        domain_loss = self.task_loss(pred_domain, domains)

        optimizer.zero_grad()
        self.domain_optimizer.zero_grad()
        loss = self.task_loss(preds, targets) - self.domain_loss_weight * domain_loss
        loss.backward()
        optimizer.step()

        self.domain_optimizer.zero_grad()
        # TODO retain graph above to avoid recomputation?
        pred_domain = self.model.domain_model(feats)
        domain_loss = self.task_loss(pred_domain, domains)
        domain_loss.backward()
        self.domain_optimizer.step()
        batch_metrics = {'epoch': epoch, 'loss': loss.item(), 'disc_loss': domain_loss.item(), 'total': len(inputs)}
        return batch_metrics


class IndptExpertsLoss(nn.Module):
    custom_input = True

    def __init__(self, model, pretrain_domain, domain_optimizer_name='adam', binary=False):
        super().__init__()
        self.model = model
        self.pretrain_domain = pretrain_domain

        if domain_optimizer_name == 'adam':
            self.domain_optimizer = optim.Adam(
                self.model.domain_model.parameters(), lr=1e-3)
        elif domain_optimizer_name == 'momentum':
            self.domain_optimizer = optim.SGD(
                self.model.domain_model.parameters(), lr=1e-3, momentum=0.9)
        else:
            raise ValueError('domain optimizer name not supported')

        if binary:
            self.task_loss = nn.BCEWithLogitsLoss()
        else:
            self.task_loss = nn.CrossEntropyLoss()

    def forward(self, batch, epoch, optimizer, device):
        inputs, targets, domains = batch['data'], batch['target'], batch['domain_label']
        inputs, targets, domains = inputs.to(device), targets.to(device), domains.to(device)

        if self.pretrain_domain:
            preds = self.model.domain_model(inputs)
            loss = nn.CrossEntropyLoss()(preds, domains)
            self.domain_optimizer.zero_grad()
            loss.backward()
            self.domain_optimizer.step()
            batch_metrics = {'epoch': epoch, 'loss': loss.item()}
        else:
            domain_losses = []
            preds = self.model(inputs, d=domains)
            for domain_idx in range(int(domains.max().item()) + 1):
                domain_mask = (domains == domain_idx)
                num_in_domain = torch.sum(domain_mask).item()
                if num_in_domain == 0:
                    continue

                domain_preds = preds[domain_mask]
                domain_targets = targets[domain_mask]
                domain_losses.append(self.task_loss(domain_preds, domain_targets))
            optimizer.zero_grad()
            loss = torch.stack(domain_losses).sum()
            loss.backward()
            optimizer.step()

            batch_metrics = {'epoch': epoch, 'loss': loss.item(), 'total': len(inputs)}
        return batch_metrics


class MultitaskLoss(nn.Module):
    def __init__(self, losses, weights=None, weight_decay=0.0, model=None):
        '''
        Defines a wrapper loss around an ordered list of individual losses
        for a multitask model.

        Parameters
        ----------
        losses : List[Dict[str, Union[str, Dict[str, Any]]]]
            Contains dictionaries that define the construction of the
            individual loss functions. Each dictionary should have a
            "classname" key mapping to a string like
            "torch.nn.CrossEntropyLoss". Each dict can also have an "args" key
            that maps to another dictionary with key/value pairs for
            initialization.

        weights : List[float, ...], default None
            Optional list to specify weights for each loss. By default, each
            loss will be unweighted.

        weight_decay : float
            L2 regularization strength
        '''
        super(MultitaskLoss, self).__init__()
        self.losses = nn.ModuleList()
        for loss_dict in losses:
            self.losses.append(initialize(loss_dict))

        if weights is None:
            weights = [1 for _ in range(len(losses))]  # Default unweighted.
        assert isinstance(weights, list) and len(weights) == len(losses)
        self.weights = weights

        self.weight_decay = weight_decay
        if weight_decay > 0.0:
            assert(model is not None)
        self.model = model

    def forward(self, outputs, targets):
        '''
        Sums the individual task losses together and returns the resulting
        loss.

        Parameters
        ----------
        outputs : List[torch.Tensor, ...]
            Tuple of outputs from the model, one for each task. Should have the
            same length as targets.
        targets : List[torch.Tensor, ...]
            Tuple of targets, one for each task. Should have the same length as
            outputs.

        Returns
        -------
        Summed loss over all the individual tasks.
        '''
        multitask_losses = [self.weights[i] * self.losses[i](outputs[i], targets[i])
                   for i in range(len(self.losses))]
        multitask_loss = sum(multitask_losses)
        if self.weight_decay > 0.0:
            weight_norm = l2_weight_norm(self.model)
        else:
            weight_norm = 0.0
        return multitask_loss + self.weight_decay * weight_norm


'''class MultitaskLoss(nn.Module):
    custom_input = True

    def __init__(self, model, domain_loss_weight, binary=False):
        super().__init__()
        self.model = model
        self.domain_loss_weight = domain_loss_weight

        if binary:
            self.task_loss = nn.BCEWithLogitsLoss()
        else:
            self.task_loss = nn.CrossEntropyLoss()

    def forward(self, batch, epoch, optimizer, device):
        inputs, targets, domains = batch['data'], batch['target'], batch['domain_label']
        inputs, targets, domains = inputs.to(device), targets.to(device), domains.to(device)
        preds, feats = self.model(inputs, with_feats=True)
        pred_domain = self.model.domain_model(feats)
        loss = self.task_loss(preds, targets)
        domain_loss = self.task_loss(pred_domain, domains)

        optimizer.zero_grad()
        loss = loss + self.domain_loss_weight * domain_loss
        loss.backward()
        optimizer.step()

        batch_metrics = {'epoch': epoch, 'loss': loss.item(), 'total': len(inputs)}
        return batch_metrics'''


class DomainLoss(nn.Module):
    custom_input = True

    def __init__(self, model, binary=False, reduction='mean'):
        super().__init__()
        self.model = model
        self.reduction = reduction

        if binary:
            self.task_loss = nn.BCEWithLogitsLoss(reduction=reduction)
        else:
            self.task_loss = nn.CrossEntropyLoss(reduction=reduction)

    def pgd_attack(self, X, y, epsilon=5.0, num_steps=10, step_size=0.9):

        def squared_l2_norm(x):
            flattened = x.view(x.shape[0], -1)
            return (flattened ** 2).sum(1)

        def l2_norm(x):
            return squared_l2_norm(x).sqrt()

        x_adv = X.detach() + 0.
        x_adv += 0.001 * torch.randn(X.shape).cuda().detach()
        for _ in range(num_steps):
            x_adv.requires_grad_()
            with torch.enable_grad():
                loss = nn.CrossEntropyLoss()(self.model(x_adv), y)

            grad = torch.autograd.grad(loss, [x_adv])[0]

            grad = grad / (l2_norm(grad)[:, None, None, None] + 1e-8)
            x_adv = x_adv.detach() + step_size * grad
            eta_x_adv = x_adv - X
            norm_eta = l2_norm(eta_x_adv)
            project_mask = (norm_eta > epsilon)
            eta_x_adv = eta_x_adv * epsilon / norm_eta[:, None, None, None]
            x_adv[project_mask] = X[project_mask] + eta_x_adv[project_mask]

            x_adv = torch.clamp(x_adv, 0.0, 1.0)
        return x_adv

    def forward(self, batch, epoch, optimizer, device):
        inputs, targets, domains = batch['data'], batch['target'], batch['domain_label']
        inputs, targets, domains = inputs.to(device), targets.to(device), domains.to(device)
        outputs = self.model(inputs)
        loss = self.task_loss(outputs, domains)
        # get an adv example
        pgd_inputs = self.pgd_attack(inputs, domains)
        pgd_outputs = self.model(pgd_inputs)
        pgd_preds = pgd_outputs.max(1, keepdim=True)[1]
        pgd_correct_mask = pgd_preds.eq(domains.view_as(pgd_preds))
        pgd_acc = pgd_correct_mask.sum().item() / inputs.shape[0]
        loss_2 = self.task_loss(pgd_outputs, domains)

        loss = 0.5 * loss + 0.5 * loss_2

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        batch_metrics = {'epoch': epoch, 'loss': loss.item(), 'adv_acc': pgd_acc, 'total': len(inputs)}
        return batch_metrics
