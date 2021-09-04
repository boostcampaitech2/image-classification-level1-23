import numpy as np
from numpy.lib.twodim_base import tril_indices
from sklearn.utils import validation
import torch
from torchvision.utils import make_grid
from base import BaseTrainer
from utils import inf_loop, MetricTracker, mixup_data, mixs_criterion, cutmix_data
from sklearn.model_selection import KFold , StratifiedKFold
import copy
from torch.utils.data import DataLoader

class Trainer(BaseTrainer):
    """
    Trainer class
    """
    def __init__(self, model, criterion, metric_ftns, optimizer, config, device,# data_loader, valid_data_loader=None, 
                data_set  = None, transform = None,
                 lr_scheduler=None, len_epoch=None, cut_mix = False, beta= 0.8, mix_up= False):
        super().__init__(model, criterion, metric_ftns, optimizer, config)
        self.config = config
        self.device = device
        # self.data_loader = data_loader
        # self.valid_data_loader = valid_data_loader
        # self.do_validation = self.valid_data_loader is not None
        self.do_validation = True
        self.data_set = data_set
        self.transform = transform
        self.cut_mix = cut_mix 
        self.beta = beta
        self.mix_up = mix_up
        if len_epoch is None:
        #     # epoch-based training
            self.len_epoch = 1000
        else:
            # iteration-based training
            self.data_loader = inf_loop(self.data_set)
            self.len_epoch = len_epoch
        self.lr_scheduler = lr_scheduler
        # self.log_step = int(np.sqrt(data_loader.batch_size)) * 4
        self.log_step = int(np.sqrt(64)) * 4

        self.train_metrics = MetricTracker('loss', *[m.__name__ for m in self.metric_ftns], writer=self.writer)
        self.valid_metrics = MetricTracker('loss', *[m.__name__ for m in self.metric_ftns], writer=self.writer)

    def _train_epoch(self, epoch):
        """
        Training logic for an epoch

        :param epoch: Integer, current training epoch.
        :return: A log that contains average loss and metric in this epoch.
        """
        self.model.train()
        self.train_metrics.reset()
        stratified_kfold = StratifiedKFold(n_splits=5, shuffle=True, random_state=None)
        labels = [value["label"] for key, value in self.data_set.data_dic.items()]
        k_idx = 0
        for train_index, validation_index in stratified_kfold.split(self.data_set, labels):
            k_idx+=1
            print(f'####### K-FOLD :: {k_idx}th')
            train_dataset = torch.utils.data.dataset.Subset(self.data_set,train_index)
            
            copied_dataset = copy.deepcopy(self.data_set)
            valid_dataset = torch.utils.data.dataset.Subset(copied_dataset,validation_index)
            
            train_dataset.dataset.set_transform(self.transform.transformations['train'])
            valid_dataset.dataset.set_transform(self.transform.transformations['val'])        
            self.data_loader = DataLoader(
                        train_dataset,
                        batch_size=50, 
                        num_workers=4,
                        shuffle=True)

            self.valid_data_loader = DataLoader(
                        valid_dataset,
                        batch_size=50,
                        num_workers=4,
                        shuffle=False) 
            target_count = 0
            for batch_idx, (data, target) in enumerate(self.data_loader):
                data, target = data.type(torch.FloatTensor).to(self.device), target.to(self.device)
                target_count+=len(target)
                self.optimizer.zero_grad()
                rand_num = np.random.random_integers(3) # 같이 넣어야 할까?
                if self.cut_mix and rand_num==1: # cutmix가 실행될 경우 
                    data, target_a, target_b, lam = cutmix_data(data, target, self.beta, self.device)
                    outputs = self.model(data)
                    loss= mixs_criterion(self.criterion, outputs, target_a, target_b, lam)
                elif self.mix_up and rand_num ==2:
                    data, target_a, target_b, lam = mixup_data(data, target)
                    outputs = self.model(data)
                    loss= mixs_criterion(self.criterion, outputs, target_a, target_b, lam)
                else:
                    outputs= self.model(data) 
                    loss= self.criterion(outputs, target) 

                # Backpropagation
                loss.backward()
                self.optimizer.step()

                self.writer.set_step((epoch - 1) * self.len_epoch + batch_idx)
                self.train_metrics.update('loss', loss.item())
                for met in self.metric_ftns:
                    self.train_metrics.update(met.__name__, met(outputs, target))

                if batch_idx % self.log_step == 0:
                    self.logger.debug('Train Epoch: {} {} Loss: {:.6f}'.format(
                        epoch,
                        self._progress(batch_idx),
                        loss.item()))
                    self.writer.add_image('input', make_grid(data.cpu(), nrow=8, normalize=True))

                if batch_idx == self.len_epoch:
                    break
            log = self.train_metrics.result()

            if self.do_validation:
                val_log = self._valid_epoch(epoch)
                log.update(**{'val_'+k : v for k, v in val_log.items()})

            if self.lr_scheduler is not None:
                self.lr_scheduler.step(loss)

        return log

    def _valid_epoch(self, epoch):
        """
        Validate after training an epoch

        :param epoch: Integer, current training epoch.
        :return: A log that contains information about validation
        """
        self.model.eval()
        self.valid_metrics.reset()
        target_count = 0
        with torch.no_grad():
            for batch_idx, (data, target) in enumerate(self.valid_data_loader):
                data, target = data.to(self.device), target.to(self.device)
                target_count += len(target)
                output = self.model(data)
                loss = self.criterion(output, target)

                self.writer.set_step((epoch - 1) * len(self.valid_data_loader) + batch_idx, 'valid')
                self.valid_metrics.update('loss', loss.item())
                for met in self.metric_ftns:
                    self.valid_metrics.update(met.__name__, met(output, target))
                self.writer.add_image('input', make_grid(data.cpu(), nrow=8, normalize=True))

        # add histogram of model parameters to the tensorboard
        for name, p in self.model.named_parameters():
            self.writer.add_histogram(name, p, bins='auto')
        return self.valid_metrics.result()

    def _progress(self, batch_idx):
        base = '[{}/{} ({:.0f}%)]'
        if hasattr(self.data_loader, 'n_samples'):
            current = batch_idx * self.data_loader.batch_size
            total = self.data_loader.n_samples
        else:
            current = batch_idx
            total = self.len_epoch
        return base.format(current, total, 100.0 * current / total)

# ['T', '__abs__', '__add__', '__and__', '__array__', '__array_priority__', '__array_wrap__', '__bool__', '__class__', '__complex__', '__contains__', '__deepcopy__', '__delattr__', '__delitem__', '__dict__', '__dir__', '__div__', '__doc__', '__eq__', '__float__', '__floordiv__', '__format__', '__ge__', '__getattribute__', '__getitem__', '__gt__', '__hash__', '__iadd__', '__iand__', '__idiv__', '__ifloordiv__', '__ilshift__', '__imod__', '__imul__', '__index__', '__init__', '__init_subclass__', '__int__', '__invert__', '__ior__', '__ipow__', '__irshift__', '__isub__', '__iter__', '__itruediv__', '__ixor__', '__le__', '__len__', '__long__', '__lshift__', '__lt__', '__matmul__', '__mod__', '__module__', '__mul__', '__ne__', '__neg__', '__new__', '__nonzero__', '__or__', '__pos__', '__pow__', '__radd__', '__rdiv__', '__reduce__', '__reduce_ex__', '__repr__', '__reversed__', '__rfloordiv__', '__rmul__', '__rpow__', '__rshift__', '__rsub__', '__rtruediv__', '__setattr__', '__setitem__', '__setstate__', '__sizeof__', '__str__', '__sub__', '__subclasshook__', '__torch_function__', '__truediv__', '__weakref__', '__xor__', '_backward_hooks', '_base', '_cdata', '_coalesced_', '_dimI', '_dimV', '_grad', '_grad_fn', '_indices', '_is_view', '_make_subclass', '_nnz', '_reduce_ex_internal', '_to_sparse_csr', '_update_names', '_values', '_version', 'abs', 'abs_', 'absolute', 'absolute_', 'acos', 'acos_', 'acosh', 'acosh_', 'add', 'add_', 'addbmm', 'addbmm_', 'addcdiv', 'addcdiv_', 'addcmul', 'addcmul_', 'addmm', 'addmm_', 'addmv', 'addmv_', 'addr', 'addr_', 'align_as', 'align_to', 'all', 'allclose', 'amax', 'amin', 'angle', 'any', 'apply_', 'arccos', 'arccos_', 'arccosh', 'arccosh_', 'arcsin', 'arcsin_', 'arcsinh', 'arcsinh_', 'arctan', 'arctan_', 'arctanh', 'arctanh_', 'argmax', 'argmin', 'argsort', 'as_strided', 'as_strided_', 'as_subclass', 'asin', 'asin_', 'asinh', 'asinh_', 'atan', 'atan2', 'atan2_', 'atan_', 'atanh', 'atanh_', 'backward', 'baddbmm', 'baddbmm_', 'bernoulli', 'bernoulli_', 'bfloat16', 'bincount', 'bitwise_and', 'bitwise_and_', 'bitwise_not', 'bitwise_not_', 'bitwise_or', 'bitwise_or_', 'bitwise_xor', 'bitwise_xor_', 'bmm', 'bool', 'broadcast_to', 'byte', 'cauchy_', 'cdouble', 'ceil', 'ceil_', 'cfloat', 'char', 'cholesky', 'cholesky_inverse', 'cholesky_solve', 'chunk', 'clamp', 'clamp_', 'clamp_max', 'clamp_max_', 'clamp_min', 'clamp_min_', 'clip', 'clip_', 'clone', 'coalesce', 'col_indices', 'conj', 'contiguous', 'copy_', 'copysign', 'copysign_', 'cos', 'cos_', 'cosh', 'cosh_', 'count_nonzero', 'cpu', 'cross', 'crow_indices', 'cuda', 'cummax', 'cummin', 'cumprod', 'cumprod_', 'cumsum', 'cumsum_', 'data', 'data_ptr', 'deg2rad', 'deg2rad_', 'dense_dim', 'dequantize', 'det', 'detach', 'detach_', 'device', 'diag', 'diag_embed', 'diagflat', 'diagonal', 'diff', 'digamma', 'digamma_', 'dim', 'dist', 'div', 'div_', 'divide', 'divide_', 'dot', 'double', 'dsplit', 'dtype', 'eig', 'element_size', 'eq', 'eq_', 'equal', 'erf', 'erf_', 'erfc', 'erfc_', 'erfinv', 'erfinv_', 'exp', 'exp2', 'exp2_', 'exp_', 'expand', 'expand_as', 'expm1', 'expm1_', 'exponential_', 'fill_', 'fill_diagonal_', 'fix', 'fix_', 'flatten', 'flip', 'fliplr', 'flipud', 'float', 'float_power', 'float_power_', 'floor', 'floor_', 'floor_divide', 'floor_divide_', 'fmax', 'fmin', 'fmod', 'fmod_', 'frac', 'frac_', 'frexp', 'gather', 'gcd', 'gcd_', 'ge', 'ge_', 'geometric_', 'geqrf', 'ger', 'get_device', 'grad', 'grad_fn', 'greater', 'greater_', 'greater_equal', 'greater_equal_', 'gt', 'gt_', 'half', 'hardshrink', 'has_names', 'heaviside', 'heaviside_', 'histc', 'hsplit', 'hypot', 'hypot_', 'i0', 'i0_', 'igamma', 'igamma_', 'igammac', 'igammac_', 'imag', 'index_add', 'index_add_', 'index_copy', 'index_copy_', 'index_fill', 'index_fill_', 'index_put', 'index_put_', 'index_select', 'indices', 'inner', 'int', 'int_repr', 'inverse', 'is_coalesced', 'is_complex', 'is_contiguous', 'is_cuda', 'is_distributed', 'is_floating_point', 'is_leaf', 'is_meta', 'is_mkldnn', 'is_mlc', 'is_nonzero', 'is_pinned', 'is_quantized', 'is_same_size', 'is_set_to', 'is_shared', 'is_signed', 'is_sparse', 'is_sparse_csr', 'is_vulkan', 'is_xpu', 'isclose', 'isfinite', 'isinf', 'isnan', 'isneginf', 'isposinf', 'isreal', 'istft', 'item', 'kron', 'kthvalue', 'layout', 'lcm', 'lcm_', 'ldexp', 'ldexp_', 'le', 'le_', 'lerp', 'lerp_', 'less', 'less_', 'less_equal', 'less_equal_', 'lgamma', 'lgamma_', 'log', 'log10', 'log10_', 'log1p', 'log1p_', 'log2', 'log2_', 'log_', 'log_normal_', 'log_softmax', 'logaddexp', 'logaddexp2', 'logcumsumexp', 'logdet', 'logical_and', 'logical_and_', 'logical_not', 'logical_not_', 'logical_or', 'logical_or_', 'logical_xor', 'logical_xor_', 'logit', 'logit_', 'logsumexp', 'long', 'lstsq', 'lt', 'lt_', 'lu', 'lu_solve', 'map2_', 'map_', 'masked_fill', 'masked_fill_', 'masked_scatter', 'masked_scatter_', 'masked_select', 'matmul', 'matrix_exp', 'matrix_power', 'max', 'maximum', 'mean', 'median', 'min', 'minimum', 'mm', 'mode', 'moveaxis', 'movedim', 'msort', 'mul', 'mul_', 'multinomial', 'multiply', 'multiply_', 'mv', 'mvlgamma', 'mvlgamma_', 'name', 'names', 'nan_to_num', 'nan_to_num_', 'nanmedian', 'nanquantile', 'nansum', 'narrow', 'narrow_copy', 'ndim', 'ndimension', 'ne', 'ne_', 'neg', 'neg_', 'negative', 'negative_', 'nelement', 'new', 'new_empty', 'new_empty_strided', 'new_full', 'new_ones', 'new_tensor', 'new_zeros', 'nextafter', 'nextafter_', 'nonzero', 'norm', 'normal_', 'not_equal', 'not_equal_', 'numel', 'numpy', 'orgqr', 'ormqr', 'outer', 'output_nr', 'permute', 'pin_memory', 'pinverse', 'polygamma', 'polygamma_', 'positive', 'pow', 'pow_', 'prelu', 'prod', 'put', 'put_', 'q_per_channel_axis', 'q_per_channel_scales', 'q_per_channel_zero_points', 'q_scale', 'q_zero_point', 'qr', 'qscheme', 'quantile', 'rad2deg', 'rad2deg_', 'random_', 'ravel', 'real', 'reciprocal', 'reciprocal_', 'record_stream', 'refine_names', 'register_hook', 'reinforce', 'relu', 'relu_', 'remainder', 'remainder_', 'rename', 'rename_', 'renorm', 'renorm_', 'repeat', 'repeat_interleave', 'requires_grad', 'requires_grad_', 'reshape', 'reshape_as', 'resize', 'resize_', 'resize_as', 'resize_as_', 'retain_grad', 'roll', 'rot90', 'round', 'round_', 'rsqrt', 'rsqrt_', 'scatter', 'scatter_', 'scatter_add', 'scatter_add_', 'select', 'set_', 'sgn', 'sgn_', 'shape', 'share_memory_', 'short', 'sigmoid', 'sigmoid_', 'sign', 'sign_', 'signbit', 'sin', 'sin_', 'sinc', 'sinc_', 'sinh', 'sinh_', 'size', 'slogdet', 'smm', 'softmax', 'solve', 'sort', 'sparse_dim', 'sparse_mask', 'sparse_resize_', 'sparse_resize_and_clear_', 'split', 'split_with_sizes', 'sqrt', 'sqrt_', 'square', 'square_', 'squeeze', 'squeeze_', 'sspaddmm', 'std', 'stft', 'storage', 'storage_offset', 'storage_type', 'stride', 'sub', 'sub_', 'subtract', 'subtract_', 'sum', 'sum_to_size', 'svd', 'swapaxes', 'swapaxes_', 'swapdims', 'swapdims_', 'symeig', 't', 't_', 'take', 'take_along_dim', 'tan', 'tan_', 'tanh', 'tanh_', 'tensor_split', 'tile', 'to', 'to_dense', 'to_mkldnn', 'to_sparse', 'tolist', 'topk', 'trace', 'transpose', 'transpose_', 'triangular_solve', 'tril', 'tril_', 'triu', 'triu_', 'true_divide', 'true_divide_', 'trunc', 'trunc_', 'type', 'type_as', 'unbind', 'unflatten', 'unfold', 'uniform_', 'unique', 'unique_consecutive', 'unsafe_chunk', 'unsafe_split', 'unsafe_split_with_sizes', 'unsqueeze', 'unsqueeze_', 'values', 'var', 'vdot', 'view', 'view_as', 'vsplit', 'where', 'xlogy', 'xlogy_', 'xpu', 'zero_']
# Parameter containing:
# tensor([2.], requires_grad=