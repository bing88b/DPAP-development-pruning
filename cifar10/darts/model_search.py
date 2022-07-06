import torch
import torch.nn as nn
import torch.nn.functional as F
from operations import *
from torch.autograd import Variable
from genotypes import PRIMITIVES
from genotypes import Genotype
from utils import parse

from SpikingNN.base.layers import VotingLayer
from SpikingNN.base.nodes import *
from SpikingNN.model_zoo.base_module import BaseModule


class darts_fun(torch.autograd.Function):
    @staticmethod
    def forward(ctx, inputs, weights):
        output = inputs * weights
        ctx.save_for_backward(inputs, weights, output)
        return output

    @staticmethod
    def backward(ctx, grad_output):
        grad_inputs, grad_weights = None, None

        inputs, weights, output = ctx.saved_tensors

        if ctx.needs_input_grad[0]:
            grad_inputs = grad_output * weights
        if ctx.needs_input_grad[1]:
            # grad_weights = inputs.sum()
            x = torch.where(output > 0., 1, 0) * inputs
            grad_weights = (x).sum()

        return grad_inputs, -grad_weights


class MixedOp(nn.Module):
    def __init__(self, C, stride, act_fun):
        super(MixedOp, self).__init__()
        self._ops = nn.ModuleList()
        for primitive in PRIMITIVES:
            op = OPS[primitive](C, stride, False, act_fun)
            if 'pool' in primitive:
                op = nn.Sequential(op, nn.BatchNorm2d(C, affine=False))
            self._ops.append(op)

        self.multiply = darts_fun.apply

    def forward(self, x, weights):
        # return sum(w * op(x) for w, op in zip(weights, self._ops))
        return sum(self.multiply(op(x), w) for w, op in zip(weights, self._ops))


class Cell(nn.Module):

    def __init__(self, steps, multiplier, C_prev_prev, C_prev, C, reduction, reduction_prev, act_fun):
        super(Cell, self).__init__()
        self.reduction = reduction

        if reduction_prev:
            self.preprocess0 = FactorizedReduce(C_prev_prev, C, affine=False, act_fun=act_fun)
        else:
            self.preprocess0 = ReLUConvBN(C_prev_prev, C, 1, 1, 0, affine=False, act_fun=act_fun)
        self.preprocess1 = ReLUConvBN(C_prev, C, 1, 1, 0, affine=False, act_fun=act_fun)
        self._steps = steps
        self._multiplier = multiplier

        self._ops = nn.ModuleList()

        for i in range(self._steps):
            for j in range(2 + i):
                stride = 2 if reduction and j < 2 else 1
                op = MixedOp(C, stride, act_fun)
                self._ops.append(op)

        # self.node = LIFNode()

    def forward(self, s0, s1, weights):
        s0 = self.preprocess0(s0)
        s1 = self.preprocess1(s1)

        states = [s0, s1]
        offset = 0

        for i in range(self._steps):
            s = sum(self._ops[offset+j](h, weights[offset+j]) for j, h in enumerate(states))
            offset += len(states)
            states.append(s)

        outputs = torch.cat(states[-self._multiplier:], dim=1)
        return outputs


class Network(BaseModule):

    def __init__(self, C, num_classes, layers, criterion, steps=4, multiplier=4, stem_multiplier=3,
                 parse_method='darts', op_threshold=None, step=1, node_type='ReLUNode', **kwargs):

        super(Network, self).__init__(
            step=step,
            num_classes=num_classes,
            encode_type='direct',
            spike_output=True
        )

        self.act_fun = eval(node_type)

        self._C = C
        self._num_classes = num_classes
        self._layers = layers
        self._criterion = criterion
        self._steps = steps
        self._multiplier = multiplier
        self.parse_method = parse_method
        self.op_threshold = op_threshold

        C_curr = stem_multiplier * C
        self.stem = nn.Sequential(
            nn.Conv2d(3, C_curr, 3, padding=1, bias=False),
            nn.BatchNorm2d(C_curr),
            # nn.ReLU(),
            self.act_fun(),
            nn.Conv2d(C_curr, C_curr, 3, padding=1, bias=False),
            nn.BatchNorm2d(C_curr)
        )

        C_prev_prev, C_prev, C_curr = C_curr, C_curr, C
        self.cells = nn.ModuleList()
        reduction_prev = False
        for i in range(layers):
            if i in [layers//3, 2*layers//3]:
                C_curr *= 2
                reduction = True
            else:
                reduction = False
            cell = Cell(steps, multiplier, C_prev_prev, C_prev, C_curr, reduction, reduction_prev, self.act_fun)
            reduction_prev = reduction
            self.cells += [cell]

            C_prev_prev, C_prev = C_prev, multiplier * C_curr
        self.global_pooling = nn.Sequential(self.act_fun(), nn.AdaptiveAvgPool2d(1))
        self.classifier = nn.Sequential(nn.Linear(C_prev, 10 * num_classes), self.act_fun())
        # self.global_pooling = nn.AdaptiveAvgPool2d(1)
        # self.classifier = nn.Linear(C_prev, num_classes * 10)
        self.vote = VotingLayer(10)
        self._initialize_alphas()

    def new(self):
        model_new = Network(self._C, self._num_classes, self._layers, self._criterion).cuda()
        for x, y in zip(model_new.arch_parameters(), self.arch_parameters()):
            x.data.copy_(y.data)
        return model_new

    def forward(self, inputs):
        inputs = self.encoder(inputs)

        self.reset()
        if not self.training:
            self.fire_rate.clear()

        outputs = []
        for t in range(self.step):
            x = inputs[t]
            s0 = s1 = self.stem(x)
            for i, cell in enumerate(self.cells):
                if cell.reduction:
                    weights = F.softmax(self.alphas_reduce, dim=-1)
                else:
                    weights = F.softmax(self.alphas_normal, dim=-1)
                s0, s1 = s1, cell(s0, s1, weights)
            out = self.global_pooling(s1)
            out = self.classifier(out.view(out.size(0), -1))
            logits = self.vote(out)
            outputs.append(logits)
        return sum(outputs) / len(outputs)

    def _loss(self, input1, target1):
        logits = self(input1)
        return self._criterion(logits, target1)

    def _initialize_alphas(self):
        # k = 2 + 3 + 4 + 5 = 14
        k = sum(1 for i in range(self._steps) for n in range(2+i))
        num_ops = len(PRIMITIVES)

        self.alphas_normal = Variable(torch.zeros(k, num_ops).cuda(), requires_grad=True)
        self.alphas_reduce = Variable(torch.zeros(k, num_ops).cuda(), requires_grad=True)

        # init the history
        self.alphas_normal_history = {}
        self.alphas_reduce_history = {}
        mm = 0
        last_id = 1
        node_id = 0
        for i in range(k):
            for j in range(num_ops):
                self.alphas_normal_history['edge: {}, op: {}'.format((node_id, mm), PRIMITIVES[j])] = []
                self.alphas_reduce_history['edge: {}, op: {}'.format((node_id, mm), PRIMITIVES[j])] = []
            if mm == last_id:
                mm = 0
                last_id += 1
                node_id += 1
            else:
                mm += 1

    def arch_parameters(self):
        return [self.alphas_normal, self.alphas_reduce]

    def genotype(self):

        # alphas_normal
        gene_normal = parse(F.softmax(self.alphas_normal, dim=-1).data.cpu().numpy(), PRIMITIVES, self.op_threshold, self.parse_method, self._steps)
        gene_reduce = parse(F.softmax(self.alphas_reduce, dim=-1).data.cpu().numpy(), PRIMITIVES, self.op_threshold, self.parse_method, self._steps)

        concat = range(2 + self._steps - self._multiplier, self._steps+2)
        genotype = Genotype(
            normal=gene_normal, normal_concat=concat,
            reduce=gene_reduce, reduce_concat=concat
        )
        return genotype

    def states(self):
        return {
            'alphas_normal': self.alphas_normal,
            'alphas_reduce': self.alphas_reduce,
            'alphas_normal_history': self.alphas_normal_history,
            'alphas_reduce_history': self.alphas_reduce_history,
            'criterion': self._criterion
        }

    def restore(self, states):
        self.alphas_normal = states['alphas_normal']
        self. alphas_reduce = states['alphas_reduce']
        self.alphas_normal_history = states['alphas_normal_history']
        self.alphas_reduce_history = states['alphas_reduce_history']

    def update_history(self):

        mm = 0
        last_id = 1
        node_id = 0
        weights1 = F.softmax(self.alphas_normal, dim=-1).data.cpu().numpy()
        weights2 = F.softmax(self.alphas_reduce, dim=-1).data.cpu().numpy()

        k, num_ops = weights1.shape
        for i in range(k):
            for j in range(num_ops):
                self.alphas_normal_history['edge: {}, op: {}'.format((node_id, mm), PRIMITIVES[j])].append(float(weights1[i][j]))
                self.alphas_reduce_history['edge: {}, op: {}'.format((node_id, mm), PRIMITIVES[j])].append(float(weights2[i][j]))
            if mm == last_id:
                mm = 0
                last_id += 1
                node_id += 1
            else:
                mm += 1
