import torch
from torch import nn
from torch.nn import Parameter
import torch.nn.functional as F
import all_constants as ac


class MultiheadAttention(nn.Module):
    """
    MultiheadAttention module
    I learned a lot from https://github.com/pytorch/fairseq/blob/master/fairseq/modules/multihead_attention.py
    """
    def __init__(self, args):
        super(MultiheadAttention, self).__init__()
        self.embed_dim = args.embed_dim
        self.num_heads = args.num_heads
        self.dropout = args.att_dropout
        self.use_bias = args.use_bias

        if self.embed_dim % self.num_heads != 0:
            raise ValueError("Required: embed_dim % num_heads == 0")

        self.head_dim = self.embed_dim // self.num_heads
        self.scale = self.head_dim ** -0.5

        # Parameters for linear projections of queries, keys, values and output
        self.weights = Parameter(torch.Tensor(4 * self.embed_dim, self.embed_dim))
        if self.use_bias:
            self.biases = Parameter(torch.Tensor(4 * self.embed_dim))

        # initializing
        # If we do Xavier normal initialization, std = sqrt(2/(2D))
        # but it's too big and causes unstability in PostNorm
        # so we use the smaller std of feedforward module, i.e. sqrt(2/(5D))
        mean = 0
        std = (2 / (5 * self.embed_dim)) ** 0.5
        nn.init.normal_(self.weights, mean=mean, std=std)
        if self.use_bias:
            nn.init.constant_(self.biases, 0.)

    def forward(self, q, k, v, mask, do_proj_qkv=True):
        def _split_heads(tensor):
            bsz, length, embed_dim = tensor.size()
            tensor = tensor.reshape(bsz, length, self.num_heads, self.head_dim).transpose(1, 2).reshape(bsz * self.num_heads, -1, self.head_dim)
            return tensor

        if do_proj_qkv:
            q, k, v = self.proj_qkv(q, k, v)

        q = _split_heads(q)
        k = _split_heads(k)
        v = _split_heads(v)

        att_weights = torch.bmm(q, k.transpose(1, 2)) * self.scale
        bsz_x_num_heads, src_len, tgt_len = att_weights.size()
        bsz = bsz_x_num_heads // self.num_heads
        att_weights = att_weights.reshape(bsz, self.num_heads, src_len, tgt_len)
        if mask is not None:
            att_weights.masked_fill_(mask, -1e9)

        att_weights = F.softmax(att_weights, dim=-1)
        att_weights = F.dropout(att_weights, p=self.dropout, training=self.training)
        _att_weights = att_weights.reshape(bsz_x_num_heads, src_len, tgt_len)
        output = torch.bmm(_att_weights, v)
        output = output.reshape(bsz, self.num_heads, src_len, self.head_dim).transpose(1, 2).reshape(bsz, src_len, -1)
        output = self.proj_o(output)

        return output, att_weights

    def proj_qkv(self, q, k, v):
        qkv_same = q.data_ptr() == k.data_ptr() == v.data_ptr()
        kv_same = k.data_ptr() == v.data_ptr()

        if qkv_same:
            q, k, v = self._proj(q, end=3 * self.embed_dim).chunk(3, dim=-1)
        elif kv_same:
            q = self._proj(q, end=self.embed_dim)
            k, v = self._proj(k, start=self.embed_dim, end=3 * self.embed_dim).chunk(2, dim=-1)
        else:
            q = self.proj_q(q)
            k = self.proj_k(k)
            v = self.proj_v(v)

        return q, k, v

    def _proj(self, x, start=0, end=None):
        weight = self.weights[start:end, :]
        bias = None if not self.use_bias else self.biases[start:end]
        return F.linear(x, weight=weight, bias=bias)

    def proj_q(self, q):
        return self._proj(q, end=self.embed_dim)

    def proj_k(self, k):
        return self._proj(k, start=self.embed_dim, end=2 * self.embed_dim)

    def proj_v(self, v):
        return self._proj(v, start=2 * self.embed_dim, end=3 * self.embed_dim)

    def proj_o(self, x):
        return self._proj(x, start=3 * self.embed_dim)


class FeedForward(nn.Module):
    """FeedForward"""
    def __init__(self, args):
        super(FeedForward, self).__init__()
        self.dropout = args.ff_dropout
        self.ff_dim = args.ff_dim
        self.embed_dim = args.embed_dim
        self.use_bias = args.use_bias

        self.in_proj = nn.Linear(self.embed_dim, self.ff_dim, bias=self.use_bias)
        self.out_proj = nn.Linear(self.ff_dim, self.embed_dim, bias=self.use_bias)

        # initializing
        mean = 0
        std = (2 / (self.ff_dim + self.embed_dim)) ** 0.5
        nn.init.normal_(self.in_proj.weight, mean=mean, std=std)
        nn.init.normal_(self.out_proj.weight, mean=mean, std=std)
        if self.use_bias:
            nn.init.constant_(self.in_proj.bias, 0.)
            nn.init.constant_(self.out_proj.bias, 0.)

    def forward(self, x):
        # my preliminary experiments show all RELU-variants
        # work the same and slower, RELU FTW!!!
        y = F.relu(self.in_proj(x))
        y = F.dropout(y, p=self.dropout, training=self.training)
        return self.out_proj(y)


class ScaleNorm(nn.Module):
    """ScaleNorm"""
    def __init__(self, scale, eps=1e-5):
        super(ScaleNorm, self).__init__()
        self.scale = Parameter(torch.tensor(scale))
        self.eps = eps

    def forward(self, x):
        norm = self.scale / torch.norm(x, dim=-1, keepdim=True).clamp(min=self.eps)
        return x * norm


class Encoder(nn.Module):
    """Self-attention Encoder"""
    def __init__(self, args):
        super(Encoder, self).__init__()
        self.dropout = args.dropout
        self.num_layers = args.num_enc_layers
        self.pre_act = args.pre_act

        self.atts = nn.ModuleList([MultiheadAttention(args) for _ in range(self.num_layers)])
        self.ffs = nn.ModuleList([FeedForward(args) for _ in range(self.num_layers)])

        num_scales = self.num_layers * 2 + 1 if self.pre_act else self.num_layers * 2
        if args.scnorm:
            self.scales = nn.ModuleList([ScaleNorm(args.embed_dim ** 0.5) for _ in range(num_scales)])
        else:
            self.scales = nn.ModuleList([nn.LayerNorm(args.embed_dim) for _ in range(num_scales)])

    def forward(self, src_inputs, src_mask):
        pre_act = self.pre_act
        post_act = not pre_act

        x = F.dropout(src_inputs, p=self.dropout, training=self.training)
        for i in range(self.num_layers):
            att = self.atts[i]
            ff = self.ffs[i]
            att_scale = self.scales[2 * i]
            ff_scale = self.scales[2 * i + 1]

            residual = x
            x = att_scale(x) if pre_act else x
            x, _ = att(q=x, k=x, v=x, mask=src_mask)
            x = residual + F.dropout(x, p=self.dropout, training=self.training)
            x = att_scale(x) if post_act else x

            residual = x
            x = ff_scale(x) if pre_act else x
            x = ff(x)
            x = residual + F.dropout(x, p=self.dropout, training=self.training)
            x = ff_scale(x) if post_act else x

        x = self.scales[-1](x) if pre_act else x
        return x


class Decoder(nn.Module):
    """Self-attention Decoder"""
    def __init__(self, args):
        super(Decoder, self).__init__()
        self.dropout = args.dropout
        self.num_layers = args.num_dec_layers
        self.pre_act = args.pre_act

        # sublayers
        self.atts = nn.ModuleList([MultiheadAttention(args) for _ in range(self.num_layers)])
        self.cross_atts = nn.ModuleList([MultiheadAttention(args) for _ in range(self.num_layers)])
        self.ffs = nn.ModuleList([FeedForward(args) for _ in range(self.num_layers)])

        num_scales = self.num_layers * 3 + 1 if self.pre_act else self.num_layers * 3
        if args.scnorm:
            self.scales = nn.ModuleList([ScaleNorm(args.embed_dim ** 0.5) for _ in range(num_scales)])
        else:
            self.scales = nn.ModuleList([nn.LayerNorm(args.embed_dim) for _ in range(num_scales)])

    def forward(self, tgt_inputs, tgt_mask, encoder_out, encoder_mask):
        pre_act = self.pre_act
        post_act = not pre_act

        x = F.dropout(tgt_inputs, p=self.dropout, training=self.training)
        for i in range(self.num_layers):
            att = self.atts[i]
            cross_att = self.cross_atts[i]
            ff = self.ffs[i]
            att_scale = self.scales[3 * i]
            cross_att_scale = self.scales[3 * i + 1]
            ff_scale = self.scales[3 * i + 2]

            residual = x
            x = att_scale(x) if pre_act else x
            x, _ = att(q=x, k=x, v=x, mask=tgt_mask)
            x = residual + F.dropout(x, p=self.dropout, training=self.training)
            x = att_scale(x) if post_act else x

            residual = x
            x = cross_att_scale(x) if pre_act else x
            x, _ = cross_att(q=x, k=encoder_out, v=encoder_out, mask=encoder_mask)
            x = residual + F.dropout(x, p=self.dropout, training=self.training)
            x = cross_att_scale(x) if post_act else x

            residual = x
            x = ff_scale(x) if pre_act else x
            x = ff(x)
            x = residual + F.dropout(x, p=self.dropout, training=self.training)
            x = ff_scale(x) if post_act else x

        x = self.scales[-1](x) if pre_act else x
        return x

    def beam_step(self, inp, cache):
        pre_act = self.pre_act
        post_act = not pre_act

        x = inp # [bsz x beam, 1, D]
        for i in range(self.num_layers):
            att = self.atts[i]
            cross_att = self.cross_atts[i]
            ff = self.ffs[i]
            att_scale = self.scales[3 * i]
            cross_att_scale = self.scales[3 * i + 1]
            ff_scale = self.scales[3 * i + 2]

            residual = x
            x = att_scale(x) if pre_act else x
            q, k, v = att.proj_qkv(x, x, x)
            k, v = cache.update_att(i, k, v)

            x, _ = att(q=q, k=k, v=v, mask=None, do_proj_qkv=False)
            x = residual + x
            x = att_scale(x) if post_act else x

            residual = x
            x = cross_att_scale(x) if pre_act else x
            q = cross_att.proj_q(x)
            k, v, mask = cache.retrieve_cross_att(i)
            x, _ = cross_att(q=q, k=k, v=v, mask=mask, do_proj_qkv=False)
            x = residual + x
            x = cross_att_scale(x) if post_act else x

            residual = x
            x = ff_scale(x) if pre_act else x
            x = ff(x)
            x = residual + x
            x = ff_scale(x) if post_act else x

        x = self.scales[-1](x) if pre_act else x

        return x

    # Cache switches back between two different shapes:
    #   [batch_size * beam_size, ....]
    #   [batch_size, beam_size, ....]
    # The cache maintains the correct shape by itself; the user should not need to think about this.
    class Cache():
        """Cache for storing intermediate decoder results"""
        # Creates cache in [batch_size, beam_size, ....] format
        def __init__(self, encoder_mask, encoder_out, decoder):
            self.decoder = decoder
            self.num_layers = decoder.num_layers

            self.in_batch_comma_beam_format = True
            self.batch_size = encoder_mask.size(0)
            self.beam_size = 1

            self.cache = {'encoder_mask': encoder_mask.unsqueeze_(1)}
            for i in range(self.num_layers):
                self.cache[i] = {'att': {'k': None, 'v': None}}
                self.cache[i]['cross_att_k'] = self.decoder.cross_atts[i].proj_k(encoder_out).unsqueeze_(1)
                self.cache[i]['cross_att_v'] = self.decoder.cross_atts[i].proj_v(encoder_out).unsqueeze_(1)

        # Switch format from [batch_size, beam_size, ....] to [batch_size * beam_size, ....]
        def batch_times_beam(self):
            self.in_batch_comma_beam_format = False
            batch_size = self.batch_size
            beam_size = self.beam_size

            self.cache['encoder_mask'] = self.cache['encoder_mask'].reshape(batch_size * beam_size, 1, 1, -1)
            for i in range(self.num_layers):
                seq_len = self.cache[i]['cross_att_k'].size(2)
                self.cache[i]['cross_att_k'] = self.cache[i]['cross_att_k'].reshape(batch_size * beam_size, seq_len, -1)
                self.cache[i]['cross_att_v'] = self.cache[i]['cross_att_v'].reshape(batch_size * beam_size, seq_len, -1)

                if self.cache[i]['att']['k'] is not None:
                    seq_len = self.cache[i]['att']['k'].size(2)
                    self.cache[i]['att']['k'] = self.cache[i]['att']['k'].reshape(batch_size * beam_size, seq_len, -1)
                    self.cache[i]['att']['v'] = self.cache[i]['att']['v'].reshape(batch_size * beam_size, seq_len, -1)

        # Switch format from [batch_size * beam_size, ....] to [batch_size, beam_size, ....]
        def batch_comma_beam(self):
            self.in_batch_comma_beam_format = True
            batch_size = self.batch_size
            beam_size = self.beam_size

            self.cache['encoder_mask'] = self.cache['encoder_mask'].reshape(batch_size, beam_size, 1, 1, -1)
            for i in range(self.num_layers):
                seq_len = self.cache[i]['cross_att_k'].size(1)
                self.cache[i]['cross_att_k'] = self.cache[i]['cross_att_k'].reshape(batch_size, beam_size, seq_len, -1)
                self.cache[i]['cross_att_v'] = self.cache[i]['cross_att_v'].reshape(batch_size, beam_size, seq_len, -1)

                seq_len = self.cache[i]['att']['k'].size(1)
                self.cache[i]['att']['k'] = self.cache[i]['att']['k'].reshape(batch_size, beam_size, seq_len, -1)
                self.cache[i]['att']['v'] = self.cache[i]['att']['v'].reshape(batch_size, beam_size, seq_len, -1)

        # Update self-attention + return result
        def update_att(self, i, k, v):
            if self.in_batch_comma_beam_format:
                self.batch_times_beam()

            if self.cache[i]['att']['k'] is not None:
                k = torch.cat((self.cache[i]['att']['k'], k), 1)
                v = torch.cat((self.cache[i]['att']['v'], v), 1)

            self.cache[i]['att']['k'] = k
            self.cache[i]['att']['v'] = v
            
            return k, v

        # Retrieve cached cross-attention
        def retrieve_cross_att(self, i):
            if self.in_batch_comma_beam_format:
                self.batch_times_beam()

            k = self.cache[i]['cross_att_k']
            v = self.cache[i]['cross_att_v']
            mask = self.cache['encoder_mask']
            return k, v, mask

        # Expand sample size from 1 to sample_size
        def expand_to_sample_size(self, sample_size):
            if not self.in_batch_comma_beam_format:
                self.batch_comma_beam()
            new_batch_size = self.batch_size * sample_size
            self.cache['encoder_mask'] = self.cache['encoder_mask'].expand(-1, sample_size, -1, -1, -1).reshape(new_batch_size, 1, 1, 1, -1)
            for i in range(self.num_layers):
                seq_len = self.cache[i]['cross_att_k'].size(2)
                self.cache[i]['cross_att_k'] = self.cache[i]['cross_att_k'].expand(-1, sample_size, -1, -1).reshape(new_batch_size, 1, seq_len, -1)
                self.cache[i]['cross_att_v'] = self.cache[i]['cross_att_v'].expand(-1, sample_size, -1, -1).reshape(new_batch_size, 1, seq_len, -1)

                if self.cache[i]['att']['k'] is not None:
                    seq_len = self.cache[i]['att']['k'].size(2)
                    self.cache[i]['att']['k'] = self.cache[i]['att']['k'].expand(-1, sample_size, -1, -1).reshape(new_batch_size, 1, seq_len, -1)
                    self.cache[i]['att']['v'] = self.cache[i]['att']['v'].expand(-1, sample_size, -1, -1).reshape(new_batch_size, 1, seq_len, -1)
            self.batch_size = new_batch_size

        # Expand beam size from 1 to beam_size
        def expand_to_beam_size(self, beam_size):
            if not self.in_batch_comma_beam_format:
                self.batch_comma_beam()
            self.cache['encoder_mask'] = self.cache['encoder_mask'].expand(-1, beam_size, -1, -1, -1)
            for i in range(self.num_layers):
                self.cache[i]['att']['k'] = self.cache[i]['att']['k'].expand(-1, beam_size, -1, -1)
                self.cache[i]['att']['v'] = self.cache[i]['att']['v'].expand(-1, beam_size, -1, -1)
                self.cache[i]['cross_att_k'] = self.cache[i]['cross_att_k'].expand(-1, beam_size, -1, -1)
                self.cache[i]['cross_att_v'] = self.cache[i]['cross_att_v'].expand(-1, beam_size, -1, -1)
            self.beam_size = beam_size

        # Remove sentences from the cache
        def trim_finished_sents(self, finished_sents):
            if not self.in_batch_comma_beam_format:
                self.batch_comma_beam()
            self.cache['encoder_mask'] = self.cache['encoder_mask'][~finished_sents]
            for i in range(self.num_layers):
                self.cache[i]['att']['k'] = self.cache[i]['att']['k'][~finished_sents]
                self.cache[i]['att']['v'] = self.cache[i]['att']['v'][~finished_sents]
                self.cache[i]['cross_att_k'] = self.cache[i]['cross_att_k'][~finished_sents]
                self.cache[i]['cross_att_v'] = self.cache[i]['cross_att_v'][~finished_sents]
            self.batch_size = self.cache['encoder_mask'].size(0)

        # This is used for topk selection during beam search
        def keep_beams(self, parent_idxs):
            if not self.in_batch_comma_beam_format:
                self.batch_comma_beam()
            batch_size = self.batch_size
            beam_size = self.beam_size

            for i in range(self.num_layers):
                seq_len = self.cache[i]['att']['k'].size(2)
                self.cache[i]['att']['k'] = self.cache[i]['att']['k'].reshape(batch_size * beam_size, seq_len, -1)[parent_idxs].reshape(batch_size, beam_size, seq_len, -1)
                self.cache[i]['att']['v'] = self.cache[i]['att']['v'].reshape(batch_size * beam_size, seq_len, -1)[parent_idxs].reshape(batch_size, beam_size, seq_len, -1)
