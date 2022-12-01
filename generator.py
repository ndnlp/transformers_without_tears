import torch
import all_constants as ac

class Generator():
    """
    Generator which implements various decoding algorithms
    """
    def __init__(self, args, model):
        self.args = args
        self.model = model
    
    #def sample(self):
    
    #def beam_search(self):
    
    #def MBR(self):
    
    #def cluster_search_exact(self):
    
    #def cluster_search_beam(self):
    
    # src: [batch_size, length]
    # logit_mask: ???
    # src_lang_idx, tgt_lang_idx, beam_size: int
    def generate(self, src, src_lang_idx, tgt_lang_idx, logit_mask, beam_size):

        if self.args.use_rel_max_len:
            max_lengths = torch.sum(src != ac.PAD_ID, dim=-1).type(src.type()) + self.args.rel_max_len
        else:
            max_lengths = torch.tensor([self.args.abs_max_len] * src.size(0)).type(src.type())
        max_possible_length = max_lengths.max().item()
            
        decoder_one_step_fn, cache = self.model.get_decoder_one_step_fn(src, src_lang_idx, tgt_lang_idx, logit_mask, max_possible_length)
        
        bos_id = ac.BOS_ID
        eos_id = ac.EOS_ID
        alpha = self.args.beam_alpha
        decode_method = self.args.decode_method
        allow_empty = self.args.allow_empty
        
        # below this point is stuff I copied from the old beam_decode function in layers.py
        # with very small modifications

        # first step, beam=1
        batch_size = src.size(0)
        tgt = torch.tensor([bos_id] * batch_size).reshape(batch_size, 1) # [bsz, 1]
        next_token_probs = decoder_one_step_fn(tgt, 0, cache) # [bsz, V]
        
        if decode_method == ac.BEAM_SEARCH:
            # by default, do not allow EOS in first position
            if not allow_empty:
                next_token_probs[:, eos_id] = float('-inf')
            chosen_probs, chosen_symbols = torch.topk(next_token_probs, beam_size, dim=-1) # ([bsz, beam], [bsz, beam])
        else:
            chosen_symbols = torch.multinomial(torch.exp(next_token_probs), beam_size, replacement=True)
            chosen_probs = torch.gather(next_token_probs, -1, chosen_symbols)

        cumulative_probs = chosen_probs.reshape(batch_size, beam_size, 1)
        cumulative_scores = cumulative_probs.clone()
        cumulative_symbols = chosen_symbols.reshape(batch_size, beam_size, 1)

        cache.expand_to_beam_size(beam_size)

        num_classes = next_token_probs.size(-1)
        device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        not_eos_mask = (torch.arange(num_classes, device=device).reshape(1, -1) != eos_id)
        ret = [None] * batch_size
        batch_idxs = torch.arange(batch_size)
        for time_step in range(1, max_possible_length + 1):
            # once all the beams/samples for a sentence are finished, can stop doing computations on it
            surpass_length = (max_lengths < time_step) + (time_step == max_possible_length)
            finished_decoded = torch.sum((cumulative_symbols[:, :, -1] == eos_id).type(max_lengths.type()), -1) == beam_size
            finished_sents = surpass_length + finished_decoded
            if finished_sents.any():
                for j in range(finished_sents.size(0)):
                    if finished_sents[j]:
                        ret[batch_idxs[j]] = {
                            'symbols': cumulative_symbols[j].clone(),
                            'probs': cumulative_probs[j].clone(),
                            'scores': cumulative_scores[j].clone()
                        }

                cumulative_symbols = cumulative_symbols[~finished_sents]
                cumulative_probs = cumulative_probs[~finished_sents]
                cumulative_scores = cumulative_scores[~finished_sents]
                max_lengths = max_lengths[~finished_sents]
                batch_idxs = batch_idxs[~finished_sents]
                cache.trim_finished_sents(finished_sents)

            if finished_sents.all():
                break

            bsz = cumulative_symbols.size(0)
            last_symbols = cumulative_symbols[:, :, -1]
            next_token_probs = decoder_one_step_fn(last_symbols, time_step, cache) # [bsz x beam, V]
            cumulative_probs = cumulative_probs.reshape(-1, 1) # [bsz x beam, 1]
            cumulative_scores = cumulative_scores.reshape(-1, 1) # [bsz x beam, 1]
            finished_mask = last_symbols.reshape(-1) == eos_id
            
            # beam search
            if decode_method == ac.BEAM_SEARCH:
                # compute the cumulative probabilities and scores for all beams under consideration
                beam_probs = next_token_probs.clone()
                if finished_mask.any():
                    beam_probs[finished_mask] = cumulative_probs[finished_mask].expand(-1, num_classes).masked_fill(not_eos_mask, float('-inf'))
                    beam_probs[~finished_mask] = cumulative_probs[~finished_mask] + next_token_probs[~finished_mask]
                else:
                    beam_probs = cumulative_probs + next_token_probs

                beam_scores = beam_probs.clone()
                length_penalty = 1.0 if alpha == -1 else (5.0 + time_step + 1.0) ** alpha / 6.0 ** alpha
                if finished_mask.any():
                    beam_scores[finished_mask] = cumulative_scores[finished_mask].expand(-1, num_classes).masked_fill(not_eos_mask, float('-inf'))
                    beam_scores[~finished_mask] = beam_probs[~finished_mask] / length_penalty
                else:
                    beam_scores = beam_probs / length_penalty

                # choose top k beams
                beam_probs = beam_probs.reshape(bsz, -1) # [bsz, beam x D]
                beam_scores = beam_scores.reshape(bsz, -1) # [bsz, beam x D]
                k_scores, idxs = torch.topk(beam_scores, beam_size, dim=-1) # ([bsz, beam], [bsz, beam])

                parent_idxs = torch.div(idxs, num_classes, rounding_mode='floor')
                symbols = (idxs - parent_idxs * num_classes).type(idxs.type()) # [bsz, beam]

                cumulative_probs = torch.gather(beam_probs, -1, idxs) # [bsz, beam]
                cumulative_scores = k_scores
                parent_idxs = parent_idxs + torch.arange(bsz).unsqueeze_(1).type(parent_idxs.type()) * beam_size
                parent_idxs = parent_idxs.reshape(-1)
                cumulative_symbols = cumulative_symbols.reshape(bsz * beam_size, -1)[parent_idxs].reshape(bsz, beam_size, -1)
                cumulative_symbols = torch.cat((cumulative_symbols, symbols.unsqueeze_(-1)), -1)
                cache.keep_beams(parent_idxs)

            # sampling
            else:
                # (currently, probs and scores are always the same during sampling)
                beam_probs = cumulative_probs + next_token_probs
                beam_scores = cumulative_scores + next_token_probs
                if finished_mask.any():
                    next_token_probs[finished_mask][:, :] = float('-inf')
                    next_token_probs[finished_mask][:, eos_id] = 0
                idxs = torch.multinomial(torch.exp(next_token_probs), 1) # [bsz x beam, 1]

                cumulative_probs = torch.gather(beam_probs, -1, idxs) # [bsz x beam, 1]
                cumulative_scores = torch.gather(beam_scores, -1, idxs) # [bsz x beam, 1]
                cumulative_probs = cumulative_probs.reshape(bsz, beam_size)
                cumulative_scores = cumulative_scores.reshape(bsz, beam_size)
                symbols = idxs.reshape(bsz, beam_size, 1)
                cumulative_symbols = torch.cat((cumulative_symbols, symbols), -1)

        # Some hypotheses might haven't reached EOS yet and are cut off by length limit
        # make sure they are returned
        if batch_idxs.size(0) > 0:
            for j in range(batch_idxs.size(0)):
                ret[batch_idxs[j]] = {
                    'symbols': cumulative_symbols[j].clone(),
                    'probs': cumulative_probs[j].clone(),
                    'scores': cumulative_scores[j].clone()
                }

        return ret
