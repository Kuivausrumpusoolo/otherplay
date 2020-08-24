# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#
import torch
import torch.nn as nn
from torch.nn import functional as F
from typing import Tuple, Dict
import common_utils
import numpy as np

class R2D2Net(torch.jit.ScriptModule):
    __constants__ = ["hid_dim", "out_dim", "num_lstm_layer"]

    def __init__(self, device, in_dim, hid_dim, out_dim):
        super().__init__()
        self.in_dim = in_dim
        self.hid_dim = hid_dim
        self.out_dim = out_dim
        self.num_lstm_layer = 2

        self.net = nn.Sequential(nn.Linear(self.in_dim, self.hid_dim), nn.ReLU())

        self.lstm = nn.LSTM(
            self.hid_dim,
            self.hid_dim,
            num_layers=self.num_lstm_layer,  # , batch_first=True
        ).to(device)
        self.fc_v = nn.Linear(self.hid_dim, 1)
        self.fc_a = nn.Linear(self.hid_dim, self.out_dim)

        self.lstm.flatten_parameters()

    @torch.jit.script_method
    def get_h0(self, batchsize: int) -> Dict[str, torch.Tensor]:
        shape = (self.num_lstm_layer, batchsize, self.hid_dim)

        hid = {"h0": torch.zeros(*shape), "c0": torch.zeros(*shape)}
        return hid

    @torch.jit.script_method
    def duel(
        self, v: torch.Tensor, a: torch.Tensor, legal_move: torch.Tensor
    ) -> torch.Tensor:
        assert a.size() == legal_move.size()
        legal_a = a * legal_move
        q = v + legal_a - legal_a.mean(2, keepdim=True)
        return q

    @torch.jit.script_method
    def act(
        self, s: torch.Tensor, legal_move: torch.Tensor, hid: Dict[str, torch.Tensor]
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        assert s.dim() == 2, "should be 2 [batch, dim], get %d" % s.dim()
        s = s.unsqueeze(0)
        x = self.net(s)
        o, (h, c) = self.lstm(x, (hid["h0"], hid["c0"]))
        a = self.fc_a(o)
        a = a.squeeze(0)
        return a, {"h0": h, "c0": c}

    @torch.jit.script_method
    def forward(
        self,
        s: torch.Tensor,
        legal_move: torch.Tensor,
        action: torch.Tensor,
        hid: Dict[str, torch.Tensor],
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        assert s.dim() == 4, "[seq_len, batch, num_player, dim]"

        seq_len, batch, num_player, dim = s.size()
        s = s.view(seq_len, batch * num_player, dim)
        legal_move = legal_move.view(seq_len, batch * num_player, self.out_dim)
        action = action.view(seq_len, batch * num_player)

        x = self.net(s)
        if len(hid) == 0:
            o, (h, c) = self.lstm(x)
        else:
            o, (h, c) = self.lstm(x, (hid["h0"], hid["c0"]))
        a = self.fc_a(o)
        v = self.fc_v(o)
        q = self.duel(v, a, legal_move)

        # q: [seq_len, batch * num_player, num_action]
        # action: [seq_len, batch * num_player]
        qa = q.gather(2, action.unsqueeze(2)).squeeze(2)
        qa = qa.view(seq_len, batch, num_player)
        sum_q = qa.sum(2)

        assert q.size() == legal_move.size()
        legal_q = (1 + q - q.min()) * legal_move
        # greedy_action: [seq_len, batch * num_player]
        greedy_action = legal_q.argmax(2).detach()
        greedy_action = greedy_action.view(seq_len, batch, num_player)
        return sum_q, greedy_action


class R2D2Agent(torch.jit.ScriptModule):
    __constants__ = ["multi_step", "gamma", "eta"]

    def __init__(self, multi_step, gamma, eta, device, in_dim, hid_dim, out_dim):
        super().__init__()
        self.online_net = R2D2Net(device, in_dim, hid_dim, out_dim)
        self.target_net = R2D2Net(device, in_dim, hid_dim, out_dim)
        self.multi_step = multi_step
        self.gamma = gamma
        self.eta = eta

    @torch.jit.script_method
    def get_h0(self, batchsize: int) -> Dict[str, torch.Tensor]:
        return self.online_net.get_h0(batchsize)

    def clone(self, device):
        cloned = R2D2Agent(
            self.multi_step,
            self.gamma,
            self.eta,
            device,
            self.online_net.in_dim,
            self.online_net.hid_dim,
            self.online_net.out_dim,
        )
        cloned.load_state_dict(self.state_dict())
        return cloned.to(device)

    def sync_target_with_online(self):
        self.target_net.load_state_dict(self.online_net.state_dict())

    @torch.jit.script_method
    def greedy_act(
        self, s: torch.Tensor, legal_move: torch.Tensor, hid: Dict[str, torch.Tensor]
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        q, new_hid = self.online_net.act(s, legal_move, hid)
        legal_q = (1 + q - q.min()) * legal_move
        greedy_action = legal_q.argmax(1).detach()
        return greedy_action, new_hid

    @torch.jit.script_method
    def act(
        self, obs: Dict[str, torch.Tensor], hid: Dict[str, torch.Tensor],
    ) -> Tuple[Dict[str, torch.Tensor], Dict[str, torch.Tensor]]:
        """
        Acts on the given obs, with eps-greedy policy.
        output: {'a' : actions}, a long Tensor of shape [batchsize]
        """
        batchsize, num_player, dim = obs["s"].size()
        s = obs["s"].view(batchsize * num_player, dim)
        legal_move = obs["legal_move"].view(batchsize * num_player, -1)
        eps = obs["eps"].view(batchsize * num_player)

        greedy_action, new_hid = self.greedy_act(s, legal_move, hid)
        random_action = legal_move.multinomial(1).squeeze(1)
        rand = torch.rand(greedy_action.size(0), device=greedy_action.device)
        assert rand.size() == eps.size()
        rand = (rand < eps).long()
        action = (greedy_action * (1 - rand) + random_action * rand).long()

        action = action.view(batchsize, num_player)
        greedy_action = greedy_action.view(batchsize, num_player)
        return (
            {"a": action.cpu().detach(), "greedy_a": greedy_action.cpu().detach()},
            new_hid,
        )

    @torch.jit.script_method
    def compute_priority(
        self,
        obs: Dict[str, torch.Tensor],
        action: Dict[str, torch.Tensor],
        reward: torch.Tensor,
        terminal: torch.Tensor,  # todo remove this?
        bootstrap: torch.Tensor,
        next_obs: Dict[str, torch.Tensor],
        hid: Dict[str, torch.Tensor],
        next_hid: Dict[str, torch.Tensor],
    ) -> torch.Tensor:
        """
        compute priority for one batch
        """
        batchsize, num_player, dim = obs["s"].size()
        s = obs["s"].unsqueeze(0)
        legal_move = obs["legal_move"].unsqueeze(0)
        assert action["a"].dim() == 2
        a = action["a"].unsqueeze(0)
        online_q = self.online_net(s, legal_move, a, hid)[0].squeeze(0)

        # computing next_q with double q-learning
        next_s = next_obs["s"]
        next_legal_move = next_obs["legal_move"]
        online_next_a, _ = self.greedy_act(
            next_s.view(batchsize * num_player, dim),
            next_legal_move.view(batchsize * num_player, -1),
            next_hid,
        )
        online_next_a = online_next_a.view(1, batchsize, num_player)
        bootstrap_q = self.target_net(
            next_s.unsqueeze(0), next_legal_move.unsqueeze(0), online_next_a, next_hid
        )[0].squeeze(0)

        assert reward.size() == bootstrap.size()
        assert reward.size() == bootstrap_q.size()
        target = (
            reward + bootstrap.float() * (self.gamma ** self.multi_step) * bootstrap_q
        )
        priority = (target - online_q).abs()
        return priority.cpu().detach()

    @torch.jit.script_method
    def aggregate_priority(
        self, priority: torch.Tensor, seq_len: torch.Tensor
    ) -> torch.Tensor:
        """
        Given priority, compute the aggregated priority.
        Assumes priority is float Tensor of size [batchsize, seq_len]
        """
        mask = torch.arange(0, priority.size(0), device=seq_len.device)
        mask = (mask.unsqueeze(1) < seq_len.unsqueeze(0)).float()
        assert priority.size() == mask.size()
        priority = priority * mask

        p_mean = priority.sum(0) / seq_len
        p_max = priority.max(0)[0]
        agg_priority = self.eta * p_max + (1.0 - self.eta) * p_mean
        return agg_priority.cpu().detach()

    def _err(
        self,
        obs: Dict[str, torch.Tensor],
        hid: Dict[str, torch.Tensor],
        action: Dict[str, torch.Tensor],
        reward: torch.Tensor,
        terminal: torch.Tensor,
        bootstrap: torch.Tensor,
        seq_len: torch.Tensor,
    ) -> torch.Tensor:
        max_seq_len = obs["s"].size(0)
        s = obs["s"]
        legal_move = obs["legal_move"]

        action = action["a"]

        # flat_hid = {}
        # for key, val in hid.items():
        #     assert val.sum() == 0, val.sum()
        #     hd0, hd1, hd2, hd3 = val.size()
        #     assert hd1 == batchsize and hd2 == num_player
        #     flat_hid[key] = val.view(hd0, hd1 * hd2, hd3)

        # hid = flat_hid

        # hid is None for simplification
        hid = {}

        # this only works because the trajectories are padded,
        # i.e. no terminal in the middle
        online_qas, target_as = self.online_net(s, legal_move, action, hid)
        # sum_q, greedy_action
        with torch.no_grad():
            target_qas, _ = self.target_net(s, legal_move, target_as, hid)

        terminal = terminal.float()
        bootstrap = bootstrap.float()

        errs = []
        for i in range(max_seq_len):
            target_i = i + self.multi_step
            target_qa = 0
            if target_i < max_seq_len:
                target_qa = target_qas[target_i]
            bootstrap_qa = (self.gamma ** self.multi_step) * target_qa
            target = reward[i] + bootstrap[i] * bootstrap_qa

            # sanity check
            should_padding = i >= seq_len
            if i > 0:
                is_padding = (terminal[i] + terminal[i - 1] == 2).float()
                assert (is_padding.long() == should_padding.long()).all()

            err = (target.detach() - online_qas[i]) * (1 - should_padding.float())
            errs.append(err)
        return torch.stack(errs, 0)

    def loss(self, batch, args):
        """
        Input:
            obs = {
              eps: of shape (max_len, batch_size, num_players)  # not used.
                the epsilon used in the epsilon-greedy strategy of the acting agent?
              legal_move: of shape (max_len, batch_size, num_players, num_legal_moves)
              s: of shape (max_len, batch_size, num_players, dim)
                dim is the size of an observation
            }
            hid: empty dictionary,
              the hidden state of the LSTM seems to be initialized during the burn-in period
            action of shape (max_len, batch_size, num_players)
            reward of shape (max_len, batch_size)
            terminal of shape (max_len, batch_size)
            bootstrap of shape (max_len, batch_size)
            seq_len of shape (batch_size,)

        legal_move, s, action need to be permuted
        """
        if args.other_play:
            num_color = 5
            perm = np.random.permutation(range(num_color))
            batch = make_permutation(batch, perm, args)

        err = self._err(
            batch.obs,
            batch.h0,
            batch.action,
            batch.reward,
            batch.terminal,
            batch.bootstrap,
            batch.seq_len,
        )
        loss = nn.functional.smooth_l1_loss(
            err, torch.zeros_like(err), reduction="none"
        )
        # sum over seq dim
        loss = loss.sum(0)
        seq_len, batchsize, num_player, _ = batch.obs["s"].size()
        p = err.abs()
        priority = self.aggregate_priority(p, batch.seq_len)
        return loss, priority

def make_permutation(batch, perm, args):
    """
    Input:
        batch
        perm: np.array of shape (num_color,)
          eg. perm [2 3 0 4 1]: color 2 becomes color 0, color 3 becomes color 1 etc.
        args: dictionary of arguments given to main.py

    Returns
        batch with the s (observation), legal_move and action (of acting agent) permuted
        for the part that is fed into p0
    TODO
        add permutations for players n-1 for games with n players
    """
    with torch.no_grad():
        s = batch.obs["s"]  # (max_len, batch_size, num_players, dim)
        legal_move = batch.obs["legal_move"]  # (max_len, batch_size, num_player, num_legal_move)
        action = batch.action["a"]  # (max_len, batch_size, num_player)

        max_len, batch_size, num_player, dim = s.size()
        num_rank, hand_size = 5, 5
        num_color = len(perm)

        assert num_player == 2, "Error: too many players ({})".format(num_player)
        assert args.method == "vdn", "Error: IQL not supported"

        def permutate(tensor, start, color_major=False, step=num_rank):
            """
            Permutes elements at [:,...,:, start:start+num_color] (color_major=False)
              at [:,...,:, [start, start + num_rank, ..., start + num_color*num_rank]] (color_major=True)

            Assumes that the permutation happens in the last dimension,
            and that the dimension -2 corresponds to the player.
            """
            if color_major:
                target = step * torch.arange(num_color) + start
                source = step * perm + start
            else:
                target = torch.arange(num_color) + start
                source = perm + start

            assert len(tensor.shape) in [3, 4], \
                "Unexpected tensor of shape {}".format(tensor.shape)
            if len(tensor.shape) == 3:
                tensor[:, 0, target] = tensor[:, 0, source]
            else:
                tensor[:, :, 0, target] = tensor[:, :, 0, source]

        def permute_card(tensor, start):
            """
            Applies the permutation to data of format:
            R     Y     G     W     B
            ..... ..... ..... ..... .....

            This can be a single card eg.
            00000 00000 00000 00100 00000

            knowledge about the possible identities of a card eg.
            10111 10111 10111 00000 10111

            or the state of the fireworks on the board eg.
            11000 10000 11111 00000 11100
            """
            for color in range(num_color):
                permutate(tensor, start + color, color_major=True)
            return start + bits_per_card  # returns the new offset

        #################
        """LEGAL MOVES"""
        color_hint_start = 2 * hand_size  # discard moves + play moves
        for n in range(num_player - 1):
            start = color_hint_start + n * num_color
            permutate(legal_move, start)

        #################
        """OBSERVATION"""
        offset = 0

        # - HANDS
        #   - Cards of players 0,..,num_player (including empty bits for p0)
        #     where each card with a color-major rep (num_color * num_rank bits).
        #     Absent cards at the end of the game are encoded as zeros.
        #   - Bool player is missing card(s) at the end of the game
        #     (num_player bits)

        bits_per_card = num_color * num_rank

        for player_id in range(num_player):
            for card in range(hand_size):
                offset = permute_card(s, offset)

        offset += num_player  # missing card

        # - BOARD
        #   - remaining deck size
        #     (max_deck_size - num_player * hand_size bits; thermometer)
        #   - state of the fireworks
        #     (num_rank bits per color; one-hot)
        #   - information tokens remaining
        #     (max_information_tokens bits; thermometer)
        #   - life tokens remaining
        #     (max_life_tokens bits; thermometer)

        max_deck_size = 50
        offset += max_deck_size - num_player * hand_size  # remaining deck size

        offset = permute_card(s, offset)  # current state of the fireworks # TODO doesnt seem to be working

        max_information_tokens = 8
        offset += max_information_tokens

        max_life_tokens = 3
        offset += max_life_tokens

        # - DISCARDS
        # colour-major ordering, bits corresponding to ranks 1112233445
        # eg. 1100011101: discarded cards are of ranks 1, 1, 3, 3, 4, 5

        discard_bits_per_color = 10

        for start in range(discard_bits_per_color):
            permutate(s, offset + start,
                      color_major=True, step=discard_bits_per_color)
        offset += num_color * discard_bits_per_color

        # - LAST PLAYER ACTION (greedy / explorative)
        #   - acting player index relative to ourself (num_player bits; one-hot)
        #   - the move type (4 bits; one-hot)
        #   - target player index relative to acting player (num_player bits; one-hot)
        #   - color revealed, empty if not a hint (num_color bits; one-hot)
        #   - rank revealed (num_rank bits; one-hot)
        #   - reveal outcome (hand_size bits; one-hot)
        #     where each bit is 1 if the card was hinted at
        #   - position played/discarded (hand_size bits; one-hot)
        #   - card played/discarded (num_color * num_rank bits; one-hot)
        #   - was successful and/or added information token
        #     missing from comments!

        def permute_action(tensor, start):
            pos = start
            pos += num_player  # acting player
            pos += 4  # move type
            pos += num_player  # target player

            permutate(tensor, pos)  # color revealed
            pos += num_color

            pos += num_rank  # rank revealed
            pos += hand_size  # reveal outcome
            pos += hand_size  # position played/discarded

            pos = permute_card(tensor, pos)  # card played/discarded

            pos += 2  # was successful and/or added information token
            return pos

        offset = permute_action(s, offset)  # action played (explorative) by last player

        # - V0 Belief (assuming observation type includes V0)
        #   for each player: for each card:
        #   - identity probabilities: (num_color * num_rank bits;
        #     (bool: hint excludes?) * count / total)
        #   - color hints (num_color bits; one-hot)
        #   - rank hints (num_rank bits; one-hot)

        belief_bits_per_card = num_color * num_rank + num_color + num_rank
        for player_id in range(num_player):
            for card_id in range(hand_size):
                offset = permute_card(s, offset)  # belief

                permutate(s, offset)  # color hints
                offset += num_color

                offset += num_rank  # rank hints

        if args.greedy_extra:
            # last greedy action, seems to only be used for the last player instead of all others
            offset = permute_action(s, offset)

        assert offset == s.shape[-1], "Observation dim ({}) not equal to offset ({})"\
            .format(s.shape[-1], offset)

        ############
        """ACTION"""
        # Tensor of integers:
        # - discard moves (hand_size)                   [0, hand_size - 1]
        # - play_moves (hand_size)                      [hand_size, 2 * hand_size - 1]
        # - reveal_color ((num_player-1) * num_color)   [2*hand_size, 2*hand_size + (num_player-1)*num_color - 1]
        # - reveal_rank ((num_player-1) * num_rank)     [2*hand_size + (num_player-1)*num_color,
        #                                                 2*hand_size + (num_player-1)*(num_color + num_rank - 1)]
        # - last index is used to indicate the "move" of the non-active player

        discard_moves = hand_size
        play_moves = hand_size
        reveal_color_moves = (num_player - 1) * num_color
        reveal_rank_moves = (num_player - 1) * num_rank

        num_moves = discard_moves + play_moves + reveal_color_moves + reveal_rank_moves
        action = F.one_hot(action, num_classes=num_moves + 1)

        start = discard_moves + play_moves
        for player_id in range(num_player - 1):
            permutate(action, start)  # reveal color of player_id
            start += num_color
            start += num_rank

        action = torch.argmax(action, -1)

        batch.obs["s"] = s
        batch.obs["legal_move"] = legal_move
        batch.action["a"] = action
        return batch


def unit_tests(batch, args):
    num_color = 5
    identity = np.arange(num_color)
    identity_permuted_batch = make_permutation(batch, identity, args)

    assert torch.equal(batch.obs["s"], identity_permuted_batch.obs["s"]), \
        "Identity permutation changed observation"
    assert torch.equal(batch.obs["legal_move"], identity_permuted_batch.obs["legal_move"]), \
        "Identity permutation changed legal moves"

    perm = np.random.permutation(range(num_color))
    permuted_batch = make_permutation(batch, perm, args)
    assert torch.sum(batch.obs["s"]) == torch.sum(permuted_batch.obs["s"]),\
        "Permutation changed sum"

    repermuted_batch = make_permutation(permuted_batch, perm, args)
    assert torch.equal(batch.obs["s"], repermuted_batch.obs["s"]), \
        "Permuting twice doesn't produce original batch observation"


def print_(batch, game_id, transition_id, args, player_id):
    s = batch.obs["s"]
    legal_move = batch.obs["legal_move"]
    action = batch.action["a"]

    max_len, batch_size, num_player, dim = s.size()
    num_rank, hand_size, num_color = 5, 5, 5

    bits_per_card = num_color * num_rank

    def get_bitstring(tensor, start, length, separator='', format_specifier="%d"):
        bits = tensor[transition_id, game_id, player_id, np.arange(length) + start]
        rounded = [format_specifier % i for i in bits.tolist()]
        return separator.join(map(str, rounded))

    def bitstring_with_spaces(tensor, start, length, spaces_every=5):
        res = ""
        pos = 0
        while pos < length:
            bitstring = get_bitstring(tensor, start + pos, spaces_every)
            res += bitstring + " "
            pos += spaces_every
        return res

    ################
    """OBSERVATION"""
    offset = 0
    print("TRANSITION {}".format(transition_id))
    print("Observation from the point of view of player {}".format(player_id))

    # - HANDS
    print("HANDS")
    for agent_id in range(num_player):
        print("Cards of player {}".format(agent_id))
        for card_id in range(hand_size):
            print(bitstring_with_spaces(s, offset, bits_per_card))
            offset += bits_per_card
        print("")
    print("")

    missing_cards = get_bitstring(s, offset, num_player)
    offset += num_player
    print("Missing cards: {}\n".format(missing_cards))

    # - BOARD
    print("BOARD")
    remaining_deck_bits = 50 - num_player * hand_size
    remaining_deck = get_bitstring(s, offset, remaining_deck_bits)
    offset += remaining_deck_bits
    print("Remaining deck: {}".format(remaining_deck))

    fireworks = bitstring_with_spaces(s, offset, bits_per_card)
    offset += bits_per_card
    print("State of the fireworks: {}".format(fireworks))

    information_tokens = get_bitstring(s, offset, 8)
    offset += 8
    print("Information tokens: {}".format(information_tokens))

    life_tokens = get_bitstring(s, offset, 3)
    offset += 3
    print("Life tokens: {}\n".format(life_tokens))

    # - DISCARDS
    print("DISCARDS")
    discard_bits_per_color = 10
    for color_id in range(num_color):
        discards = get_bitstring(s, offset, discard_bits_per_color)
        offset += discard_bits_per_color
        print("Discards for color {}: {}".format(color_id, discards))
    print("")

    def print_action(tensor, start):
        pos = start

        acting_player = get_bitstring(s, pos, num_player)
        pos += num_player
        print("Acting player index relative to player {}: {}".format(player_id, acting_player))

        move_type = get_bitstring(s, pos, 4)
        pos += 4
        print("Move type (play, discard, reveal colour, reveal rank): {}".format(move_type))

        target_player = get_bitstring(s, pos, num_player)
        pos += num_player
        print("Target player index relative to player {}: {}".format(player_id, target_player))

        color_revealed = get_bitstring(s, pos, num_color)
        pos += num_color
        print("Color revealed: {}".format(color_revealed))

        rank_revealed = get_bitstring(s, pos, num_rank)
        pos += num_rank
        print("Rank revealed: {}".format(rank_revealed))

        reveal_outcome = get_bitstring(s, pos, hand_size)
        pos += hand_size
        print("Affected cards in target player's hand: {}".format(reveal_outcome))

        pos_played = get_bitstring(s, pos, hand_size)
        pos += hand_size
        print("Position played/discarded: {}".format(pos_played))

        card_played = bitstring_with_spaces(s, pos, bits_per_card)
        pos += bits_per_card
        print("Card played/discarded: {}".format(card_played))

        success = get_bitstring(s, pos, 1)
        pos += 1
        print("Successful placement: {}".format(success))

        got_token = get_bitstring(s, pos, 1)
        pos += 1
        print("Added information token: {}\n".format(got_token))

        return pos

    # - LAST PLAYER ACTION
    print("LAST PLAYER EXPLORATIVE")
    offset = print_action(s, offset)

    def get_card_belief(tensor, start):
        res = ""
        pos = 0
        for _ in range(num_color):
            res += get_bitstring(s, start + pos, num_rank,
                                 format_specifier="%.2f", separator=' ')
            res += " -- "
            pos += num_rank
        return res

    # - V0 BELIEF
    print("V0 BELIEF")
    for player_id in range(num_player):
        print("V0 for player {}".format(player_id))
        for card_id in range(hand_size):
            print("Identity probabilities for player {} card {}".format(player_id, card_id))
            print(get_card_belief(s, offset))
            offset += bits_per_card

            color_hints = get_bitstring(s, offset, num_color)
            offset += num_color
            print("Color hints given: {}".format(color_hints))

            rank_hints = get_bitstring(s, offset, num_rank)
            offset += num_rank
            print("Rank hints given: {}".format(rank_hints))

        print("")
    print("")

    # GREEDY ACTION
    if args.greedy_extra:
        print("LAST GREEDY")
        offset = print_action(s, offset)

    ################
    """ACTION"""
    print(action[transition_id, game_id, :])
    #print(action[:, game_id, :])

    ################
    """LEGAL ACTION"""
