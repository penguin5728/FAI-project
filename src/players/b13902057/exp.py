import random
import time


class Exp:
	"""Parametrized determinization Monte Carlo agent for experiments.

	Knobs (passed via config args):
	  depth: 1 (greedy tail only) or 2 (one optimized reply then tail)
	  objective: 'pen' (min our penalty) or 'rank' (min expected rank)
	  root_k, reply_k, budget: search width / time.
	"""

	def __init__(self, player_idx, depth=2, objective='pen',
	             root_k=6, reply_k=4, budget=0.78, seed=0):
		self.player_idx = player_idx
		self.depth = depth
		self.objective = objective
		self.ROOT_K = root_k
		self.REPLY_K = reply_k
		self.TIME_BUDGET = budget
		self.bull = [0] + [self._bullheads(c) for c in range(1, 105)]
		self.rng = random.Random(1000003 * (player_idx + 1) + seed)

	@staticmethod
	def _bullheads(card):
		if card % 55 == 0:
			return 7
		if card % 11 == 0:
			return 5
		if card % 10 == 0:
			return 3
		if card % 5 == 0:
			return 2
		return 1

	def _place(self, board, card, bull):
		best, best_last = -1, -1
		for i in range(len(board)):
			last = board[i][-1]
			if best_last < last < card:
				best_last = last
				best = i
		if best != -1:
			row = board[best]
			if len(row) >= 5:
				pen = 0
				for c in row:
					pen += bull[c]
				board[best] = [card]
				return pen
			row.append(card)
			return 0
		bi, bkey, bpen = 0, None, 0
		for i in range(len(board)):
			row = board[i]
			rv = 0
			for c in row:
				rv += bull[c]
			key = (rv, len(row), i)
			if bkey is None or key < bkey:
				bkey = key
				bi = i
				bpen = rv
		board[bi] = [card]
		return bpen

	def _eval(self, board, card, bull):
		best, best_last = -1, -1
		for i in range(len(board)):
			last = board[i][-1]
			if best_last < last < card:
				best_last = last
				best = i
		if best == -1:
			mn = None
			pen = 0
			for i in range(len(board)):
				row = board[i]
				rv = 0
				for c in row:
					rv += bull[c]
				k = (rv, len(row), i)
				if mn is None or k < mn:
					mn = k
					pen = rv
			return pen * 10.0
		row = board[best]
		if len(row) >= 5:
			pen = 0
			for c in row:
				pen += bull[c]
			return pen * 10.0
		new_len = len(row) + 1
		rv = bull[card]
		for c in row:
			rv += bull[c]
		return rv * (new_len / 5.0) ** 2 + (card - row[-1]) * 0.01

	def _greedy(self, board, hand, bull):
		best_card = hand[0]
		best = None
		for c in hand:
			s = self._eval(board, c, bull)
			if best is None or s < best:
				best = s
				best_card = c
		return best_card

	def _topk(self, hand, board, bull, k):
		if len(hand) <= k:
			return hand
		scored = [(self._eval(board, c, bull), c) for c in hand]
		scored.sort()
		return [c for _, c in scored[:k]]

	def _sim_round(self, board, my, opp, mc, bull):
		my.remove(mc)
		n_opp = len(opp)
		pens = [0] * (n_opp + 1)
		plays = [(mc, 0)]
		for k in range(n_opp):
			oc = self._greedy(board, opp[k], bull)
			opp[k].remove(oc)
			plays.append((oc, k + 1))
		plays.sort(key=lambda x: x[0])
		for card, who in plays:
			pens[who] += self._place(board, card, bull)
		return pens

	def _greedy_tail(self, board, my, opp, bull):
		n_opp = len(opp)
		pens = [0] * (n_opp + 1)
		for _ in range(len(my)):
			mc = self._greedy(board, my, bull)
			pens_round = self._sim_round(board, my, opp, mc, bull)
			for i in range(n_opp + 1):
				pens[i] += pens_round[i]
		return pens

	def _value(self, board0, my_hand, opp_hands, c, bull):
		n_opp = len(opp_hands)
		b = [r[:] for r in board0]
		my = list(my_hand)
		opp = [h[:] for h in opp_hands]
		p1 = self._sim_round(b, my, opp, c, bull)
		if not my:
			return p1
		if self.depth <= 1:
			pt = self._greedy_tail(b, my, opp, bull)
			return [p1[i] + pt[i] for i in range(n_opp + 1)]
		best_mine = None
		best_pens = None
		for c2 in self._topk(my, b, bull, self.REPLY_K):
			b2 = [r[:] for r in b]
			my2 = list(my)
			opp2 = [h[:] for h in opp]
			p2 = self._sim_round(b2, my2, opp2, c2, bull)
			if my2:
				pt = self._greedy_tail(b2, my2, opp2, bull)
				cont = [p2[i] + pt[i] for i in range(n_opp + 1)]
			else:
				cont = p2
			if best_mine is None or cont[0] < best_mine:
				best_mine = cont[0]
				best_pens = cont
		return [p1[i] + best_pens[i] for i in range(n_opp + 1)]

	@staticmethod
	def _unseen(hand, history):
		seen = set(hand)
		for row in history["board"]:
			seen.update(row)
		for cards in history.get("history_matrix", []):
			seen.update(cards)
		bh = history.get("board_history")
		if bh:
			for row in bh[0]:
				seen.update(row)
		return [c for c in range(1, 105) if c not in seen]

	def action(self, hand, history):
		if len(hand) == 1:
			return hand[0]
		t0 = time.perf_counter()
		board = history["board"]
		bull = self.bull
		scores = history["scores"]
		n_opp = max(0, len(scores) - 1)
		candidates = self._topk(hand, board, bull, self.ROOT_K)
		if n_opp == 0:
			return min(candidates, key=lambda c: (self._eval(board, c, bull), -c))

		my_score = scores[self.player_idx]
		opp_scores = [scores[i] for i in range(len(scores)) if i != self.player_idx]
		unseen = self._unseen(hand, history)
		R = len(hand)
		need = n_opp * R
		rank_tot = {c: 0.0 for c in candidates}
		rank_sq = {c: 0.0 for c in candidates}
		pen_tot = {c: 0.0 for c in candidates}
		counts = {c: 0 for c in candidates}
		deadline = t0 + self.TIME_BUDGET

		while time.perf_counter() < deadline:
			if len(unseen) >= need:
				deal = self.rng.sample(unseen, need)
			else:
				deal = unseen + self.rng.choices(unseen, k=need - len(unseen))
			opp_hands = [deal[i * R:(i + 1) * R] for i in range(n_opp)]
			for c in candidates:
				pens = self._value(board, hand, opp_hands, c, bull)
				mine = my_score + pens[0]
				rank = 1.0
				for k in range(n_opp):
					ops = opp_scores[k] + pens[k + 1]
					if ops < mine:
						rank += 1.0
					elif ops == mine:
						rank += 0.5
				rank_tot[c] += rank
				rank_sq[c] += rank * rank
				pen_tot[c] += pens[0]
				counts[c] += 1

		best_card = candidates[0]
		best_key = None
		for c in candidates:
			if counts[c] == 0:
				continue
			n = counts[c]
			mr = rank_tot[c] / n
			mp = pen_tot[c] / n
			if self.objective == 'rank':
				key = (mr, mp, -c)
			elif self.objective == 'robust':
				var = max(0.0, rank_sq[c] / n - mr * mr)
				key = (mr + 0.5 * var ** 0.5, mp, -c)
			else:
				key = (mp, mr, -c)
			if best_key is None or key < best_key:
				best_key = key
				best_card = c
		return best_card
