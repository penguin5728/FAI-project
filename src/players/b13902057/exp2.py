import random
import time


class Exp2:
	"""Sum-cached determinization MC for 6 Nimmt! experiments.

	Same algorithm family as exp.py, but the board is carried as rows + a
	parallel list of per-row bullhead sums, so the hot inner loops never
	re-sum a row. This is a pure constant-factor speedup -> more
	determinizations per time budget (same decisions, lower variance).

	modes:
	  'greedy' -> no MC, just the one-ply greedy heuristic pick.
	  'mc'     -> depth-1 determinization rollout (default).
	objective: 'rank' | 'robust' | 'pen'.
	"""

	def __init__(self, player_idx, mode='mc', objective='rank',
	             root_k=6, reply_k=4, depth=1, budget=0.80, hard_cap=0.95,
	             risk=0.5, seed=0):
		self.player_idx = player_idx
		self.mode = mode
		self.objective = objective
		self.ROOT_K = root_k
		self.REPLY_K = reply_k
		self.depth = depth
		self.TIME_BUDGET = budget
		self.HARD_CAP = hard_cap
		self.RISK = risk
		self.bull = [0] + [self._bullheads(c) for c in range(1, 105)]
		self.rng = random.Random(7654321 * (player_idx + 1) + seed)

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

	# --- sum-cached primitives -----------------------------------------
	def _eval(self, rows, sums, card, bull):
		best, best_last = -1, -1
		for i in range(len(rows)):
			last = rows[i][-1]
			if best_last < last < card:
				best_last = last
				best = i
		if best == -1:
			mn = None
			pen = 0
			for i in range(len(rows)):
				k = (sums[i], len(rows[i]), i)
				if mn is None or k < mn:
					mn = k
					pen = sums[i]
			return pen * 10.0
		if len(rows[best]) >= 5:
			return sums[best] * 10.0
		new_len = len(rows[best]) + 1
		rv = bull[card] + sums[best]
		return rv * (new_len / 5.0) ** 2 + (card - rows[best][-1]) * 0.01

	def _place(self, rows, sums, card, bull):
		best, best_last = -1, -1
		for i in range(len(rows)):
			last = rows[i][-1]
			if best_last < last < card:
				best_last = last
				best = i
		if best != -1:
			if len(rows[best]) >= 5:
				pen = sums[best]
				rows[best] = [card]
				sums[best] = bull[card]
				return pen
			rows[best].append(card)
			sums[best] += bull[card]
			return 0
		bi, bkey = 0, None
		for i in range(len(rows)):
			key = (sums[i], len(rows[i]), i)
			if bkey is None or key < bkey:
				bkey = key
				bi = i
		pen = sums[bi]
		rows[bi] = [card]
		sums[bi] = bull[card]
		return pen

	def _greedy(self, rows, sums, hand, bull):
		best_card = hand[0]
		best = None
		for c in hand:
			s = self._eval(rows, sums, c, bull)
			if best is None or s < best:
				best = s
				best_card = c
		return best_card

	def _topk(self, hand, rows, sums, bull, k):
		if len(hand) <= k:
			return hand
		scored = [(self._eval(rows, sums, c, bull), c) for c in hand]
		scored.sort()
		return [c for _, c in scored[:k]]

	def _sim_round(self, rows, sums, my, opp, mc, bull):
		my.remove(mc)
		n_opp = len(opp)
		pens = [0] * (n_opp + 1)
		plays = [(mc, 0)]
		for k in range(n_opp):
			oc = self._greedy(rows, sums, opp[k], bull)
			opp[k].remove(oc)
			plays.append((oc, k + 1))
		plays.sort()
		for card, who in plays:
			pens[who] += self._place(rows, sums, card, bull)
		return pens

	def _greedy_tail(self, rows, sums, my, opp, bull):
		n_opp = len(opp)
		pens = [0] * (n_opp + 1)
		for _ in range(len(my)):
			mc = self._greedy(rows, sums, my, bull)
			pr = self._sim_round(rows, sums, my, opp, mc, bull)
			for i in range(n_opp + 1):
				pens[i] += pr[i]
		return pens

	def _value(self, rows0, sums0, my_hand, opp_hands, c, bull):
		n_opp = len(opp_hands)
		rows = [r[:] for r in rows0]
		sums = sums0[:]
		my = list(my_hand)
		opp = [h[:] for h in opp_hands]
		p1 = self._sim_round(rows, sums, my, opp, c, bull)
		if not my:
			return p1
		if self.depth <= 1:
			pt = self._greedy_tail(rows, sums, my, opp, bull)
			return [p1[i] + pt[i] for i in range(n_opp + 1)]
		best_mine = None
		best_pens = None
		for c2 in self._topk(my, rows, sums, bull, self.REPLY_K):
			r2 = [r[:] for r in rows]
			s2 = sums[:]
			my2 = list(my)
			opp2 = [h[:] for h in opp]
			p2 = self._sim_round(r2, s2, my2, opp2, c2, bull)
			if my2:
				pt = self._greedy_tail(r2, s2, my2, opp2, bull)
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
		rows0 = board
		sums0 = [sum(bull[c] for c in row) for row in board]
		scores = history["scores"]
		n_opp = max(0, len(scores) - 1)

		candidates = self._topk(hand, rows0, sums0, bull, self.ROOT_K)

		if self.mode == 'greedy' or n_opp == 0:
			return min(candidates, key=lambda c: (self._eval(rows0, sums0, c, bull), -c))

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
		hard = t0 + self.HARD_CAP
		perf = time.perf_counter

		stop = False
		while not stop and perf() < deadline:
			if len(unseen) >= need:
				deal = self.rng.sample(unseen, need)
			else:
				deal = unseen + self.rng.choices(unseen, k=need - len(unseen))
			opp_hands = [deal[i * R:(i + 1) * R] for i in range(n_opp)]
			for c in candidates:
				if perf() >= hard:
					stop = True
					break
				pens = self._value(rows0, sums0, hand, opp_hands, c, bull)
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
				var = rank_sq[c] / n - mr * mr
				if var < 0.0:
					var = 0.0
				key = (mr + self.RISK * var ** 0.5, mp, -c)
			else:
				key = (mp, mr, -c)
			if best_key is None or key < best_key:
				best_key = key
				best_card = c
		return best_card
