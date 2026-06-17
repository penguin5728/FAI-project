import random
import time
from math import log, sqrt


class _Node:
	__slots__ = ('N', 'W', 'children')

	def __init__(self):
		self.N = 0
		self.W = 0.0
		self.children = {}


class player3:
	"""Determinized Information-Set MCTS agent for 6 Nimmt! (rank-min).

	The tournament scores by *rank*, and opponents' hands are hidden, so each
	search iteration:
	  1. determinizes the hidden opponent hands by dealing the unseen pool
	     (which also holds the never-dealt cards -- genuine, correct
	     uncertainty),
	  2. descends a tree keyed by *our own* card sequence using UCB1, with
	     opponents playing a fast greedy policy and every round resolved
	     exactly like the engine (sum-cached, never re-summed),
	  3. greedy-rolls out the remaining cards to game end,
	  4. backpropagates the normalized rank we achieved.
	The chosen move is the most-visited root child (robust), or best-mean if
	configured.

	Every knob is a constructor argument with a default, so the agent is
	directly hyperparameter-swept from the tournament config's ``args`` dict
	(the engine instantiates ``player3(player_idx=..., **args)``). Defaults
	form a sane standalone player; the sweep just overrides them.

	Knobs:
	  c        -- UCB exploration constant.
	  root_k   -- branching cap: only the top-k engine-greedy cards expanded.
	  budget   -- wall-clock seconds before stopping new iterations.
	  hard_cap -- absolute wall-clock guard against the 1.0s timeout.
	  reward   -- 'rank' (default): normalized seat rank, the tournament metric
	              -- empirically the strongest objective here. 'blend': convex
	              mix of rank and a normalized bullhead-penalty reward (a large
	              penalty weight regressed in testing, so blend defaults low and
	              is a sweep knob). 'pen': bullhead penalty only.
	  blend    -- weight on the penalty term in 'blend' (0 = pure rank,
	              1 = pure penalty).
	  pen_scale-- bullhead count mapped to reward 0 (penalties >= this clamp).
	  n_det    -- size of the common-random-number determinization pool
	              reused round-robin so sibling moves share opponent deals
	              (0 = fresh deal every iteration).
	  final    -- 'visit' (most-visited child) or 'mean' (best mean reward).
	  self_bull_w -- weight (0..1) on our own card's bullheads in the safe-fit
	              risk of the base heuristic `_eval`. 1.0 = original; lower
	              values stop over-penalizing safely offloading big cards.
	  iters    -- >0 forces a fixed iteration count (deterministic, for
	              reproducible tuning); 0 uses the wall-clock budget.
	  seed     -- RNG seed offset for determinization.
	"""

	def __init__(self, player_idx, c=0.6, root_k=10, budget=0.85,
	             hard_cap=0.90, reward='rank', blend=0.5, pen_scale=33.0,
	             n_det=64, final='visit', self_bull_w=1.0, iters=0, seed=0):
		self.player_idx = player_idx
		self.c = c
		self.root_k = root_k
		self.TIME_BUDGET = budget
		self.HARD_CAP = hard_cap
		self.reward = reward
		self.blend = blend
		self.pen_scale = pen_scale
		self.n_det = n_det
		self.final = final
		self.sbw = self_bull_w
		self.iters = iters
		self.bull = [0] + [self._bullheads(c) for c in range(1, 105)]
		self.rng = random.Random(2654435761 * (player_idx + 1) + seed)

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

	# --- sum-cached engine-faithful primitives -------------------------
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
		# Safe fit: immediate cost 0; estimate the row's future take-risk.
		# Our own card's bullheads are charged only at weight `sbw` (< 1): if we
		# never take this row those bullheads hurt whoever does, so penalizing a
		# safe offload of a big card at full weight is wrong. sbw=1.0 recovers
		# the original heuristic.
		rv = sums[best] + bull[card] * self.sbw
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
			return list(hand)
		scored = [(self._eval(rows, sums, c, bull), c) for c in hand]
		scored.sort()
		return [c for _, c in scored[:k]]

	# --- one round: mutate rows/sums/my/opp, return seat penalties ------
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

	# --- main -----------------------------------------------------------
	def action(self, hand, history):
		if len(hand) == 1:
			return hand[0]

		t0 = time.perf_counter()
		bull = self.bull
		rows0 = history["board"]
		sums0 = [sum(bull[c] for c in row) for row in rows0]
		scores = history["scores"]
		n_opp = max(0, len(scores) - 1)
		n_tot = n_opp + 1

		if n_opp == 0:
			cands = self._topk(hand, rows0, sums0, bull, self.root_k)
			return min(cands, key=lambda c: (self._eval(rows0, sums0, c, bull), -c))

		my_score = scores[self.player_idx]
		opp_scores = [scores[i] for i in range(len(scores)) if i != self.player_idx]
		unseen = self._unseen(hand, history)
		R = len(hand)
		need = n_opp * R

		root = _Node()
		c_uct = self.c
		root_k = self.root_k
		reward_mode = self.reward
		blend = self.blend
		pen_scale = self.pen_scale
		deadline = t0 + self.TIME_BUDGET
		hard = t0 + self.HARD_CAP
		perf = time.perf_counter
		rng = self.rng

		# Pre-sample a CRN pool of determinizations (reused round-robin so that
		# sibling root moves are compared on the same opponent deals).
		det_pool = None
		if self.n_det and self.n_det > 0:
			det_pool = []
			for _ in range(self.n_det):
				if len(unseen) >= need:
					d = rng.sample(unseen, need)
				else:
					d = unseen + rng.choices(unseen, k=need - len(unseen))
				det_pool.append([d[i * R:(i + 1) * R] for i in range(n_opp)])

		fixed_iters = self.iters
		it = 0
		while True:
			if fixed_iters > 0:
				if it >= fixed_iters:
					break
			elif perf() >= deadline:
				break
			# 1. determinize opponent hands (CRN pool if enabled)
			if det_pool is not None:
				opp = [h[:] for h in det_pool[it % self.n_det]]
			else:
				if len(unseen) >= need:
					deal = rng.sample(unseen, need)
				else:
					deal = unseen + rng.choices(unseen, k=need - len(unseen))
				opp = [deal[i * R:(i + 1) * R] for i in range(n_opp)]
			it += 1

			rows = [r[:] for r in rows0]
			sums = sums0[:]
			my = list(hand)
			my_pen = 0
			opp_pen = [0] * n_opp

			node = root
			path = [root]
			# 2. tree policy (selection + one expansion)
			while my:
				cands = self._topk(my, rows, sums, bull, root_k)
				children = node.children
				untried = None
				for c in cands:
					if c not in children:
						untried = c
						break
				if untried is not None:
					child = _Node()
					children[untried] = child
					pr = self._sim_round(rows, sums, my, opp, untried, bull)
					my_pen += pr[0]
					for k in range(n_opp):
						opp_pen[k] += pr[k + 1]
					node = child
					path.append(node)
					break
				# select best child by UCB among current candidates
				logN = log(node.N + 1.0)
				best_val = None
				best_c = cands[0]
				for c in cands:
					ch = children[c]
					val = ch.W / ch.N + c_uct * sqrt(logN / ch.N)
					if best_val is None or val > best_val:
						best_val = val
						best_c = c
				ch = children[best_c]
				pr = self._sim_round(rows, sums, my, opp, best_c, bull)
				my_pen += pr[0]
				for k in range(n_opp):
					opp_pen[k] += pr[k + 1]
				node = ch
				path.append(node)

			# 3. greedy rollout to game end
			if my:
				pt = self._greedy_tail(rows, sums, my, opp, bull)
				my_pen += pt[0]
				for k in range(n_opp):
					opp_pen[k] += pt[k + 1]

			# 4. reward + backprop
			mine = my_score + my_pen
			# normalized penalty reward in [0, 1]: fewer bullheads -> higher.
			pen_r = 1.0 - (my_pen if my_pen < pen_scale else pen_scale) / pen_scale
			if reward_mode == 'pen':
				r = pen_r
			else:
				rank = 1.0
				for k in range(n_opp):
					o = opp_scores[k] + opp_pen[k]
					if o < mine:
						rank += 1.0
					elif o == mine:
						rank += 0.5
				rank_r = (n_tot - rank) / (n_tot - 1)
				if reward_mode == 'blend':
					r = (1.0 - blend) * rank_r + blend * pen_r
				else:
					r = rank_r
			for nd in path:
				nd.N += 1
				nd.W += r

			if perf() >= hard:
				break

		# choose root child: most-visited (robust) or best-mean
		best_card = None
		best_key = None
		use_mean = (self.final == 'mean')
		for c, ch in root.children.items():
			if ch.N == 0:
				continue
			mean = ch.W / ch.N
			key = (mean, ch.N) if use_mean else (ch.N, mean)
			if best_key is None or key > best_key:
				best_key = key
				best_card = c
		if best_card is None:
			return min(hand, key=lambda c: self._eval(rows0, sums0, c, bull))
		return best_card
