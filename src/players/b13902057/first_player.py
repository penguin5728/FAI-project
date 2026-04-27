import random
import time


class FirstPlayer:
	"""A strong heuristic player with bounded one-round Monte Carlo lookahead."""

	def __init__(self, player_idx):
		self.player_idx = player_idx
		self._rng = random.Random(2026 + player_idx)
		self._card_scores = [0] + [self._bullheads(i) for i in range(1, 105)]

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

	def _row_score(self, row):
		return sum(self._card_scores[c] for c in row)

	def _choose_low_card_row(self, board):
		return min(
			range(len(board)),
			key=lambda i: (self._row_score(board[i]), len(board[i]), i),
		)

	def _place_card(self, board, card):
		"""Mutates board and returns penalty incurred by this placement."""
		best_row_idx = -1
		max_under = -1
		for idx, row in enumerate(board):
			last = row[-1]
			if last < card and last > max_under:
				max_under = last
				best_row_idx = idx

		if best_row_idx != -1:
			if len(board[best_row_idx]) >= 5:
				penalty = self._row_score(board[best_row_idx])
				board[best_row_idx] = [card]
				return penalty
			board[best_row_idx].append(card)
			return 0

		low_idx = self._choose_low_card_row(board)
		penalty = self._row_score(board[low_idx])
		board[low_idx] = [card]
		return penalty

	def _immediate_penalty(self, board, card):
		board_copy = [row[:] for row in board]
		return self._place_card(board_copy, card)

	def _candidate_cards(self, hand, board):
		immediate = [(c, self._immediate_penalty(board, c)) for c in hand]
		immediate.sort(key=lambda x: (x[1], x[0]))

		best_pen = immediate[0][1]
		shortlist = [c for c, p in immediate if p <= best_pen + 2]
		if len(shortlist) < min(5, len(hand)):
			shortlist = [c for c, _ in immediate[: min(5, len(hand))]]
		return shortlist

	def _collect_unseen_cards(self, hand, history):
		seen = set(hand)
		for row in history["board"]:
			seen.update(row)

		for played_row in history.get("history_matrix", []):
			seen.update(played_row)

		return [c for c in range(1, 105) if c not in seen]

	def _estimate_card(self, my_card, board, unseen_cards, opp_count, n_samples, deadline):
		total_penalty = 0.0
		total_sq = 0.0
		used = 0

		for _ in range(n_samples):
			if time.time() >= deadline:
				break

			sim_board = [row[:] for row in board]
			if len(unseen_cards) >= opp_count:
				opp_cards = self._rng.sample(unseen_cards, opp_count)
			else:
				opp_cards = unseen_cards[:]

			played = [(my_card, True)] + [(c, False) for c in opp_cards]
			played.sort(key=lambda x: x[0])

			my_penalty = 0
			for card, is_me in played:
				p = self._place_card(sim_board, card)
				if is_me:
					my_penalty = p

			total_penalty += my_penalty
			total_sq += my_penalty * my_penalty
			used += 1

		if used == 0:
			return float(self._immediate_penalty(board, my_card))

		mean = total_penalty / used
		variance = max(0.0, total_sq / used - mean * mean)
		return mean + 0.12 * variance + 0.002 * my_card

	def action(self, hand, history):
		if len(hand) == 1:
			return hand[0]

		board = history["board"]
		n_players = len(history["scores"])
		opp_count = max(0, n_players - 1)

		# Keep a safety margin below the framework timeout.
		deadline = time.time() + 0.82

		candidates = self._candidate_cards(hand, board)
		unseen_cards = self._collect_unseen_cards(hand, history)

		if not unseen_cards or opp_count == 0:
			return min(candidates, key=lambda c: (self._immediate_penalty(board, c), c))

		rounds_left = len(hand)
		base_samples = 56 if rounds_left >= 6 else 84
		per_card_samples = max(20, base_samples // max(1, len(candidates)))

		best_card = candidates[0]
		best_score = float("inf")

		for card in candidates:
			if time.time() >= deadline:
				break
			score = self._estimate_card(
				my_card=card,
				board=board,
				unseen_cards=unseen_cards,
				opp_count=opp_count,
				n_samples=per_card_samples,
				deadline=deadline,
			)
			if score < best_score or (score == best_score and card > best_card):
				best_score = score
				best_card = card

		return best_card

