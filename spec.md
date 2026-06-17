# Building a 6 Nimmt! Agent

### FAI 2026 Final Project

#### Due date: 2026.06.14. (Sun.)

## Game Introduction

### The Objective: Stay Low!

Every card has "Bullheads" (🐮) on it. These are **Negative Points**.

**Winner = Player with the FEWEST Bullheads!**

- Normal Cards: 1
- Ends in 5: 2
- % 10 == 0: 3
- % 11 == 0: 5
- 55 (The King): 7

### Game Setup

- The Deck: 104 cards (1 to 104). Each number is unique.
- The Field: **4 rows** are initialized, each with 1 starting card.
- Player Hands: Players receive **10 cards** each. Game lasts 10 rounds.
- The Rounds: In each round, cards are processed from **Smallest to Largest**.

### How to Play: The Closest Fit

Cards must be placed in a row where the end-number is **smaller** than your card, but **closest** to it.
```
ROW1: 12
ROW2: 35
ROW3: 44
```
Let's say you play 40. It **MUST** go to Row 2 (closest smaller number)

### The Nightmare: The 6th Card

A row can only hold 5 cards. If you are forced to place the 6th card...
```
10 15 22 31 40 [42]
```
**BOOM!** You take all 5 cards (penalty += sum(3, 2, 5, 1, 3)), and your "42" becomes the new start of the row.

### What if your card is too small?

If your card is lower than **all** row-ends, you must take a row, then place your card.
**Selection is AUTOMATED (No user choice)**

## Game Configurations

### Game Specifications – Rules

- Recall: If your card is lower than all row-ends, you must take a row, then place your card.
- Which row will you take? We deterministically choose:
    - The row with the least penalty (number of bullheads)
    - If there is a tie, the row with less number of cards
    - If there is still a tie, the row with the smallest index

### What you need to know about the game engine

```
Class engine:
    __init__(self, cfg, players):
        setup game config, instantiate players, deal cards
    play_game(self):
        play 10 rounds
    play_round(self):
        request each player to play a card and additionally,
        - check for timeouts
        - check for disqualifications
        - check for invalid moves
        - check for memory usage
        - update game history
```

### Game Specifications

- Timeouts: You have __1 second__ to play your card.
    - The timeout is a self-defined class
        ```python
        class TimeoutException(BaseException):
            pass
        ```
- Once the time is out, we will send a TimeoutException to stop your player
    - We automatically play the __smallest card__ for you
    - Your final penalty is your penalty + penalty gained by the min player
- If time is out _and_ you catch the exception
    - You get a penalty (see [Game Specifications – Penalties](#game-specifications--penalties))
    - Do not `except BaseException (as e):` or use a single `except:`
    - Be careful especially if you use LLM to write code!
- Memory Errors:
    - Each player has a RAM usage limit of 1GB.
    - Be careful of storing huge dictionaries (while doing some tree search?)
- Multi-threading, Multi-processing:
    - Forbidden
- Invalid Moves:
    - If you play an invalid card (wrong type, number not in your hand, etc.), we automatically play __the smallest card__ for you.

### Game Specifications – Penalties

If you violate the following rules, your final score is deducted based on severity:
1. Modify the game history
    - We always pass a copied version of board history and your hands to you. No penalty, since you won’t succeed anyway.
2. Change the random seed
    - We have a customized random number generator. No penalty, since you won’t succeed anyway.
3. You purposely or accidentally use more than the RAM limit 1GB
    - **If you hit this limit several times, your score for the final project will be multiplied by up to 50%**
4. If you try to use multi-threading or multi-process
    - **your score for the final project will be multiplied by up to 50%**
5. Timeouts - If you exceed the time limit (1 second) when making decisions, we will automatically play the smallest card for you.
    - In the normal case, this would incur no additional penalty.
    - **However, if you try to catch the exception, your score for the final project will be multiplied by up to 70%.**
    - **Be careful with torch—large batch sizes or models may cause timeouts.**

If you run a tournament, these errors will be recorded and marked:
- DQ: Player timeouts and swallows a timeout exception.
- EXC: Player code raised an exception during their turn.
- OOM: Player’s **matchup** subprocess was killed due to exceeding the max_memory_mb_per_matchup limit.
- ERR: Other errors occurred in the matchup.

Please refer to [README.md](../README.md) for more details.

## Tournament Configurations

### A Tournament

A tournament is what TAs would run to evaluate all of your players.
- A tournament consists of multiple partitions
    - A partition is dividing all players into groups of four
    - If there is a remainder, random players are padded
- Each partition consists of multiple games
    - Specifically, if there are N students, there would be ceil(N / 4) games
- We will run at least 500 partitions to mitigate randomness of final results
    - If we think the results haven’t converged, we will run more

### Scoring Policy (Per Game)

- In each game, we rank players by their respective penalties
    - If there is a tie, all players receive the average of the total rank
        - For example, if Players 1-4 get penalty [10, 10, 10, 3], they rank [ (2+3+4)/3, (2+3+4)/3, (2+3+4)/3, 1] respectively
        - If players 1-4 get penalty [0, 1, 5, 5], they rank [1, 2, 3.5, 3.5] respectively
    - If your player gets disqualified, they are replaced by a min player. Final penalty is your original penalty + penalty gained by the min player

### Scoring Policy (Per Tournament)

- Your final ranking depends on your final average total rank.
- Conversion from the average total rank to your score for the final project is shown in [Grading Policy](#grading-policy).

## Environment Specifications

### Machine Specifications

- We will run the tournament on a machine at least as powerful as the CSIE workstations (ws1–ws7).
    - No GPU will be provided. Please use CPU for all computation.
    - Test your players on ws1–ws7 to check for OOM errors and timeouts.
- Remember:
    - **Only 1 second for each decision**
    - **Only 1GB memory allowed for each player**
    - **You cannot use multi-process or multi-thread!**

### Environment Specifications

- Python version 3.13.11
- Allowed packages are listed in [requirements.txt](../requirements.txt)
    - Tournaments will be run under this setting.
- If you want to use any additional packages, email us for permission first

## Sample Code Explanation


### Write your own players

Please refer to [README.md](../README.md) for more details.

Notes for `__init__()`:
- The player_idx is given just for you to lookup info on the board_history.
- You can pass in other arguments if you want for testing, but remember to remove them or set default values in your final submitted version.

Notes about `action()`:
- Hands are sorted! You don’t have to sort them again.
- It is recommended to compute values you need from the board_history instead of storing them, since your values might be incorrect if timeouts occur.

History format:
```python
history = {
    "board": self.board,
    "scores": self.scores,  # current score of each agent
    "round": self.round,
    "history_matrix": self.history_matrix,  # cards players played in each round
    "board_history": self.board_history,  # cards on the table in each round
    "score_history": self.score_history,  # penalties players gained in each round
}
```
e.g. you can get you current score from `history["scores"][self.player_idx]`, or get the cards played in the last round from `history["history_matrix"][-1]`, where the card you played is `history["history_matrix"][-1][self.player_idx]`.

Example:
```python
history = {
    "board": [
        [ 4, 8 ],
        [ 53 ],
        [ 55, 102 ],
        [ 13, 22 ]
    ],
    "scores": [
        0, 1
    ],
    "round": 2,
    "history_matrix": [
        [ 22, 4 ],
        [ 102, 8 ]
    ],
    "board_history": [
        [
            [ 41 ],
            [ 53 ],
            [ 55 ],
            [ 13 ]
        ],
        [
            [ 4 ],
            [ 53 ],
            [ 55 ],
            [ 13, 22 ]
        ],
        [
            [ 4, 8 ],
            [ 53 ],
            [ 55, 102 ],
            [ 13, 22 ]
        ]
    ],
    "score_history": [
        [ 0, 1 ],
        [ 0, 1 ]
    ]
}
```
### Running Games and Tournaments

Please refer to [README.md](../README.md) for more details.

### Saved Results – Game

```python
output_data = {
    "config": game_config,
    "game_results": {
        "initial_hands": initial_hands,
        "final_scores": final_scores,
        "history": history  # full_history
    }
}
```
```python
full_history = {
    "board_history": self.board_history,
    "flags_matrix": self.flags_matrix,
    "final_scores": self.scores,
    "disqualified_players": list(self.disqualified_players),
    "timeout_counts": dict(self.timeout_counts),
    "exception_counts": dict(self.exception_counts)
}
```
Results are saved in `results/game`.

### Saved Results – Tournament

```python
output_data = {
    "config": config,
    "standings": final_standings,
    "history": history
}
```
`final_standings` includes:
- Disqualification counts
- Timeout counts
- Exception catching counts
- Player total scores
- Player total ranks
- Player total games played
- Player ranks for each game
- Player scores for each game
`history` is a list:
```python
matchup_history.append({
    "matchup_id": idx,
    "players": list(combo),
    "results": matchup_res_list
})
```

### Setting Configurations

Check [`configs/`](../configs/) for examples.

You can set the configurations of a single game or a whole tournament.
- Player config
- Engine config
- Tournament config

There are many arguments for each configuration category.
Please refer to [README.md](../README.md) for more details.

### Setting Configurations – Players

All of these are valid formats:
```json
"players": [
    {
        "path": "src.players.your_student_id.best_player1",
        "class": "BestPlayer1",
        "args": {}
    },
    [ "src.players.your_student_id.best_player1", "BestPlayer1" ],
    [ "src.players.your_student_id.best_player1", "BestPlayer2", {} ],
]
```
Check [`configs/`](../configs/) for examples.

## Grading Policy & Schedule

### Grading Policy

#### Performance (70 pts)
- Midterm Submission (15 pts)
    - Tournament with all students + 40 baselines (a weaker subset)
- Final Submission (55 pts)
    - Tournament with all students + 55 baselines (the full set)

#### Report (30 pts)

#### Novelty: qualitative + quantitative, up to +5 pts, total capped at 100

### Baselines

Please refer to [README.md](../README.md) for more details.

- We will provide baselines in binary form.
- There are **55** baselines in total.
    - 5 baselines from B1-B27 will be released before the midterm.
    - 5 baselines from B1-B55 will be released after the midterm.
        - The released baselines are only a subset, so **avoid overfitting** to them.
- Baselines may be used for training, but **cannot** be submitted directly or imported into your final submission.
- Baselines are sorted by performance (B1 is the weakest, B55 is the strongest).

### Baselines Scoring Criteria

We will use linear interpolation to convert your average total rank to your score for the final project.
- For midterm, B5 = 60 pts, B20 = 90 pts
- For final, B20 = 40 pts, B55 = 90 pts

## Submission Policy

### Code

You can submit two best players
- Name them:
    - BestPlayer1 in best_player1.py
    - BestPlayer2 in best_player2.py
- Try to think of diverse strategies that can counter different playing styles!
    - You don’t know what the player pool would be like.

All files that were used to produce your final methods should be submitted

- e.g., training scripts, data collection pipelines, etc.
- If you have intermediate files (other than best players), you need to write an additional README.md explaining how to reproduce your results

Your code will be evaluated in a closed network, i.e., you cannot download models from the web.

### Format
- Please compress your player and the related files in a single .zip named with your student id in lowercase.
- The size of the submitted .zip file should be less than 2GB.
- Don’t hand in unnecessary pycache files.

> Use `zip -r student_id.zip student_id/` in terminal to compress your files.

### Report

- Your report should include but not be limited to
    - Methods tried: Compare all the approaches you have explored.
    - Core Idea: Explain the main idea and logic of your 2 final algorithms.
    - Implementation: Details on optimizations and hyperparameter tuning.
    - Self-Assessment: What you are good at and what you struggled with.
    - Strategic Profiling: Opponent styles favorable and unfavorable to your 2 algorithms.
    - Agent Comparison: Compare the trade-offs of your two agents.
    - (Optional) Game analysis (vs. Texas Hold’em), future improvements...
- You should write your report in maximum 3 A4 pages in pdf format
- Formatting: Use reasonable font sizes and margins.