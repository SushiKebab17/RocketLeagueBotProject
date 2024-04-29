# imports
from trueskill import Rating, rate_1vs1, setup, BETA
from trueskill.backends import cdf
from math import sqrt

# initialise global environment to have a draw probability of 0
setup(draw_probability=0.0)

# create dictionary of Ratings for all bots
bots = {"Tekus1C": Rating(),
        "Tekus1D": Rating(),
        "Tekus1E": Rating(),
        "Tekus2A": Rating(),
        "Rookie": Rating(),
        "Pro": Rating()
        }

# create a dictionary of dictionaries for bot winrate vs another bot

winrate = {
    "Tekus1C": {
        "Tekus1C": [0, 0],
        "Tekus1D": [0, 0],
        "Tekus1E": [0, 0],
        "Tekus2A": [0, 0],
        "Rookie": [0, 0],
        "Pro": [0, 0]
    },
    "Tekus1D": {
        "Tekus1C": [0, 0],
        "Tekus1D": [0, 0],
        "Tekus1E": [0, 0],
        "Tekus2A": [0, 0],
        "Rookie": [0, 0],
        "Pro": [0, 0]
    },
    "Tekus1E": {
        "Tekus1C": [0, 0],
        "Tekus1D": [0, 0],
        "Tekus1E": [0, 0],
        "Tekus2A": [0, 0],
        "Rookie": [0, 0],
        "Pro": [0, 0]
    },
    "Tekus2A": {
        "Tekus1C": [0, 0],
        "Tekus1D": [0, 0],
        "Tekus1E": [0, 0],
        "Tekus2A": [0, 0],
        "Rookie": [0, 0],
        "Pro": [0, 0]
    },
    "Rookie": {
        "Tekus1C": [0, 0],
        "Tekus1D": [0, 0],
        "Tekus1E": [0, 0],
        "Tekus2A": [0, 0],
        "Rookie": [0, 0],
        "Pro": [0, 0]
    },
    "Pro": {
        "Tekus1C": [0, 0],
        "Tekus1D": [0, 0],
        "Tekus1E": [0, 0],
        "Tekus2A": [0, 0],
        "Rookie": [0, 0],
        "Pro": [0, 0]
    }
}

# open results.txt
results_file = open("results.txt", "r")
results = results_file.readlines()

# get new TrueSkill ratings for each matchup, and update winrate table
for result in results[1:120]:
    winner, loser = result.strip().split(",")
    bots[winner], bots[loser] = rate_1vs1(bots[winner], bots[loser])
    winrate[winner][loser][0] += 1
    winrate[winner][loser][1] += 1
    winrate[loser][winner][1] += 1

# compute and output each bot's conservative skill estimate
for bot, rating in bots.items():
    print(f"{bot}: conservative: {rating.mu - 3 * rating.sigma}")

# output the winrate table
for bot1, bot2 in winrate.items():
    print(f"{bot1}: {bot2}")

# define the win probabilty of player against opponent


def win_probability(player_rating, opponent_rating):
    delta_mu = player_rating.mu - opponent_rating.mu
    denom = sqrt(2 * (BETA * BETA) + pow(player_rating.sigma,
                 2) + pow(opponent_rating.sigma, 2))
    return cdf(delta_mu / denom)


# make Necto use a predefined rating, adapted from Neville Walo's Reference League in their paper about Seer.
necto = Rating(mu=56.14, sigma=3.29)

# output the win probability of two bots
print(win_probability(bots["Tekus2A"], necto))
