from trueskill import Rating, rate_1vs1, setup


setup(draw_probability=0.0)

bots = {"Tekus1A": Rating(),
        "Tekus1B": Rating(),
        "Tekus1C": Rating(),
        "Tekus1D": Rating(),
        "Tekus1E": Rating(),
        "Tekus2A": Rating(),
        "Rookie": Rating(),
        "Pro": Rating()}

results_file = open("results.txt", "r")
results = results_file.readlines()

for result in results[1:]:
    winner, loser = result.strip().split(",")
    bots[winner], bots[loser] = rate_1vs1(bots[winner], bots[loser])


for bot, rating in bots.items():
    print(f"{bot}: mu={rating.mu}, sigma={rating.sigma}")
