# Suket Arya Third Year Project

This project contains the implementation of the training of HitBall and Tekus.

This project is divided into two sections - everything to do with the HitBall bot is in `/RLHitBall`, and everything to do with Tekus is in `/RLTekus`.

## Instructions on running with RLBot

Please read the instructions below carefully before performing the steps.

1) To run any version of Tekus, you will need to have the game Rocket League installed through Epic Games or Steam. Note that this will mean you must have a Windows PC capable of running Rocket League at the minimum of 120fps. If you do have it installed, please ensure that it is updated. If you do not have Rocket League installed, download the game [here](https://store.epicgames.com/en-US/p/rocket-league) through Epic Games, and install it.

2) Install Python version 3.9.13 if you do not have it already.

3) Currently, the RLBot system is configured to put Tekus(2A) on the blue team, playing against the Rookie bot on the orange team. If you would like to play against Tekus2A, go into `/RLTekus/Tekus/rlbot.cfg` and set `participant_type_1` to `human`.

4) In the terminal, go into the folder `/RLTekus/Tekus`, and run `run.py` with Python 3.9.13. This should install all the dependencies listed below, then start up the RLBot system. Then, it should load the game, and put the bots (or the bot and you) in the kickoff positions. Before or during the first countdown, you must escape (using `Esc`) into the pause menu, as at this point, Tekus might not have fully loaded in, sometimes seen by a stream of messages in the console. When the console prints "Tekus Ready - Index: 0" and no more messages are being printed, you can exit the pause menu and resume the game.

Python dependencies include:
- `gym`==0.21.0
- `torch`==2.0.1+cpu
- `stable_baselines3`==1.8
- `numpy`>=1.24.3
- `rlgym_sim`==1.2.5
- `rlgym_compat`>=1.1.0
- `rlbot`==1.*

5) If the game has finished (or you would like to end the program before), make sure to kill the process in the terminal (by doing `Ctrl`+`C`) before closing the game.