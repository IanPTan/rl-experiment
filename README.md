# What is this?
I got bored so I decided to play around with tic-tac-toe and reinforcement learning. The approach is very naive, I randomly thought about it and wanted to test it out. Basically, one model (the actor) predicts the right move based on the current state of the board while another (the critic) predicts the success of that move. The critic is trained on game results to predict which moves lead to wins while the actor is trained on the critic's predictions of its success.
# Tasks
