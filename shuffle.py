import random
lines = open("combined_train.txt").readlines()
random.shuffle(lines)
open("rain.txt", 'w').writelines(lines)