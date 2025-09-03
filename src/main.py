#!/usr/bin/env python3
import argparse
from Game import Game
from Direction import Direction
from Model import Model
import tkinter as tk


def main():
    parser = argparse.ArgumentParser(description="Learn2Slither")
    parser.add_argument("-visual", choices=["on", "off"], default="off", help="enable or disable visual mode")
    parser.add_argument("-dontlearn", action="store_false", help="disable model learning")
    parser.add_argument("-noprint", action="store_false", help="disable printing")
    args = parser.parse_args()
    game = Game()
    model = Model(game, "model/frist_model.npy", args.dontlearn, args.noprint, args.visual == "on")
    # game.print_snake_view()
    # state = model.convert_state()
    for s in range(0, 5):
        if args.visual == "on":
            root = tk.Tk()
        else:
            root = None
        if args.noprint: print(f"Session {s}:")
        model.training(root)
        if args.visual == "on":
            root.mainloop()


if __name__ == "__main__":
    main()

# Gerer la fermeture du tkinter pour eviter le crash
# Revoir state voir si il peut etre simplifie pour le snake
# Revoir si move oppose (= toujours mort) doit vraiment etre possible ou non