#!/usr/bin/env python3
import argparse
from Game import Game
from Model import Model


def get_args():
    parser = argparse.ArgumentParser(description="Learn2Slither")
    parser.add_argument(
        "-visual",
        choices=["on", "off"],
        default="off",
        help="enable or disable visual mode")
    parser.add_argument(
        "-dontlearn",
        action="store_false",
        help="disable model learning"
        )
    parser.add_argument(
        "-noprint",
        action="store_false",
        help="disable printing"
        )
    parser.add_argument(
        "-sessions",
        type=int,
        required=True,
        help="number of sessions"
        )
    return parser.parse_args()


def main():
    args = get_args()
    sessions = args.sessions
    game = Game()
    model = Model(game, "model/first_model.npy", args)
    if args.noprint:
        print(f"Session {sessions}:")
    model.training(sessions)


if __name__ == "__main__":
    main()

# Clean les fonctions trop longues avant de reprendre
# Revoir state voir si il peut etre simplifie pour le snake
# Revoir si move oppose (= toujours mort) doit vraiment
# etre possible ou non
# Ajouter stats de train
# Revoir comment gerer sessions, soit valeure par defaut,
# soit erreure
# Separer le load et le save du modele, le faire si flag
# utilise (-load / -save)
# Ajouter un mode step by step -> Facile a integrer en
# console, mais mieux via la fenetre tkinter pour le mode visual
