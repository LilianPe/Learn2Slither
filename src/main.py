#!/usr/bin/env python3
import argparse
import os
from Game import Game
from Model import Model


def path_is_valid(path: str) -> bool:
    if path:
        _, ext = os.path.splitext(path)
        if ext != ".pth":
            print(f"{path} is not valid, it need to be a .pth file.")
            return False
    return True

def get_args():
    parser = argparse.ArgumentParser(description="Learn2Slither")
    parser.add_argument(
        "-visual",
        choices=["on", "off"],
        default="off",
        help="enable or disable visual mode"
        )
    parser.add_argument(
        "-step-by-step",
        action="store_true",
        help="Enable step-by-step mode"
        )
    parser.add_argument(
        "-speed",
        type=int,
        choices=[1, 2, 3, 4, 5, 6],
        default=1,
        help="Vitesse du jeu : 1 = lent, 2 = moyen, 3 = rapide"
    )
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
    parser.add_argument(
        "-load",
        help="model .pth to load"
        )
    parser.add_argument(
        "-save",
        help="save the model into the specified path"
        )
    return parser.parse_args()

def main():
    args = get_args()
    sessions = args.sessions
    if not path_is_valid(args.load) or not path_is_valid(args.save):
        return 1
    game = Game()
    model = Model(game, args)
    if model.init_error:
        exit(1)
        pass
    model.training(sessions)


if __name__ == "__main__":
    main()

# Clean les fonctions trop longues avant de reprendre
# Separer le load et le save du modele, le faire si flag
# utilise (-load / -save)
# Ajouter un mode step by step
# Ajouter un parametre pour gerer la vitesse d'update 
# Passer une session a 1 iteration
# Verifier epsilon decay