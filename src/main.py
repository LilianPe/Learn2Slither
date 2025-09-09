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
# Enlever la possibilite de faire le move oppose (= suicide)
# -> Finir, ajouter des protections lors de la recherche de la premiere direction pour ne pas chercher en dehors du tableau
# Ajouter stats de train
# Revoir comment gerer sessions, soit valeure par defaut,
# soit erreure
# Separer le load et le save du modele, le faire si flag
# utilise (-load / -save)
# Ajouter un mode step by step -> Facile a integrer en
# console, mais mieux via la fenetre tkinter pour le mode visual
# Passer une session a 1 iteration
# Verifier epsilon decay
# Revoir ancien code pour fix le bug actuel, mais surement remettre le systeme d'index pour les directions et pas direct les directions
# Surement que des int dans la Q-Table et avec la modif ca met les direcions au lieu des int