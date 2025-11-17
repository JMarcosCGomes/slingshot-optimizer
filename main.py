from Universo import Universo


if __name__ == "__main__":
    universo = Universo()

    maneuver_params = {
        "t":1e6,
        "dvx":1.2e2,
        "dvy":2e2,
    }
    universo.maneuver_add(**maneuver_params)

    solucao = universo.simular()
    universo.animar(solucao)