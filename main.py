from Universo import Universo


if __name__ == "__main__":
    universo = Universo()

    #a maneuver to test if it's working. na real a manobra ta errada!
    '''
    maneuver_params = {
        "t":1e3,
        "dvx":0,
        "dvy":1e4,
    }
    universo.maneuver_add(**maneuver_params)
    #'''

    # so pra ver as posicoes
    #teste = universo.simple_plot()
    solucao = universo.simular()
    universo.animar(solucao)