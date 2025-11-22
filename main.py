from Universo import Universo


if __name__ == "__main__":
    universo = Universo()

    # so pra ver as posicoes
    #teste = universo.simple_plot()

    #testing if the version to optimizer will work correctly
    solucao = universo.simular_teste()

    #without optimizing anything
    #solucao = universo.simular_simple()
    universo.animar(solucao)

