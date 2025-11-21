import numpy as np

class Corpo_Celeste:

    def __init__(self, massa, raio, color, name, orbit_radius, angle_deg, wir_id=None, is_orbiting=True, vel_x=0.0, vel_y=0.0):
        self.massa = massa
        self.raio = raio
        self.color = color
        self.name = name
        self.trace = []
        self.orbit_radius = orbit_radius
        self.angle_rad = np.deg2rad(angle_deg)
        self.pos_x = self.orbit_radius * np.cos(self.angle_rad)
        self.pos_y = self.orbit_radius * np.sin(self.angle_rad)
        self.wir_id = wir_id #who is related id, lua e relacionada com a terra, marte é relacionado com o sol .. daria pra colocar nome mas preferi id
        self.is_orbiting = is_orbiting #isso aqui caso tenha um monte de cc e decidir adicionar no loop pra somar as posicoes e calcular velocidades
        self.vel_x = float(vel_x)
        self.vel_y = float(vel_y)


    # essa funcao é pro equacoes_movimento
    # x, y, vx, vy = estado
    def return_estado(self):
        info = [self.pos_x, self.pos_y, self.vel_x, self.vel_y]
        return info

    def return_pos(self):
        info = [self.pos_x, self.pos_y]
        return info

    def return_vel(self):
        info = [self.vel_x, self.vel_y]
        return info

    
    def return_cov_parameters(self):
        #se vai calcular a velocidade da terra com o sol você usa isso no sol pra pegar as infos
        params = {"planet_mass": self.massa,
                  "planet_x": self.pos_x,
                  "planet_y": self.pos_y,
                  "planet_vx": self.vel_x,
                  "planet_vy": self.vel_y,
                  }
        return params


    def Calculate_Orbital_Velocity(self, planet_mass, planet_x, planet_y, planet_vx, planet_vy):
        const_G = 6.67430e-11
        v_planet = np.array([planet_vx, planet_vy])
        vector_planet_satelite = np.array([self.pos_x - planet_x, self.pos_y - planet_y])
        real_distance = np.linalg.norm(vector_planet_satelite) #norma
        speed_module = np.sqrt(const_G * planet_mass / real_distance)
        direcao_tangente = np.array([-vector_planet_satelite[1], vector_planet_satelite[0]]) / real_distance
        relative_speed = direcao_tangente * speed_module
        resultant_speed_x = v_planet[0] + relative_speed[0]
        resultant_speed_y = v_planet[1] + relative_speed[1]

        return resultant_speed_x, resultant_speed_y


#self.Planet = Corpo_Celeste(massa=pmass, raio=praio, color=pcolor, name=pname, orbit_radius=por, angle_deg=pad, wir_id=self.corpos_celestes.index(self.Pwir))
#self.Planet.pos_x += self.Pwir.pos_x
#self.Planet.pos_y += self.Pwir.pos_y
#covp = self.corpos_celestes[self.Planet.wir_id].return_cov_parameters()
#self.Planet.vel_x, self.Planet.vel_y = self.Planet.Calculate_Orbital_Velocity(**covp)
#self.corpos_celestes.append(self.Planet)