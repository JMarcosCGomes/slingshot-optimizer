import numpy as np

class Corpo_Celeste:

    def __init__(self, massa, raio, color, name, orbit_radius, angle_deg, vel_x=0.0, vel_y=0.0):
        self.massa = massa
        self.raio = raio
        self.color = color
        self.name = name
        self.trace = []
        self.orbit_radius = orbit_radius
        self.angle_rad = np.deg2rad(angle_deg)
        self.pos_x = self.orbit_radius * np.cos(self.angle_rad)
        self.pos_y = self.orbit_radius * np.sin(self.angle_rad)
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

    #when you want to get the params to Canculate_PlanetSatelite_Speed
    def return_cpss_params(self):
        params = {"planet_mass": self.massa,
                  "planet_x": self.pos_x,
                  "planet_y": self.pos_y,
                  "planet_vx": self.vel_x,
                  "planet_vy": self.vel_y,
                  }
        return params


    def Calculate_SunPlanet_Speed(self):
        const_G = 6.67430e-11
        sun_mass = 2e30
        speed_module = np.sqrt(const_G * sun_mass / self.orbit_radius)
        relative_vx = -speed_module * np.sin(self.angle_rad)
        relative_vy = speed_module * np.cos(self.angle_rad)
        return relative_vx, relative_vy
    

    def Calculate_PlanetSatelite_Speed(self, planet_mass, planet_x, planet_y, planet_vx, planet_vy):
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




#planet creation process
#planeta simples
#self.Planet(massa=pmass, raio=praio, color=pcolor, name=pname, orbit_radius=porbit_radius, angle_deg=pangle_deg)
#self.Planet.vel_x, self.Planet.vel_y = Calculate_SunPlanet_Speed()
#self.corpos_celestes.append(self.Planet)

#se ta orbitando algum planeta; Pqo = planetaqueorbita
#self.Satelite(massa=smass, raio=sraio, color=scolor, name=sname, orbit_radius=sorbit_radius, angle_deg=sangle_deg)
#self.Satelite.pos_x += self.Pqo.pos_x
#self.Satelite.pos_y += self.Pqo.pos_y
#cpss_params = self.Pqo.return_cpss_params()
#self.Satelite.vel_x, self.Satelite.vel_y = self.Satelite.Calculate_PlanetSatelite_Speed(**cpss_params)
#self.corpos_celestes.append(self.Satelite)