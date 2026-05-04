import numpy as np
import math

class CelestialBody:

    def __init__(self, name, role, mass, radius, color, orbit_radius, angle_deg, wir=None, is_orbiting=True, vel_x=0.0, vel_y=0.0):
        self.name = name
        self.role = role
        self.mass = mass
        self.radius = radius
        self.color = color
        self.trace = []
        self.orbit_radius = orbit_radius
        self.angle_rad = np.deg2rad(angle_deg)
        self.pos_x = self.orbit_radius * np.cos(self.angle_rad)
        self.pos_y = self.orbit_radius * np.sin(self.angle_rad)
        self.wir = wir #who is related id, lua e relacionada com a terra, marte é relacionado com o sol .. daria pra colocar nome mas preferi id
        self.is_orbiting = is_orbiting #isso aqui caso tenha um monte de cc e decidir adicionar no loop pra somar as posicoes e calcular velocidades
        self.vel_x = float(vel_x)
        self.vel_y = float(vel_y)


    # essa funcao é pro equacoes_movimento
    # x, y, vx, vy = estado
    def get_state(self):
        info = [self.pos_x, self.pos_y, self.vel_x, self.vel_y]
        return info

    def get_pos(self):
        info = [self.pos_x, self.pos_y]
        return info

    def get_vel(self):
        info = [self.vel_x, self.vel_y]
        return info

    
    def get_cov_parameters(self):
        #se vai calcular a velocidade da terra com o sol você usa isso no sol pra pegar as infos
        params = {"planet_mass": self.mass,
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
        tangent_unit_vector = np.array([-vector_planet_satelite[1], vector_planet_satelite[0]]) / real_distance
        relative_speed = tangent_unit_vector * speed_module
        resultant_speed_x = v_planet[0] + relative_speed[0]
        resultant_speed_y = v_planet[1] + relative_speed[1]

        return resultant_speed_x, resultant_speed_y
    

    def Calculate_Probe_Velocity(self, launch_speed, planet_vel):
        planet_vel_norm = np.linalg.norm(planet_vel)
        unit_tangent = planet_vel / planet_vel_norm
        rocket_vel_module = launch_speed
        rocket_vel = rocket_vel_module * unit_tangent
        probe_vel_x = planet_vel[0] + rocket_vel[0]
        probe_vel_y = planet_vel[1] + rocket_vel[1]
        return [probe_vel_x, probe_vel_y]


    def Calculate_Probe_Angle(self, planet_vel):
        planet_vel_norm = np.linalg.norm(planet_vel)
        unit_tangent = planet_vel / planet_vel_norm
        probe_angle_rad = math.atan2(unit_tangent[1], unit_tangent[0])
        probe_angle_deg = math.degrees(probe_angle_rad)
        return probe_angle_deg

    
    def Recalculate_Probe_Position(self, planet_vel):
        self.angle_deg = self.Calculate_Probe_Angle(planet_vel)
        self.angle_rad = np.deg2rad(self.angle_deg)
        self.pos_x = self.orbit_radius * np.cos(self.angle_rad)
        self.pos_y = self.orbit_radius * np.sin(self.angle_rad)
        return [self.pos_x, self.pos_y]