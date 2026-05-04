import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

class Visualizer():
    def __init__(self, animation_interval=100, duration=4e8, limit_x=4e11, limit_y=4e11):
        self.duration = duration
        self.animation_interval = animation_interval        
        self.limit_x = limit_x
        self.limit_y = limit_y
        self.solution_array = None
        self.celestial_bodies = None

    
    def set_solution_array(self, solution_array):
        self.solution_array = solution_array


    def set_celestial_bodies(self, celestial_bodies, fixed_body_index):
        self.celestial_bodies = celestial_bodies
        self.fixed_body_index = fixed_body_index


    def create_plot(self, ax):
        ax.set_xlim(-self.limit_x, self.limit_x)
        ax.set_ylim(-self.limit_y, self.limit_y)
        ax.set_aspect("equal", adjustable="datalim")
        ax.set_facecolor("gray")
        ax.set_title("Solar System")
        ax.set_xlabel("X (m)")
        ax.set_ylabel("Y (m)")
        

    def plot_celestial_bodies(self, ax):
        if self.celestial_bodies is None:
            print("Visualizer.plot_celestial_bodies() error #1")
            return
        
        else:
            for cb in self.celestial_bodies:
                if (cb.role == "satellite") | (cb.role == "probe"):
                    ax.plot([cb.pos_x], [cb.pos_y], "o", markersize=6, label=f"{cb.name}", markerfacecolor=cb.color, markeredgecolor="black", markeredgewidth=1.0)
                else:
                    ax.plot([cb.pos_x], [cb.pos_y], "o", markersize=12, label=f"{cb.name}", markerfacecolor=cb.color, markeredgecolor="black", markeredgewidth=1.0)


    def simple_plot(self):
        fig, ax = plt.subplots(figsize=(15, 8))
        self.create_plot(ax)
        self.plot_celestial_bodies(ax)
        ax.set_title("Plot simples para visualizar as posicoes iniciais")
        plt.legend()
        plt.show()

        
    def get_current_state(self, y_in_t):
        current_states = []
        ptr = 0
        for i, cb in enumerate(self.celestial_bodies):
            if i == self.fixed_body_index:
                current_states.append({"pos_x": cb.pos_x, "pos_y": cb.pos_y, "vel_x": cb.vel_x, "vel_y": cb.vel_y})
            else:
                current_states.append({"pos_x": y_in_t[ptr], "pos_y": y_in_t[ptr + 1], "vel_x": y_in_t[ptr + 2], "vel_y": y_in_t[ptr + 3]})
                ptr += 4
        return current_states


    def build_traces(self, solution_array):
        if self.celestial_bodies is None:
            print("Visualizer.build_traces() error #1")
            return
        
        for cb in self.celestial_bodies:
            cb.trace = []
        
        for step in range(len(solution_array)):
                y_in_t = solution_array[step]
                current_states = self.get_current_state(y_in_t)
                for i, state in enumerate(current_states):
                    self.celestial_bodies[i].trace.append((state["pos_x"], state["pos_y"]))


    def animate(self, add_trace=True):
        if (self.solution_array is None) or (self.celestial_bodies is None):
            print("Visualizer.animate() error #1")
            return
        
        if add_trace:
            self.build_traces(self.solution_array)

        fig, ax = plt.subplots(figsize=(15, 8))
        self.create_plot(ax)
        
        lines = []
        points = []
        for cb in self.celestial_bodies:
            (line,) = ax.plot([], [], "-", lw=1, color=cb.color, alpha=0.5)
            if (cb.role == "satellite") | (cb.role == "probe"):
                (point,) = ax.plot([],[],"o",markersize=6,label=f"{cb.name}",markerfacecolor=cb.color,markeredgecolor="black",markeredgewidth=1.0)
            else:
                (point,) = ax.plot([],[],"o",markersize=12,label=f"{cb.name}",markerfacecolor=cb.color)
            lines.append(line)
            points.append(point)


        def update(frame):
            current_states = self.get_current_state(self.solution_array[frame])

            for i, cb in enumerate(self.celestial_bodies):
                lines[i].set_data([p[0] for p in cb.trace[:frame + 1]], [p[1] for p in cb.trace[:frame + 1]])
                points[i].set_data([current_states[i]["pos_x"]], [current_states[i]["pos_y"]])

            return lines + points
        anim = FuncAnimation(fig, update, frames=len(self.solution_array), interval=self.animation_interval, blit=False, repeat=False)
        plt.legend()
        plt.show()
