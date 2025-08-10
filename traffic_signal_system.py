import tkinter as tk
from tkinter import ttk
import random
import numpy as np
from deap import base, creator, tools, algorithms
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

class TrafficSignalSystem:
    def __init__(self, root):
        self.root = root
        self.root.title("Traffic Signal Optimization System")
        
        # System configuration
        self.num_intersections = 4
        self.phases_per_intersection = 4  # NS_green, NS_yellow, EW_green, EW_yellow
        
        # GA parameters
        self.pop_size = 50
        self.generations = 30
        self.cx_prob = 0.7
        self.mut_prob = 0.2
        
        # Traffic flows (vehicles/hour)
        self.traffic_flows = [random.randint(300, 800) for _ in range(self.num_intersections)]
        
        # Current signal timings
        self.signal_timings = [30, 5, 30, 5] * self.num_intersections  # Default values
        
        # Initialize GA
        self.setup_genetic_algorithm()
        
        # Create GUI
        self.create_gui()
        
        # Initialize visualization
        self.setup_visualization()
        
    def setup_genetic_algorithm(self):
        """Configure the genetic algorithm framework"""
        creator.create("FitnessMin", base.Fitness, weights=(-1.0,))
        creator.create("Individual", list, fitness=creator.FitnessMin)
        
        self.toolbox = base.Toolbox()
        self.toolbox.register("attr_float", random.uniform, 5, 60)
        self.toolbox.register("individual", tools.initRepeat, creator.Individual,
                            self.toolbox.attr_float, n=self.phases_per_intersection*self.num_intersections)
        self.toolbox.register("population", tools.initRepeat, list, self.toolbox.individual)
        
        self.toolbox.register("mate", tools.cxBlend, alpha=0.5)
        self.toolbox.register("mutate", tools.mutGaussian, mu=0, sigma=5, indpb=0.1)
        self.toolbox.register("select", tools.selTournament, tournsize=3)
        self.toolbox.register("evaluate", self.evaluate_signal_plan)
    
    def evaluate_signal_plan(self, individual):
        """Evaluate fitness of a signal timing plan"""
        total_delay = 0
        
        for i in range(self.num_intersections):
            offset = i * self.phases_per_intersection
            ns_green = individual[offset]
            ns_yellow = individual[offset+1]
            ew_green = individual[offset+2]
            ew_yellow = individual[offset+3]
            
            cycle_time = ns_green + ns_yellow + ew_green + ew_yellow
            
            # Simple delay model
            ns_flow = self.traffic_flows[i]
            ew_flow = self.traffic_flows[(i+1)%self.num_intersections]
            
            ns_capacity = (ns_green/cycle_time) * 1800
            ew_capacity = (ew_green/cycle_time) * 1800
            
            ns_capacity = max(ns_capacity, 1)
            ew_capacity = max(ew_capacity, 1)
            
            ns_delay = 0.9 * cycle_time * (1 - ns_green/cycle_time)**2 / (1 - min(ns_flow/ns_capacity, 0.9))
            ew_delay = 0.9 * cycle_time * (1 - ew_green/cycle_time)**2 / (1 - min(ew_flow/ew_capacity, 0.9))
            
            # Penalties
            penalty = 0
            if ns_green < 10 or ew_green < 10:
                penalty += 1000
            if ns_yellow < 3 or ew_yellow < 3:
                penalty += 500
            if cycle_time > 180:
                penalty += 2000
                
            total_delay += (ns_delay * ns_flow/3600 + ew_delay * ew_flow/3600) + penalty
            
        return (total_delay,)
    
    def create_gui(self):
        """Create the main GUI interface"""
        # Control frame
        control_frame = ttk.LabelFrame(self.root, text="Control Panel", padding=10)
        control_frame.grid(row=0, column=0, padx=10, pady=10, sticky="nsew")
        
        # GA parameters
        ttk.Label(control_frame, text="Population Size:").grid(row=0, column=0, sticky="w")
        self.pop_size_entry = ttk.Entry(control_frame)
        self.pop_size_entry.grid(row=0, column=1, padx=5, pady=2)
        self.pop_size_entry.insert(0, str(self.pop_size))
        
        ttk.Label(control_frame, text="Generations:").grid(row=1, column=0, sticky="w")
        self.generations_entry = ttk.Entry(control_frame)
        self.generations_entry.grid(row=1, column=1, padx=5, pady=2)
        self.generations_entry.insert(0, str(self.generations))
        
        ttk.Label(control_frame, text="Crossover Prob:").grid(row=2, column=0, sticky="w")
        self.cx_prob_entry = ttk.Entry(control_frame)
        self.cx_prob_entry.grid(row=2, column=1, padx=5, pady=2)
        self.cx_prob_entry.insert(0, str(self.cx_prob))
        
        ttk.Label(control_frame, text="Mutation Prob:").grid(row=3, column=0, sticky="w")
        self.mut_prob_entry = ttk.Entry(control_frame)
        self.mut_prob_entry.grid(row=3, column=1, padx=5, pady=2)
        self.mut_prob_entry.insert(0, str(self.mut_prob))
        
        # Buttons
        ttk.Button(control_frame, text="Run Optimization", command=self.run_optimization).grid(
            row=4, column=0, columnspan=2, pady=10)
        
        ttk.Button(control_frame, text="Randomize Traffic", command=self.randomize_traffic).grid(
            row=5, column=0, columnspan=2, pady=5)
        
        # Results frame
        results_frame = ttk.LabelFrame(self.root, text="Optimization Results", padding=10)
        results_frame.grid(row=1, column=0, padx=10, pady=10, sticky="nsew")
        
        self.results_text = tk.Text(results_frame, height=10, width=50, wrap=tk.WORD)
        self.results_text.pack(fill=tk.BOTH, expand=True)
        
        # Traffic flow display
        flow_frame = ttk.LabelFrame(self.root, text="Traffic Flows (veh/hr)", padding=10)
        flow_frame.grid(row=0, column=1, rowspan=2, padx=10, pady=10, sticky="nsew")
        
        self.flow_labels = []
        for i in range(self.num_intersections):
            label = ttk.Label(flow_frame, text=f"Intersection {i+1}: {self.traffic_flows[i]}")
            label.pack(anchor="w", padx=5, pady=2)
            self.flow_labels.append(label)
    
    def setup_visualization(self):
        """Set up the matplotlib visualization"""
        self.fig, (self.ax1, self.ax2) = plt.subplots(1, 2, figsize=(10, 4))
        self.canvas = FigureCanvasTkAgg(self.fig, master=self.root)
        self.canvas.get_tk_widget().grid(row=2, column=0, columnspan=2, padx=10, pady=10, sticky="nsew")
        
        # Initial empty plots
        self.ax1.set_title("Fitness Evolution")
        self.ax1.set_xlabel("Generation")
        self.ax1.set_ylabel("Total Delay (veh-hr)")
        self.ax1.grid(True)
        
        self.ax2.set_title("Optimized Green Times")
        self.ax2.set_xlabel("Intersection")
        self.ax2.set_ylabel("Duration (seconds)")
        self.ax2.grid(True)
        
        self.canvas.draw()
    
    def run_optimization(self):
        """Run the genetic algorithm optimization"""
        try:
            self.pop_size = int(self.pop_size_entry.get())
            self.generations = int(self.generations_entry.get())
            self.cx_prob = float(self.cx_prob_entry.get())
            self.mut_prob = float(self.mut_prob_entry.get())
        except ValueError:
            self.results_text.insert(tk.END, "Error: Invalid parameter values!\n")
            return
        
        self.results_text.delete(1.0, tk.END)
        self.results_text.insert(tk.END, "Running optimization...\n")
        self.root.update()
        
        pop = self.toolbox.population(n=self.pop_size)
        
        stats = tools.Statistics(lambda ind: ind.fitness.values)
        stats.register("avg", np.mean)
        stats.register("min", np.min)
        stats.register("max", np.max)
        
        pop, logbook = algorithms.eaSimple(
            pop, self.toolbox, cxpb=self.cx_prob, mutpb=self.mut_prob,
            ngen=self.generations, stats=stats, verbose=False
        )
        
        best_ind = tools.selBest(pop, 1)[0]
        self.signal_timings = best_ind
        
        # Display results
        self.results_text.delete(1.0, tk.END)
        self.results_text.insert(tk.END, f"Optimization complete!\nBest solution found (delay: {best_ind.fitness.values[0]:.1f} veh-hr):\n\n")
        
        for i in range(self.num_intersections):
            offset = i * self.phases_per_intersection
            self.results_text.insert(tk.END, 
                f"Intersection {i+1}:\n"
                f"  NS Green: {best_ind[offset]:.1f}s\n"
                f"  NS Yellow: {best_ind[offset+1]:.1f}s\n"
                f"  EW Green: {best_ind[offset+2]:.1f}s\n"
                f"  EW Yellow: {best_ind[offset+3]:.1f}s\n"
                f"  Cycle Time: {sum(best_ind[offset:offset+4]):.1f}s\n\n"
            )
        
        # Update plots
        self.update_plots(logbook, best_ind)
    
    def update_plots(self, logbook, best_ind):
        """Update the visualization plots"""
        self.ax1.clear()
        self.ax2.clear()
        
        # Plot fitness over generations
        gen = logbook.select("gen")
        avg = logbook.select("avg")
        min_ = logbook.select("min")
        
        self.ax1.plot(gen, avg, label="Average")
        self.ax1.plot(gen, min_, label="Minimum")
        self.ax1.set_title("Fitness Evolution")
        self.ax1.set_xlabel("Generation")
        self.ax1.set_ylabel("Total Delay (veh-hr)")
        self.ax1.legend()
        self.ax1.grid(True)
        
        # Plot optimized green times
        intersections = range(1, self.num_intersections + 1)
        ns_greens = [best_ind[i*self.phases_per_intersection] for i in range(self.num_intersections)]
        ew_greens = [best_ind[i*self.phases_per_intersection + 2] for i in range(self.num_intersections)]
        
        width = 0.35
        self.ax2.bar([x - width/2 for x in intersections], ns_greens, width, label='NS Green')
        self.ax2.bar([x + width/2 for x in intersections], ew_greens, width, label='EW Green')
        self.ax2.set_title("Optimized Green Times")
        self.ax2.set_xlabel("Intersection")
        self.ax2.set_ylabel("Duration (seconds)")
        self.ax2.set_xticks(intersections)
        self.ax2.legend()
        self.ax2.grid(True)
        
        self.canvas.draw()
    
    def randomize_traffic(self):
        """Randomize the traffic flows"""
        self.traffic_flows = [random.randint(300, 800) for _ in range(self.num_intersections)]
        for i, label in enumerate(self.flow_labels):
            label.config(text=f"Intersection {i+1}: {self.traffic_flows[i]}")
        self.results_text.insert(tk.END, "Traffic flows randomized!\n")

if __name__ == "__main__":
    root = tk.Tk()
    app = TrafficSignalSystem(root)
    root.mainloop()