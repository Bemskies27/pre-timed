import random
import numpy as np
from deap import base, creator, tools, algorithms
import matplotlib.pyplot as plt

class TrafficSignalOptimizer:
    def __init__(self, num_intersections=4):
        # System configuration
        self.num_intersections = num_intersections
        self.phases_per_intersection = 4  # [NS_green, NS_yellow, EW_green, EW_yellow]
        
        # Traffic parameters (vehicles/hour)
        self.traffic_flows = [random.randint(300, 800) for _ in range(num_intersections)]
        
        # GA parameters
        self.pop_size = 100
        self.generations = 50
        self.cx_prob = 0.8
        self.mut_prob = 0.2
        
        # Initialize GA
        self.setup_genetic_algorithm()
        
    def setup_genetic_algorithm(self):
        """Configure the genetic algorithm framework"""
        creator.create("FitnessMin", base.Fitness, weights=(-1.0,))  # Minimize delay
        creator.create("Individual", list, fitness=creator.FitnessMin)
        
        self.toolbox = base.Toolbox()
        
        # Genetic operators
        self.toolbox.register("attr_float", random.uniform, 5, 60)  # Signal timing bounds (5-60s)
        self.toolbox.register("individual", tools.initRepeat, creator.Individual,
                            self.toolbox.attr_float, n=self.phases_per_intersection*self.num_intersections)
        self.toolbox.register("population", tools.initRepeat, list, self.toolbox.individual)
        
        self.toolbox.register("mate", tools.cxBlend, alpha=0.5)
        self.toolbox.register("mutate", tools.mutGaussian, mu=0, sigma=5, indpb=0.1)
        self.toolbox.register("select", tools.selTournament, tournsize=3)
        self.toolbox.register("evaluate", self.evaluate_signal_plan)
    
    def evaluate_signal_plan(self, individual):
        """
        Evaluate fitness of a signal timing plan
        Returns total delay in vehicle-hours
        """
        total_delay = 0
        
        for i in range(self.num_intersections):
            # Extract timings for this intersection
            offset = i * self.phases_per_intersection
            ns_green = individual[offset]
            ns_yellow = individual[offset+1]
            ew_green = individual[offset+2]
            ew_yellow = individual[offset+3]
            
            # Calculate cycle time
            cycle_time = ns_green + ns_yellow + ew_green + ew_yellow
            
            # Simple delay model (Webster's formula approximation)
            ns_flow = self.traffic_flows[i]
            ew_flow = self.traffic_flows[(i+1)%self.num_intersections]
            
            # Capacity = (green time/cycle time) * saturation flow (1800 veh/hour)
            ns_capacity = (ns_green/cycle_time) * 1800
            ew_capacity = (ew_green/cycle_time) * 1800
            
            # Avoid division by zero
            ns_capacity = max(ns_capacity, 1)
            ew_capacity = max(ew_capacity, 1)
            
            # Delay calculation (simplified)
            ns_delay = 0.9 * cycle_time * (1 - ns_green/cycle_time)**2 / (1 - min(ns_flow/ns_capacity, 0.9))
            ew_delay = 0.9 * cycle_time * (1 - ew_green/cycle_time)**2 / (1 - min(ew_flow/ew_capacity, 0.9))
            
            # Penalize unrealistic timings
            penalty = 0
            if ns_green < 10 or ew_green < 10:
                penalty += 1000
            if ns_yellow < 3 or ew_yellow < 3:
                penalty += 500
            if cycle_time > 180:  # Max 3 minute cycle
                penalty += 2000
                
            total_delay += (ns_delay * ns_flow/3600 + ew_delay * ew_flow/3600) + penalty
            
        return (total_delay,)
    
    def optimize(self):
        """Run the genetic algorithm optimization"""
        pop = self.toolbox.population(n=self.pop_size)
        
        # Statistics tracking
        stats = tools.Statistics(lambda ind: ind.fitness.values)
        stats.register("avg", np.mean)
        stats.register("min", np.min)
        stats.register("max", np.max)
        
        # Run GA
        pop, logbook = algorithms.eaSimple(
            pop, self.toolbox, cxpb=self.cx_prob, mutpb=self.mut_prob,
            ngen=self.generations, stats=stats, verbose=True
        )
        
        # Return best solution and statistics
        best_ind = tools.selBest(pop, 1)[0]
        return best_ind, logbook
    
    def visualize_results(self, best_ind, logbook):
        """Create visualization of optimization results"""
        plt.figure(figsize=(15, 5))
        
        # Plot fitness evolution
        plt.subplot(1, 2, 1)
        gen = logbook.select("gen")
        avg = logbook.select("avg")
        min_ = logbook.select("min")
        
        plt.plot(gen, avg, label="Average")
        plt.plot(gen, min_, label="Minimum")
        plt.xlabel("Generation")
        plt.ylabel("Total Delay (vehicle-hours)")
        plt.title("Fitness Evolution")
        plt.legend()
        plt.grid(True)
        
        # Plot optimized timings
        plt.subplot(1, 2, 2)
        intersections = range(1, self.num_intersections+1)
        ns_greens = [best_ind[i*self.phases_per_intersection] for i in range(self.num_intersections)]
        ew_greens = [best_ind[i*self.phases_per_intersection+2] for i in range(self.num_intersections)]
        
        width = 0.35
        plt.bar([x - width/2 for x in intersections], ns_greens, width, label='NS Green')
        plt.bar([x + width/2 for x in intersections], ew_greens, width, label='EW Green')
        plt.xlabel("Intersection")
        plt.ylabel("Duration (seconds)")
        plt.title("Optimized Green Times")
        plt.xticks(intersections)
        plt.legend()
        plt.grid(True)
        
        plt.tight_layout()
        plt.show()
    
    def print_solution(self, best_ind):
        """Print the optimized signal timing plan"""
        print("\nOptimized Signal Timing Plan:")
        print("=============================")
        
        for i in range(self.num_intersections):
            offset = i * self.phases_per_intersection
            print(f"\nIntersection {i+1}:")
            print(f"  NS Green: {best_ind[offset]:.1f}s")
            print(f"  NS Yellow: {best_ind[offset+1]:.1f}s")
            print(f"  EW Green: {best_ind[offset+2]:.1f}s")
            print(f"  EW Yellow: {best_ind[offset+3]:.1f}s")
            print(f"  Cycle Time: {sum(best_ind[offset:offset+4]):.1f}s")
        
        print(f"\nEstimated Total Delay: {best_ind.fitness.values[0]:.2f} vehicle-hours")

# In traffic_ga.py
 # More generations
if __name__ == "__main__":
    # Create and run optimizer
    
    optimizer = TrafficSignalOptimizer(num_intersections=6)  # More intersections
optimizer.pop_size = 200  # Larger population
optimizer.generations = 100 
    # Run optimization
best_solution, stats = optimizer.optimize()
    
    # Display results
optimizer.print_solution(best_solution)
optimizer.visualize_results(best_solution, stats)