# Particle-swarm-optimization-PSO

Reflection for each graph:

Sphere: Constant inertia with a global star topology (Const. w=0.7 Star) achieves the fastest and deepest convergence down to around 10^-19, as the aggressive exploitation toward the global best suits smooth, unimodal landscapes.   
Rosenbrock: All variants perform similarly along the narrow curved valley, though constant inertia configurations (Const. w=0.7) maintain better momentum to navigate the terrain compared to the linearly decaying variant.   
Rastrigin: The ring topology (Const. w=0.7 Ring, k=3) outperforms the global star topologies by preventing premature convergence. Local neighborhood communication preserves swarm diversity, allowing particles to successfully escape local optima in multimodal landscapes.   

AI was used for grammar, bug fixing, and initial structuring of the study material.
