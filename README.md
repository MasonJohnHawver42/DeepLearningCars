# DeepLearningCars
Using deep genetic learning cars are able to race around a randomly generated track.

# Genetic Learning
Genetic learning is an algorithm used to generate to most fit entity. 
For this to work two things must be implemented a way of determining the fittest (or best) entity,
and a way to mutate that entity ( ie create a random entity based off a parent entity).
Now the algorithm starts by creating random entities based on no parents. 
Then it determines the fittest entity and creates the next generation based upon that entity.
This last step is repeated until the fitness no longer improves.


wiki page - https://en.wikipedia.org/wiki/Genetic_algorithm

# Where the Deep comes in
I use a feed forward neural network to determine the steering angle, breaking force, and engine force of the car.
The inputs to this network is distances from rays cast out of the car at various angles ( this was the exact set of angles - [-40, -20, 0, 20, 40 ] );
However, I found out that the networks can learn extremely well with any set of angles!

# The NN model

input layer - 5 units ( distances of rays as I mentioned earlier - [-40, -20, 0, 20, 40 ] )

dense layer - 5 units in, 4 units out, relu activation -> f(x) = min(max(0, x), 100)

dense layer - 4 units in, 3 units out, relu activation -> f(x) = min(max(0, x), 100)

dense layer - 3 units in, 2 units out, tanh activation -> f(x) =  ( 2.0 / ( 1 + (e ^ (-2 * x / 10.0) ) ) ) - 1

I used modified activation functions because it worked better with the input (this became apparent when rendering a 
representation of the fittest car to the screen; 
the nodes were basically binary, so I adjusted range of values the activation function would accept inorder to learn 
more of the input space).

# All of it Together
Each car is given its own copy of the model.
Once the learning process has begun, 20 cars with randomly set models are generated.
Then fittest car is determined by having all 20 of the cars race in a random track. The cars actions are determined by the model.
Finally, the next generation is created by mutating the model of the fittest car 19 times. Then the process will repeat until the fitness begins to stagnate to avoid "overfitting".

# Needs
numpy, scipy, pygame

<code> pip3 install numpy, scipy, pygame </code>
# Run
<code> 
cd [ dir of DeepLearningCars.py and barcade-brawl.tff (<-its the font I used) ]

python3 DeepLearningCars.py
</code>
