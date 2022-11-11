# DeepLearningCars
video:

[![Watch the video](https://img.youtube.com/vi/rklWAOirpQ4/hqdefault.jpg)](https://youtu.be/rklWAOirpQ4)

# Genetic Learnig
Genetic learning is an algorithim used to generate to most fit entity. 
For this to work two things must be implemented a way of determining the fitest ( or best ) entity and a way to mutate that entity ( ie create a random entity based off a parent entity).
Now the algorithim starts by creating random entitties based on no parents. 
Then it determines the most fit entity and creates the next generation based upon that entity.
This last step is repeated untill the fittness no longer improves.


wiki page - https://en.wikipedia.org/wiki/Genetic_algorithm

# Where the Deep comes in
I use a feed forward neural network to determine the steering angle, breaking force, and engine force of the car.
The inputs to this network is distances from rays cast out of the car at varius angles ( this was the exact set of anles - [-40, -20, 0, 20, 40 ] );
However, I found out that the networks can learn extremly well with any set of angles!

# The NN model

input layer - 5 units ( distances of rays as I mentioned earlier - [-40, -20, 0, 20, 40 ] )

dense layer - 5 units in, 4 units out, relu activation -> f(x) = min(max(0, x), 100)

dense layer - 4 units in, 3 units out, relu activation -> f(x) = min(max(0, x), 100)

dense layer - 3 units in, 2 units out, tanh activation -> f(x) =  ( 2.0 / ( 1 + (e ^ (-2 * x / 10.0) ) ) ) - 1

I used modified activation functions because it worked better with the input ( this became imparent when rendering a repersentation of the most fit car to the screen; the nodes were bassically binary, so I adjusted range of values the activation function would accept inorder to learn more of the input space. )

# All of it Together
Each car is given its own copy of the model.
Once the learing process has begins 20 cars with randomly set models are generated.
Then the most fit car is determined by having all 20 of the cars race in a random track. The cars actions are determined by the model.
Finally the next generation is created by mutating the model of the most fit car 19 times. Then the process will repeat untill the fitness begins to stagnate to avoid overfitting.

# Needs
numpy, scipy, pygame

<code> pip3 install numpy, scipy, pygame </code>
# Run
<code> 
cd [ dir of DeepLearningCars.py and barcade-brawl.tff (<-its the font I used) ]

python3 DeepLearningCars.py
</code>
