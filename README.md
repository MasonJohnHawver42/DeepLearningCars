# DeepLearningCars
Using deep genetic learning cars are able race around a randomly generated track.

# Genetic Learnig
Genetic learning is an algorithim used to generate to most fit entity. 
For this to work two things must be implemented a way of determining the fitest ( or best ) entity and a way to mutate that entity ( ie create a random entity based off a parent entity).
Now the algorithim starts by creating random entitties based on no parents. 
Then it determines the most fit entity and creates the next generation based upon that entity.
This last step is repeated untill the fittness no longer improves.


wiki page - https://en.wikipedia.org/wiki/Genetic_algorithm

# Where the Deep comes in
I use a feed forward neural network to determine the steering angle, breaking force, and engine force of the car.
The inputs to this network is distances from rays cast out of the car at varius angles ( this was the exact set of anles - [-45, -22, -11, 0, 11, 22, 45] );
However, I found out that the networks can learn extremly well with any set of angles!

# The NN model

input layer - 7 units ( distances of rays as I mentioned earlier - [-45, -22, -11, 0, 11, 22, 45] )

dense layer - 7 units in, 6 units out, relu activation

dense layer - 6 units in, 4 units out, relu activation

dense layer - 4 units in, 2 units out, tanh activation ( tanh outputs between -1 and 1, so one output is the steering amt and the positive of the other is the amt of engine force and the negetive of the second output is the breaking force )

Again, when playing around with weird input and ouput sizes of the dense layers the cars still leqarned to drive just fine!

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
cd [dir of Auto.py]

python3 Auto.py
</code>
