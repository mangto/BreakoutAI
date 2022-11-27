import sys, numpy, random, copy, pickle
import time as timer
import number, cream

cream.csys.clear()

setting = {c.split("=")[0].replace(" ", ""):eval(c.split("=")[1]) for c in open(".\\data\\setting.txt", 'r', encoding='utf-8').read().splitlines()}

bricks = numpy.ones((6,15))
lastbricks = copy.deepcopy(bricks)
colors = setting["brick_colors"]
score = 0
time = 0
timed = timer.time()
lasttime = 0
bricktime = 0
died = False


# NETWORK SHAPE
# input layer: 95 ( 6*15 bricks + plate position + ball position (x, y) + ball speed (x, y) )
# hidden layer: 30, 15
# output layer: 3 (movement: left, stay, right)
genomes = [cream.reinforce([95, 20, 20, 10, 3], cream.functions.Linear) for _ in range(setting["population"])]


class Plate:
    def __init__(self):
        self.x = 120
        self.size = setting["plate_size"]
        self.speed = setting["plate_speed"]

    def draw(self, keystate):
        # if ("left_arrow" in keystate): self.x -= self.speed
        # if ("right_arrow" in keystate): self.x += self.speed
        key = numpy.argmax(keystate)
        match key:
            case 0: self.x -= self.speed
            case 1: pass
            case 2: self.x += self.speed

        if (self.x-self.size < 0): self.x = self.size
        elif (self.x + self.size > 240): self.x = 240 - self.size

class Ball:
    def __init__(self):
        self.xspeed = 2 if random.random() > 0.5 else -2
        self.yspeed = 2
        self.x = random.randint(10, 230)
        self.y = 240
        self.size = 2

    def draw(self):
        global ball, plate, score, bricks, died
        self.x += self.xspeed
        self.y += self.yspeed

        if (self.y > 432):
            died = True

        # wall collide
        if (self.x - self.size < 0):
            self.xspeed *= -1
            self.x = self.size
        elif (self.x + self.size > 240):
            self.xspeed *= -1
            self.x = 240 - self.size
        if (self.y - self.size < 48):
            self.yspeed *= -1
            self.y = self.size + 48

        # plate collide
        if ((self.y + self.size >=  416 and self.y + self.size <= 424) and (self.x + self.size > plate.x-plate.size and self.x - self.size < plate.x + plate.size)):
            self.yspeed *= -1
            self.y = 416 - self.size

            distance = self.x - plate.x
            if (distance < 0): distance -= 10
            else: distance += 10
            self.xspeed = distance//10
            if (abs(self.xspeed) > 3): self.xspeed = 3 if self.xspeed > 0 else -3
            genomes[index].fitness -= 1

        # brick collide
        collide = False
        for y, axis_x in enumerate(bricks):
            for x, brick in enumerate(axis_x):
                if (brick):
                    if ((self.x - self.size < (x + 1) * 16 ) and 
                            (self.x + self.size > x*16 ) and
                            (self.y - self.size < (y+1)*16 +112) and
                            (self.y + self.size > y*16 +112) and not collide):
                        score += 1
                        genomes[index].fitness += 3
                        bricks[y][x] = 0
                        collide = True

                        if (self.x <= x*16):
                            self.xspeed *= -1
                            # self.x = x*16-self.size
                        elif (self.x >= (x+1)*16):
                            self.xspeed *= -1
                            # self.x = (x+1)*16+self.size
                        elif (self.y <= y*16 ):
                            self.yspeed *= -1
                            # self.y = y*16-self.size+112
                        elif(self.y >= (y+1)*16):
                            self.yspeed *= -1
                            # self.y = (y+1)*16+self.size+112

class system:
    def reset():
        global ball, plate, score, bricks, died, time, bricktime
        ball = Ball()
        plate = Plate()
        score = 0
        bricks = numpy.ones((6,15))
        died = False
        time = 0
        bricktime = 0
    def score():
        pass
    
    def display():
        system.score()

    def event():
        pass

plate = Plate()
ball = Ball()

index = 0
generation = 0

while __name__ == "__main__":
    input_ = numpy.array(list(bricks.flatten()*0.1) + [plate.x/240, ball.x/240, ball.y/432, ball.xspeed/3, ball.yspeed/3])
    genomes[index].forward(input_)
    # if ((fps:=clock.get_fps()) > 0):
    #     time += 1/fps * (setting["realframe"]/setting["frame"])
    #     bricktime += 1/fps * (setting["realframe"]/setting["frame"])
    time += timer.time() - timed
    bricktime += timer.time() - timed
    timed = timer.time()
    if (time - lasttime >= 1):
        lasttime = time
    if (bricks != lastbricks).any():
        bricktime = 0
        lastbricks = copy.deepcopy(bricks)
    if (bricktime >= 3):
        died = True
        print("time out")
        bricktime = 0
    if (died):
        # print(f"died! | {index=:>5} | fitness={genomes[index].fitness:>5}")
        index += 1
        system.reset()

        if (index == len(genomes)):
            # leave only best genomes
            index = 0
            generation += 1
            genomes.sort(key=lambda x:x.fitness, reverse=True)
            bestgenomes = copy.deepcopy(genomes[:setting["parents"]])
            
            # cross over
            print(f"cross over! | {generation=:>4}")
            print(f"{'best fitness: '+str(bestgenomes[0].fitness):>4}")
            pickle.dump(genomes[0].weights, open(".\\data\\weights.pkl", 'wb')) # save network
            pickle.dump(genomes[0].biases, open(".\\data\\biases.pkl", 'wb')) # save network 
            for _ in range(setting["children"]):
                new = copy.deepcopy(bestgenomes[0])
                # genome_a = random.choice(bestgenomes)
                # genome_b = random.choice(bestgenomes)

                # for i in range(new.depth - 1):
                #     cut = random.randint(0, len(new.weights[i]))
                #     new.weights[i][0:cut] = genome_a.weights[i][0:cut]
                #     new.weights[i][cut:-1] = genome_b.weights[i][cut:-1]

                bestgenomes.append(new)

            genomes = copy.deepcopy(bestgenomes)
            for i, genome in enumerate(genomes):
                for w, weight in enumerate(genome.weights):
                    multiplier = numpy.random.normal(size = numpy.array(weight).shape)
                    numpy.random.shuffle(multiplier)
                    genome.weights[w] += numpy.array(weight) * multiplier * 0.1

                genome.fitness = 0
        if (bricks.flatten() ==numpy.zeros((90,))).all():
            genomes[index].fitness += 10
            died = True
    system.display()
    system.event()
    plate.draw(genomes[index].activ[-1])
    ball.draw()