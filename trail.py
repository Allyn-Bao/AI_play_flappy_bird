import pygame
import neat
import os
import random
import math

""" windows size """
WIN_WIDTH = 570
WIN_HEIGHT = 800


def load_image(image_name):
    return pygame.transform.scale2x((pygame.image.load(os.path.join("imgs", image_name))))


""" load images """
BIRD_IMGS = list(map(load_image, ["bird1.png", "bird2.png", "bird3.png", "bird2.png"]))
PIPE_IMG = load_image("pipe.png")
BASE_IMG = load_image("base.png")
BG_IMG = load_image("bg.png")

""" Font """
pygame.font.init()
SCORE_FONT = pygame.font.SysFont("comicsans", 50)


class Bird:
    IMGS = BIRD_IMGS
    ANIMATION_TIME = 5
    G_FACTOR = 1.5  # gravitational acceleration per tick
    MAX_HEIGHT = 1  # top of the screen
    MIN_HEIGHT = 16  # bottom of the screen
    VELOCITY = -10.5

    def __init__(self, x, y):
        self.x = x
        self.y = y
        self.tilt = 0
        self.tick_count = 0  # tick count for movement
        self.init_velocity = 0
        self.height = self.y
        self.img_count = 0  # tick count for image change
        self.img_index = 0
        self.img = self.IMGS[self.img_index]

    def jump(self):
        self.init_velocity = self.VELOCITY if self.init_velocity == 0 else 0  # inverse direction
        self.tick_count = 0  # reset timer
        self.height = self.y

    def move(self):
        self.tick_count += 1
        # kinetic equation for projectile
        d = self.init_velocity * self.tick_count + self.G_FACTOR * (self.tick_count) ** 2
        d = -0.001 if d == 0 else d
        # bounds
        d = min(self.MIN_HEIGHT, abs(d)) * d/abs(d)
        d = max(self.MAX_HEIGHT, abs(d)) * d/abs(d)
        # vertical movement
        self.y += d
        # adjust tilt
        slope = self.init_velocity + self.G_FACTOR * self.tick_count
        self.tilt = -int(math.degrees(math.atan(slope)))

    def draw(self, win):
        self.img_count += 1
        # for every animation count cycle, update image
        if self.img_count >= self.ANIMATION_TIME:
            # update image index, cycle from index 0 -> 3 -> 0
            self.img_index = (self.img_index + 1) % len(self.IMGS)
            self.img = self.IMGS[self.img_index]
            self.img_count -= self.ANIMATION_TIME
        # tilt
        rotated_image = pygame.transform.rotate(self.img, self.tilt)
        # by default, it rotates around the top-left corner, correct the axis back to center
        new_rect = rotated_image.get_rect(center=self.img.get_rect(topleft=(self.x, self.y)).center)
        win.blit(rotated_image, new_rect.topleft)

    def get_mask(self):
        return pygame.mask.from_surface(self.img)


class Pipe:
    GAP = 200
    VELOCITY = 5

    def __init__(self, x):
        self.x = x
        self.height = 0
        self.gap = 100
        self.y_top = 0
        self.y_bottom = 0
        self.PIPE_TOP = pygame.transform.flip(PIPE_IMG, flip_x=False, flip_y=True)
        self.PIPE_BOTTOM = PIPE_IMG
        self.passed = False
        self.set_height()

    def set_height(self):
        self.height = random.randrange(50, 450)
        self.y_top = self.height - self.PIPE_TOP.get_height()
        self.y_bottom = self.height + self.GAP

    def move(self):
        self.x -= self.VELOCITY

    def draw(self, win):
        win.blit(self.PIPE_TOP, (self.x, self.y_top))
        win.blit(self.PIPE_BOTTOM, (self.x, self.y_bottom))

    def collide(self, bird):
        bird_mask = bird.get_mask()
        top_mask = pygame.mask.from_surface(self.PIPE_TOP)
        bottom_mask = pygame.mask.from_surface(self.PIPE_BOTTOM)

        top_offset = (self.x - bird.x, self.y_top - round(bird.y))
        bottom_offset = (self.x - bird.x, self.y_bottom - round(bird.y))

        # collide points
        top_points = bird_mask.overlap(top_mask, top_offset)
        bottom_point = bird_mask.overlap(bottom_mask, bottom_offset)

        # return true if the lists of colliding point are not none
        return True if (top_points or bottom_point) else False


class Base:
    VELOCITY = 5
    WIDTH = BASE_IMG.get_width()
    IMG = BASE_IMG

    def __init__(self, ground_level):
        self.y = ground_level
        self.x1 = 0
        self.x2 = self.WIDTH

    def move(self):
        self.x1 -= self.VELOCITY
        self.x2 -= self.VELOCITY

        # rotate two ground picture to create seamless infinite ground movement
        if self.x1 + self.WIDTH <= 0:
            self.x1 = self.x2
            self.x2 = self.x1 + self.WIDTH

    def draw(self, win):
        win.blit(self.IMG, (self.x1, self.y))
        win.blit(self.IMG, (self.x2, self.y))


def draw_window(win, birds, pipes, base, score):
    # background
    win.blit(BG_IMG, (0, 0))
    # pipes
    for pipe in pipes:
        pipe.draw(win)
    # bird
    for bird in birds:
        bird.draw(win)
    # base
    base.draw(win)
    # score
    text = SCORE_FONT.render("Score: " + str(score), 1, (255, 255, 255))
    win.blit(text, (WIN_WIDTH - text.get_width() - 10, 10))
    pygame.display.update()


""" main function for the game, also the fitness function for the AI"""


def eval_genomes(genomes, config):
    """
    runs the simulation of the current population of
    birds and sets their fitness based on the distance they
    reach in the game.
    """
    gen = 0
    win = pygame.display.set_mode((WIN_WIDTH, WIN_HEIGHT))
    gen += 1

    # start by creating lists holding the genome itself, the
    # neural network associated with the genome and the
    # bird object that uses that network to play
    nets = []
    birds = []
    ge = []
    for genome_id, genome in genomes:
        genome.fitness = 0  # start with fitness level of 0
        net = neat.nn.FeedForwardNetwork.create(genome, config)
        nets.append(net)
        birds.append(Bird(230,350))
        ge.append(genome)

    base = Base(730)
    pipes = [Pipe(700)]
    score = 0

    clock = pygame.time.Clock()

    run = True
    while run and len(birds) > 0:
        clock.tick(30)

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                run = False
                pygame.quit()
                quit()
                break

        pipe_ind = 0
        if len(birds) > 0:
            if len(pipes) > 1 and birds[0].x > pipes[0].x + pipes[0].PIPE_TOP.get_width():  # determine whether to use the first or second
                pipe_ind = 1                                                                 # pipe on the screen for neural network input

        for x, bird in enumerate(birds):  # give each bird a fitness of 0.1 for each frame it stays alive
            ge[x].fitness += 0.1
            bird.move()

            # send bird location, top pipe location and bottom pipe location and determine from network whether to jump or not
            output = nets[birds.index(bird)].activate((bird.y, abs(bird.y - pipes[pipe_ind].height), abs(bird.y - pipes[pipe_ind].y_bottom)))
            if output[0] > 0.5:  # we use a tanh activation function so result will be between -1 and 1. if over 0.5 jump
                print("bird jumpped")
                bird.jump()

        base.move()

        rem = []
        add_pipe = False
        for pipe in pipes:
            pipe.move()
            # check for collision
            for bird in birds:
                if pipe.collide(bird):
                    ge[birds.index(bird)].fitness -= 1
                    nets.pop(birds.index(bird))
                    ge.pop(birds.index(bird))
                    birds.pop(birds.index(bird))

            if pipe.x + pipe.PIPE_TOP.get_width() < 0:
                rem.append(pipe)

            if not pipe.passed and pipe.x < bird.x:
                pipe.passed = True
                add_pipe = True

        if add_pipe:
            score += 1
            # can add this line to give more reward for passing through a pipe (not required)
            for genome in ge:
                genome.fitness += 5
            pipes.append(Pipe(WIN_WIDTH))

        for r in rem:
            pipes.remove(r)

        for bird in birds:
            if bird.y + bird.img.get_height() - 10 >= 730 or bird.y < -50:
                nets.pop(birds.index(bird))
                ge.pop(birds.index(bird))
                birds.pop(birds.index(bird))

        draw_window(win, birds, pipes, base, score)

        # break if score gets large enough
        '''if score > 20:
            pickle.dump(nets[0],open("best.pickle", "wb"))
            break'''


def run(config_file):
    # load config file
    config = neat.config.Config(neat.DefaultGenome,
                                neat.DefaultReproduction,
                                neat.DefaultSpeciesSet,
                                neat.DefaultStagnation,
                                config_file)
    # create population
    p = neat.Population(config)

    # console data and progress
    p.add_reporter(neat.StdOutReporter(True))
    p.add_reporter(neat.StatisticsReporter())

    # fitness function, 50 generations
    winner = p.run(eval_genomes, 50)

    print('\nBest genome:\n{!s}'.format(winner))


if __name__ == "__main__":
    # get local dir path
    local_dir = os.path.dirname(__file__)
    config_path = os.path.join(local_dir, "config-feedforward.txt")
    run(config_path)
