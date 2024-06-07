import cv2
import os
import math
import random
import numpy as np
from tqdm import tqdm
import pickle

def bintodec(binarr):
    num = 0
    for i in range(len(binarr)):
        num += binarr[i] * (2**i)
    return num

class GA:
    def __init__(self, path, population_size) -> None:
        self.path = path
        self.min_x = float("inf")
        self.min_y = float("inf")
        self.original_pixel_size = 0
        self.x_gene_size = 0
        self.y_gene_size = 0
        self.population = []
        self.population_size = population_size
        self.find_min_size()
        self.generate_init_population()

    def find_min_size(self):
        for p in os.listdir(self.path):
            if p.split('.')[-1] == 'gif':
                ret, img = cv2.VideoCapture(f"{self.path}/{p}").read()
                img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                self.min_x = min(img.shape[1], self.min_x)
                self.min_y = min(img.shape[0], self.min_y)
                self.original_pixel_size += (img.shape[0] * img.shape[1])
                del img
            elif p.split('.')[-1] == 'png':
                img = cv2.imread(f"{self.path}/{p}")
                img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                self.min_x = min(img.shape[1], self.min_x)
                self.min_y = min(img.shape[0], self.min_y)
                self.original_pixel_size += (img.shape[0] * img.shape[1])
                del img
    
    def generate_init_population(self):
        self.x_gene_size = math.ceil(math.log2(self.min_x))
        self.y_gene_size = math.ceil(math.log2(self.min_y))
        for _ in range(int(self.population_size/2)):
            temp_arr = [random.randint(0,1) for _ in range(self.x_gene_size + self.y_gene_size)]
            self.population.append(temp_arr)
    
    def breeding(self):
        for i in range(int(self.population_size/4)):
            crosspoint = random.randint(1,(self.x_gene_size + self.y_gene_size - 1))
            restsize = (self.x_gene_size + self.y_gene_size) - crosspoint
            offspring1 = []
            offspring2 = []
            for j in range(crosspoint):
                offspring1.append(self.population[i][j])
                offspring2.append(self.population[i + int(self.population_size/4)][j])
            for j in range(restsize):
                offspring1.append(self.population[i + int(self.population_size/4)][crosspoint+j])
                offspring2.append(self.population[i][crosspoint+j])
            mutation_index = random.randint(0,(self.x_gene_size + self.y_gene_size - 1))
            offspring1[mutation_index] = 1 - offspring1[mutation_index]
            self.population.append(offspring1)
            mutation_index = random.randint(0,(self.x_gene_size + self.y_gene_size - 1))
            offspring2[mutation_index] = 1 - offspring2[mutation_index]
            self.population.append(offspring2)

    
    def CAC(self, img: np.array, kernelx, kernely):
        ret, img = cv2.threshold(img, 50, 255, cv2.THRESH_BINARY)
        sum = 0
        zero_count = 0
        one_count = 0
        mix_count = 0
        mix_size = 0
        x_chunks = math.ceil(img.shape[1] / kernelx)
        y_chunks = math.ceil(img.shape[0] / kernely)
        for y_space in range(y_chunks):
            y_start = y_space * kernely
            y_end = (y_space+1) * kernely
            if y_space == (y_chunks-1):
                y_end = img.shape[0]
            for x_space in range(x_chunks):
                x_start = x_space * kernelx
                x_end = (x_space+1) * kernelx
                if x_space == (x_chunks-1):
                    x_end = img.shape[1]
                chunk = img[y_start:y_end, x_start:x_end]
                if len(np.unique(chunk)) == 2:
                    mix_count += 1
                    mix_size += (chunk.shape[0]*chunk.shape[1])
                else:
                    if np.unique(chunk)[0] == 0:
                        zero_count += 1
                    else:
                        one_count += 1
        counts = [zero_count, one_count, mix_count]
        max = np.argmax(counts)
        for i, count in enumerate(counts):
            if i == max:
                sum += count
            else:
                sum += (count*2)
        return sum + mix_size

    def decode(self, offspring):
        kernelxbin = offspring[:self.x_gene_size]
        kernelybin = offspring[self.x_gene_size:]
        kernelx = bintodec(kernelxbin)
        kernely = bintodec(kernelybin)
        return kernelx, kernely

    def fitness(self, offspring):
        kernelx, kernely = self.decode(offspring)
        if kernelx == 0 or kernely == 0 or kernelx > self.min_x or kernely > self.min_y:
            return 0
        sum = 0
        for p in os.listdir(self.path):
            if p.split('.')[-1] == 'gif':
                ret, img = cv2.VideoCapture(f"{self.path}/{p}").read()
            elif p.split('.')[-1] == 'png':
                img = cv2.imread(f"{self.path}/{p}")
            else:
                continue
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            sum += self.CAC(img, kernelx, kernely)
            del img
        return self.original_pixel_size / sum
    
    def selection(self):
        fitness_scores = dict()
        for i, off in enumerate(self.population):
            fitness = self.fitness(off)
            fitness_scores.update({i:fitness})
        fitness_scores = dict(sorted(fitness_scores.items(), key=lambda item: item[1]))
        mid = -1 * int(self.population_size/2)
        passed_off_index = list(fitness_scores.keys())[mid:]
        passed_off = []
        for i in passed_off_index:
            passed_off.append(self.population[i])
        self.population = passed_off

    def train(self, epochs):
        for _ in tqdm(range(epochs)):
            self.breeding()
            self.selection()
            x,y = self.decode(self.population[-1])
            print(f"\nBest Kernel this Epoch: x={x}, y={y}")
    
    def save_population(self, popstr):
        pickle.dump(self.population, open(popstr,'wb'))

    def load_population(self, popstr):
        self.population = pickle.load(open(popstr,'rb'))

    

if __name__ == '__main__':
    ga = GA("reducedDS", 8)
    ga.train(30)


