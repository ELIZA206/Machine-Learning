import pygame
from sklearn.cluster import DBSCAN
import random
pygame.init()
points = []
flags = []
r = 10
colors = [(0, 0, 255), (0, 255, 0), (127, 0, 255), (255, 255, 0), (255, 0, 0)]
minPts, eps = 4, 3*r
screen = pygame.display.set_mode((400, 300))
screen.fill('white')
done = True
while done:
        for event in pygame.event.get():
                if event.type == pygame.QUIT:
                        done = False
                if event.type == pygame.MOUSEBUTTONDOWN:
                    if event.button == 1:
                        point = pygame.mouse.get_pos()
                        points.append(point)
                        pygame.draw.circle(screen,'black',point,r)
                if event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_RETURN:
                        # Выдать точкам флаги
                        for point in points:
                            if random.random() < 0.33:
                                flags.append('green')
                            elif random.random() < 0.66:
                                flags.append('yellow')
                            else:
                                flags.append('red')
                    if event.key == pygame.K_r:
                        dbscan = DBSCAN(eps=eps, min_samples=minPts)
                        dbscan.fit(points)
                        labels = dbscan.labels_
                        print(labels)
                        for i in range(len(points)):
                            if len(colors) in labels:
                                colors.insert(0,(random.randint(0,255), random.randint(0,255), random.randint(0,255)))
                            pygame.draw.circle(screen,colors[labels[i]],points[i],r)
                        print(colors)
        pygame.display.flip()
pygame.quit()