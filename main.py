import networkx as nx
import matplotlib.pyplot as plt
import time
import csv
import random
import pandas as pd
import numpy as np
from itertools import permutations

def readFile(min_height):
  filepath = "./input.csv"
  data = pd.read_csv(filepath)
  data = data[data["Elevation_meters"] >= min_height]
  data = data.reset_index()
  return data

def convertToPositions(data):
  positions = {}
  for index, row in data.iterrows():
    longitude = row["Longitude"]
    latitude = row["Latitude"]
    positions[index] = (longitude, latitude)
  return positions

def calcDistance(positions, p1, p2):
  x1, y1 = positions[p1]
  x2, y2 = positions[p2]
  return ((x1 - x2) ** 2 + (y1 - y2) ** 2) ** 0.5

def calcTotalCost(positions, route):
  return sum(calcDistance(positions, x1, x2) for x1, x2 in zip(route, route[1:]+route[:1]))

def createGraph(positions):
  G = nx.Graph()
  G.add_nodes_from(positions.keys())
  for i in positions:
    for j in positions:
      if i < j:
        G.add_edge(i, j, weight = calcDistance(positions, i, j))
  return G

def solveRandom(G, positions):
  start = next(iter(positions))
  route = list(positions)
  route.remove(start)
  random.shuffle(route)
  route.insert(0, start)
  route.append(start)
  return route

def solveBruteForce(G, positions):
  start = next(iter(positions))
  positions_copy = positions.copy()
  del positions_copy[start]
  min_route = []
  min_total_cost = -1
  route_permutations = list(permutations(positions_copy))
  for route in route_permutations:
    route = list(route)
    route.insert(0, start)
    route.append(start)
    if min_total_cost == -1 or calcTotalCost(positions, route) < min_total_cost:
      min_route = route
      min_total_cost = calcTotalCost(positions, min_route)
  return min_route

def solveNearestNeighbor(G, positions):
  current = next(iter(positions))
  unvisited = list(G.neighbors(current))
  result = [current]

  while len(unvisited) > 0:
    min_weight = -1
    min_place = -1
    for place in unvisited:
      weight = G.get_edge_data(current, place)["weight"]
      if min_weight == -1 or weight < min_weight:
        min_weight = weight
        min_place = place
    current = min_place
    result.append(min_place)
    unvisited.remove(min_place)
  result.append(next(iter(positions)))
  return result

def solveTwoOpt(G, route):
  n = len(route) - 1
  route.pop(n)

  count = 1
  while count > 0:
    count = 0
    for i in range(n - 2):
      for j in range(i + 2, n - (i == 0)):
        p1 = G.get_edge_data(route[i], route[i + 1])["weight"]
        p2 = G.get_edge_data(route[j], route[(j + 1) % n])["weight"]
        p3 = G.get_edge_data(route[i], route[j])["weight"]
        p4 = G.get_edge_data(route[i + 1], route[(j + 1) % n])["weight"]

        if p1 + p2 > p3 + p4:
          new_route = route[i + 1 : j + 1]
          route[i + 1 : j + 1] = new_route[:: -1]
          count += 1
  route.append(route[0])
  return route

def solveThreeOpt(G, route):
  n = len(route) - 1
  route.pop(n)

  count = 1
  while count > 0:
    count = 0
    for i in range(n - 4):
      for j in range(i + 2, n - 2):
        for k in range(j + 2, n - (i == 0)):
          d1 = G.get_edge_data(route[i], route[i + 1])["weight"] + G.get_edge_data(route[j], route[j + 1])["weight"] + G.get_edge_data(route[k], route[(k + 1) % n])["weight"]
          d2 = G.get_edge_data(route[i], route[j])["weight"] + G.get_edge_data(route[i + 1], route[j + 1])["weight"] + G.get_edge_data(route[k], route[(k + 1) % n])["weight"]
          d3 = G.get_edge_data(route[i], route[i + 1])["weight"] + G.get_edge_data(route[j], route[k])["weight"] + G.get_edge_data(route[j + 1], route[(k + 1) % n])["weight"]
          d4 = G.get_edge_data(route[i], route[k])["weight"] + G.get_edge_data(route[j], route[j + 1])["weight"] + G.get_edge_data(route[i + 1], route[(k + 1) % n])["weight"]
          d5 = G.get_edge_data(route[i], route[j + 1])["weight"] + G.get_edge_data(route[i + 1], route[k])["weight"] + G.get_edge_data(route[j], route[(k + 1) % n])["weight"]

          if d1 > d2:
            new_route = route[i + 1 : j + 1]
            route[i + 1 : j + 1] = new_route[:: -1]
          elif d1 > d3:
            new_route = route[j + 1 : k + 1]
            route[j + 1 : k + 1] = new_route[:: -1]
          elif d1 > d4:
            new_route = route[i + 1 : k + 1]
            route[i + 1 : k + 1] = new_route[:: -1]
          elif d1 > d5:
            new_route = route[j + 1 : k + 1] + route[i + 1 : j + 1]
            route[i + 1 : k + 1] = new_route
          else:
            continue
          count += 1
  route.append(route[0])
  return route

def solveOneOrOpt(G, route):
  n = len(route) - 1
  route.pop(n)

  count = 1
  while count > 0:
    
    count = 0
    for i in range(n):
      i_0 = i
      i_1 = (i + 1) % n
      i_2 = (i + 2) % n
      for j in range(n):
        j_0 = j
        j_1 = (j + 1) % n
        if j_0 != i_0 and j_0 != i_1:
          p1 = G.get_edge_data(route[i_0], route[i_1])["weight"]
          p2 = G.get_edge_data(route[i_1], route[i_2])["weight"]
          p3 = G.get_edge_data(route[j_0], route[j_1])["weight"]
          p4 = G.get_edge_data(route[i_0], route[i_2])["weight"]
          p5 = G.get_edge_data(route[j_0], route[i_1])["weight"]
          p6 = G.get_edge_data(route[i_1], route[j_1])["weight"]
          if p1 + p2 + p3 > p4 + p5 + p6:
            i_1_node = route.pop(i_1)
            if i_1 < j_1:
              route.insert(j_0, i_1_node)
            else:
              route.insert(j_1, i_1_node)
            count += 1
  route.append(route[0])
  return route

def solveTwoOrOpt(G, route):
  n = len(route) - 1
  route.pop(n)

  count = 1
  while count > 0:
    
    count = 0
    for i in range(n):
      i_0 = i
      i_1 = (i + 1) % n
      i_2 = (i + 2) % n
      i_3 = (i + 3) % n
      for j in range(n):
        j_0 = j
        j_1 = (j + 1) % n
        if j_0 != i_0 and j_0 != i_1 and j_0 != i_2:
          p1 = G.get_edge_data(route[i_0], route[i_1])["weight"]
          p2 = G.get_edge_data(route[i_2], route[i_3])["weight"]
          p3 = G.get_edge_data(route[j_0], route[j_1])["weight"]
          p4 = G.get_edge_data(route[i_0], route[i_3])["weight"]
          p5 = G.get_edge_data(route[j_0], route[i_1])["weight"]
          p6 = G.get_edge_data(route[i_2], route[j_1])["weight"]
          if p1 + p2 + p3 > p4 + p5 + p6:
            if i_2 < i_1:
              i_1_node = route.pop(i_1)
              i_2_node = route.pop(i_2)
              route.insert(j_0, i_2_node)
              route.insert(j_0, i_1_node)
            else:
              i_2_node = route.pop(i_2)
              i_1_node = route.pop(i_1)
              if i_1 < j_1:
                route.insert(j_0 - 1, i_2_node)
                route.insert(j_0 - 1, i_1_node)
              else:
                route.insert(j_1, i_2_node)
                route.insert(j_1, i_1_node)
            count += 1
  route.append(route[0])
  return route

def plot(G, positions, result):
  nx.draw(G, pos = positions, node_size = 5, node_color = 'k', edge_color=(0, 0, 0, 0.001), with_labels = False)
  nx.draw_networkx_edges(G, pos = positions, edgelist = [(u, v) for u, v in zip(result, result[1:] + result[:1])], edge_color = 'r')
  plt.show()

def main():
  data = readFile(min_height = -10000)
  positions = convertToPositions(data)
  G = createGraph(positions)
  
  print(data)

  time_start = time.perf_counter()
  
  # 力任せ法
  #result = solveBruteForce(G, positions)
  # 最近傍法
  result = solveNearestNeighbor(G, positions)
  # ランダム法
  #result = solveRandom(G, positions)

  # 2-opt法
  result = solveTwoOpt(G, result)
  
  # 3-opt法
  #result = solveThreeOpt(G, result)
  
  # or-1-opt法
  #result = solveOneOrOpt(G, result)
  
  # or-2-opt法
  #result = solveTwoOrOpt(G, result)
  
  time_end = time.perf_counter()
  time_range = time_end - time_start

  print("最短距離の順序: {}".format(result))
  print("最短距離: {}".format(calcTotalCost(positions, result)))
  print("時間: {} [sec]".format(time_range))

  plot(G, positions, result)

if __name__ == "__main__":
  main()
