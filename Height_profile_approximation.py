from csv import reader

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.pyplot import figure
import time

def readCsv(fileName, samples_x, samples_y):
    with open('2018_paths/'+fileName, "r") as read_obj:
        csv_reader = reader(read_obj)
        # take x and y from the .csv file and write to the sample
        for row in csv_reader:
            if row[0][0:7] != "Dystans":
                samples_x.append(float(row[0]))
                samples_y.append(float(row[1]))

    read_obj.close()


def readTxtAndData(fileName, samples_x, samples_y):
    with open('2018_paths/'+fileName, 'r') as file:
        # take x and y from the .txt or .data file and write to the sample lists
        for row in file.readlines():
            if fileName[-4:] == ".txt" and row[0:7] != "Dystans":
                samples_x.append(float(row.split(' ')[0]))
                samples_y.append(float(row.split(' ')[1]))
            elif fileName[-5:] == ".data" and row[0:7] != "Dystans":
                samples_x.append(float(row.split(',')[0]))
                samples_y.append(float(row.split(',')[1]))

        file.close()


def pivoting(A, b, N):
    L = np.eye(N)
    P = np.eye(N)     # permutation matrix
    U = A

    for k in range(N - 1):
        # find index of max element
        abs_U = np.absolute(U[k:])
        max_index = abs_U.argmax(axis=0)
        index = max_index[k] + k

        # interchange rows
        U[[k, index], k:N] = U[[index, k], k:N]  # GIT
        L[[k, index], :k] = L[[index, k], :k]
        P[[k, index]] = P[[index, k]]

        # standard LU
        for j in range(k + 1, N):
            L[j][k] = U[j][k] / U[k][k]
            U[j][k:N] = U[j][k:N] - L[j][k] * U[k][k:N]

    # x = (U \ (L \ (P * b))) LU after pivot
    x = np.linalg.solve(U, (np.linalg.solve(L, P.dot(b))))

    return x


def LagrangeInterpolation(nodes_x, nodes_y, distance, nodes_num):
    result = 0.0

    for i in range(nodes_num):
        fi = 1.0
        # fi(x) = (n+1) ∏ (j=1, j!=i) = (x−xj)/(xi−xj)
        for j in range(nodes_num):
            if i != j:
                fi *= (distance - nodes_x[j]) / (nodes_x[i] - nodes_x[j])

        result += (fi * nodes_y[i])

    return result


def SplinesInterpolation(nodes_x, nodes_y, distance, nodes_num):
    # 4 * (n - 1) equations
    N = 4 * (nodes_num - 1)

    A = np.zeros((N, N))
    b = np.zeros(N)

    # S0(x0) = f(x0) -> a0 = f(x0)
    A[0][0] = 1
    b[0] = nodes_y[0]

    # h = xi+1 - xi
    h = nodes_x[1] - nodes_x[0]

    # S0(x1) = f(x1) -> a0 + b0 * h + c0 * h^2 + d0 * h^3 = f(x1)
    A[1][0] = 1
    A[1][1] = h
    A[1][2] = pow(h, 2)
    A[1][3] = pow(h, 3)
    b[1] = nodes_y[1]

    # S''0(x0) = 0 -> c0 = 0
    A[2][2] = 1
    b[2] = 0

    # h = xn - xn-1
    h = nodes_x[nodes_num - 1] - nodes_x[nodes_num - 2]

    # S''n−1(xn) = 0 -> 2 * cn-1 + 6 * dn-1 * h = 0
    A[3][4 * (nodes_num - 1) - 2] = 2
    A[3][4 * (nodes_num - 1) - 1] = 6 * h
    b[3] = 0

    for i in range(1, nodes_num - 1):
        # h = xi+1 - xi
        h = nodes_x[i] - nodes_x[i-1]

        # Si(xi) = f(xi) -> ai = f(xi)
        A[4 * i][4 * i] = 1
        b[4 * i] = nodes_y[i]

        # Si(xi+1) = f(xi+1) -> ai + bi * h + ci * h^2 + di * h^3 = f(xi+1)
        A[4 * i + 1][4 * i] = 1
        A[4 * i + 1][4 * i + 1] = h
        A[4 * i + 1][4 * i + 2] = pow(h, 2)
        A[4 * i + 1][4 * i + 3] = pow(h, 3)
        b[4 * i + 1] = nodes_y[i + 1]

        # S'i−1(xi) = S'i(xi) -> bi-1 + 2ci-1 * h + 3 * di-1 * h^2 - bi = 0
        A[4 * i + 2][4 * (i - 1) + 1] = 1
        A[4 * i + 2][4 * (i - 1) + 2] = 2 * h
        A[4 * i + 2][4 * (i - 1) + 3] = 3 * pow(h, 2)
        A[4 * i + 2][4 * i + 1] = -1
        b[4 * i + 2] = 0

        # S''i−1(xi) = S''i(xi) -> 2 * ci-1 + 6 * di-1 * h - 2 * ci = 0
        A[4 * i + 3][4 * (i - 1) + 2] = 2
        A[4 * i + 3][4 * (i - 1) + 3] = 6 * h
        A[4 * i + 3][4 * i + 2] = -2
        b[4 * i + 3] = 0

    x = pivoting(A, b, N)

    for i in range(nodes_num - 1):
        result = 0.0

        if distance >= nodes_x[i] and distance <= nodes_x[i+1]:
            h = distance - nodes_x[i]
            for j in range(4):
                result += x[4 * i + j] * pow(h, j)

            break

    return result


def createPlotForMethod(samples_x, samples_y, nodes_x, nodes_y, results_x, results_y, name, nodes_num, method_name, color):
    figure(num=None, figsize=(10, 7), dpi=60)
    plt.plot(samples_x, samples_y, color='black', label="Actual values")
    plt.scatter(nodes_x, nodes_y, color=color, label="Interpolation nodes ")
    plt.plot(results_x, results_y, color=color, label=method_name + " method")
    plt.legend(loc='upper center', bbox_to_anchor=(0.5, -0.075), fancybox=True, shadow=True, ncol=5)
    plt.title(method_name + " method in " + str(name) + " for " + str(nodes_num) + " nodes")
    plt.xlabel("distance [m]")
    plt.ylabel("height [m]")

    if name[-4:] == ".csv" or name[-4:] == ".txt":
        plt.savefig("results/" + method_name + "/" + name[:-4] + "_" + method_name + "_" + str(nodes_num) + '_nodes.png')
    elif name[-5:] == ".data":
        plt.savefig("results/" + method_name + "/" + name[:-5] + "_" + method_name + "_" + str(nodes_num) + '_nodes.png')

    plt.close()


def createPlotForTimes(nodes_num_Lagrange, times_Lagrange, nodes_num_Splines, times_Splines, name):
    figure(num=None, figsize=(10, 6), dpi=60)
    plt.plot(nodes_num_Lagrange, times_Lagrange, color='blue', label="Lagrange time")
    plt.plot(nodes_num_Splines, times_Splines, color='orange', label="Splines time")
    plt.legend(loc="upper left")
    plt.title("Times for Lagrange and Splines in " + str(name))
    plt.xlabel("nodes number")
    plt.ylabel("time [s]")

    if name[-4:] == ".csv" or name[-4:] == ".txt":
        plt.savefig("results/Times/" + name[:-4] + "_Times.png")
    elif name[-5:] == ".data":
        plt.savefig("results/Times/" + name[:-5] + "_Times.png")

    plt.close()



# files with data on which the interpolation will be performed
filesName = ["MountEverest.csv", "genoa_rapallo.txt", "WielkiKanionKolorado.csv", "GlebiaChallengera.csv"]
# the maximum number of designated distances in interpolating methods
max_num_designated_distances = 200

# the length of the intervals, how many nodes will be
# 9 nodes, 17 nodes, 33 nodes because there are 512
# samples in the data
intervals = [64, 32, 16]
samples_num = 512

for name in filesName:
    samples_x = []
    samples_y = []
    if name[-4:] == ".csv":
        readCsv(name, samples_x, samples_y)
    elif name[-4:] == ".txt" or name[-5:] == ".data":
        readTxtAndData(name, samples_x, samples_y)

    # variables to represent the time differences of methods
    times_Lagrange = []
    nodes_num_Lagrange = []
    times_Splines = []
    nodes_num_Splines = []

    for i in range(len(intervals)):
        # number of nodes
        nodes_num = int(samples_num/intervals[i]) + 1
        # value of the interval allowing for the transition to the next node
        interval = intervals[i]
        # variables to create nodes
        nodes_x = []
        nodes_y = []
        index = 0

        # creating nodes
        for j in range(nodes_num):
            # last node on last sample
            if index == samples_num:
                nodes_x.append(samples_x[index-1])
                nodes_y.append(samples_y[index-1])
            else:
                nodes_x.append(samples_x[index])
                nodes_y.append(samples_y[index])
                index += interval

        # variables to interpolation data
        results_Lagrange_x = []
        results_Lagrange_y = []
        results_Splines_x = []
        results_Splines_y = []
        change_of_distance = float(nodes_x[nodes_num - 1]/max_num_designated_distances)

        # getting results for interpolation methods
        t0 = time.time()
        for distance in np.arange(0.0, nodes_x[nodes_num - 1], change_of_distance):
            results_Lagrange_x.append(distance)
            results_Lagrange_y.append(LagrangeInterpolation(nodes_x, nodes_y, distance, nodes_num))

        t1 = time.time()
        times_Lagrange.append(t1 - t0)
        nodes_num_Lagrange.append(nodes_num)

        t0 = time.time()
        for distance in np.arange(0.0, nodes_x[nodes_num - 1], change_of_distance):
            results_Splines_x.append(distance)
            results_Splines_y.append(SplinesInterpolation(nodes_x, nodes_y, distance, nodes_num))

        t1 = time.time()
        times_Splines.append(t1 - t0)
        nodes_num_Splines.append(nodes_num)

        createPlotForMethod(samples_x, samples_y, nodes_x, nodes_y, results_Lagrange_x, results_Lagrange_y, name, nodes_num, "Lagrange", "red")
        createPlotForMethod(samples_x, samples_y, nodes_x, nodes_y, results_Splines_x, results_Splines_y, name, nodes_num, "Splines", "green")

        print("Plots for Lagrange and Splines methods in", name, "for", nodes_num, "nodes have been created.")

    print("\nResults for", name)
    print("nodes_num_Lagrange = ", nodes_num_Lagrange)
    print("times_Lagrange = ", times_Lagrange)
    print("nodes_num_Splines = ", nodes_num_Splines)
    print("times_Splines = ", times_Splines)

    createPlotForTimes(nodes_num_Lagrange, times_Lagrange, nodes_num_Splines, times_Splines, name)

    print("\nAll plots for", name, "have been created. \n")