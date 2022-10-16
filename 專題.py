import numpy as np
import math

floor = 10 - 1
ONE_FLOOR_COST = 5
np.random.seed(0)

people_arr = np.random.randint(low=0, high=8, size=floor)
people = np.sum(people_arr)
print(people_arr)
print(people)
# while (people>24):
# people_arr=np.random.randint(low=0, high=8,size=9)
# people=np.sum(people_arr)
print(people_arr)
print(people)

print(np.argwhere(people_arr > 0).reshape(-1))

people_position = np.argwhere(people_arr > 0).reshape(-1)

location = []
for i in people_position:
    for j in range(people_arr[i]):
        location.append(i + 1)
location = np.array(location)
print(location)

location = np.insert(location, 0, 0)
print("Location", location)
print(location.shape)
dist = np.zeros((people + 1, people + 1))
for i in range(people + 1):
    for j in range(people + 1):
        if i == j:
            continue
        dist[i][j] = (np.abs(location[j] - location[i])) * ONE_FLOOR_COST
print(dist)

PEOPLE = people
NUM_ELEVATOR = 2
NUM_ITERATION = 400

NUM_CHROME = 100

NUM_BIT = PEOPLE + 1 + (people // 24 * 2)
PETALTY = 1e+4
DOOR_TIME = 3

Pc = 4
Pm = 0.01
NUM_PARENT = NUM_CHROME  # 父母的個數
NUM_CROSSOVER = int(Pc * NUM_CHROME / 2)  # 交配的次數
NUM_CROSSOVER_2 = NUM_CROSSOVER * 2  # 上數的兩倍
NUM_MUTATION = int(Pm * NUM_CHROME * NUM_BIT)  # 突變的次數


def initPop():
    chrome = np.zeros((0, NUM_BIT))
    for i in range(NUM_CHROME):
        a = np.random.permutation(range(1, (PEOPLE + 1)))
        for j in range(NUM_BIT - PEOPLE):
            a = np.insert(a, np.random.randint(len(a) + 1), 0)
        chrome = np.vstack((chrome, a))

    return chrome


def fitFunc(chrome_i):
    v_dist = []  # 用以紀錄4輛車的路徑長
    v = []
    pre_city = 0  # 用以紀錄4輛車的目前位置
    route = []
    dis = 0
    e2 = []
    for i in range(NUM_BIT):
        if (chrome_i[i] == 0):
            if len(route) > 12:
                dis += PETALTY
            v.append(route)
            v_dist.append(dis + dist[pre_city][0])
            route = []
            dis = 0
            pre_city = 0
            continue
        route.append(int(chrome_i[i]))
        if dist[pre_city][int(chrome_i[i])] == 0:
            add = 0
        else:
            add = DOOR_TIME
        dis += dist[pre_city][int(chrome_i[i])] + add
        pre_city = int(chrome_i[i])

    if route != []:
        if (len(route) > 12):
            dis += PETALTY
        v.append(route)
        v_dist.append(dis + dist[pre_city][0])
    ans = v_dist[:2]
    e1 = [v[0]]
    if len(v) > 1:
        e2 = [v[1]]
        for i in range(2, len(v_dist)):
            if (ans[0] < ans[1]):
                ans[0] += v_dist[i]
                e1.append(v[i])
            else:
                ans[1] += v_dist[i]
                e2.append(v[i])

    return -max(ans), e1, e2  # 因為是最小化問題


def evaluatePop(p):
    min_max = []
    ela_1 = []
    ela_2 = []
    for i in range(len(p)):
        minimum, e1, e2 = fitFunc(p[i])
        min_max.append(minimum)
        ela_1.append(e1)
        ela_2.append(e2)
    return min_max, ela_1, ela_2


def selection(p, p_fit):  # 用二元競爭式選擇法來挑父母
    # p1=p1.tolist()
    # p2=p2.tolist()
    a = []

    for i in range(NUM_PARENT):
        [j, k] = np.random.choice(NUM_CHROME, 2, replace=False)
        if p_fit[j] > p_fit[k]:  # 擇優
            a.append(p[j].copy())
        else:
            a.append(p[k].copy())

    return a


def roulette_wheel(p, p_fit):   # 輪盤競爭式選擇法
    wheel_proportion = []
    fit_sum = np.sum(p_fit)
    for i in range(NUM_CHROME):
        wheel_proportion.append(p_fit[i]/fit_sum)

    wheel = []
    sum_ = 0
    for i in range(NUM_CHROME):
        sum_ += int(wheel_proportion[i] * 100)
        for j in range(int(wheel_proportion[i] * 100)):
            wheel.append(p[i].copy())

    a = []
    for i in range(NUM_PARENT):
        k = np.random.choice(sum_, 1, replace=False)
        a.append(wheel[k[0]].copy())
    return a


def larger_tournament(p, p_fit):
    a = []
    for i in range(NUM_PARENT):
        group = np.random.choice(NUM_CHROME, 10, replace=False)
        max = 0
        index = 0
        first = True
        for j in range(10):
            if first:
                max = p_fit[group[j]]
                index = group[j]
                first = False
            elif p_fit[group[j]] > max:
                max = p_fit[group[j]]
                index = group[j]
        a.append(p[index].copy())
    return a

def probabilistic_selection(p, p_fit):
    a = []
    for i in range(NUM_PARENT):
        [j, k] = np.random.choice(NUM_PARENT, 2, replace=False)
        sum = p_fit[j] + p_fit[k]
        wheel = []
        [j_num, k_num] = [int((p_fit[j]/sum)*100) , int((p_fit[k]/sum)*100)]
        for i in range(j_num):
            wheel.append(p[j].copy())
        for i in range(k_num):
            wheel.append(p[k].copy())

        k = np.random.randint(len(wheel))
        a.append(wheel[k])

    return a


def crossover_single(p):    # single point crossover
    a = []
    for i in range(NUM_CROSSOVER):
        c = np.random.randint(1, NUM_BIT)
        [j, k] = np.random.choice(NUM_PARENT, 2, replace=False)

        a.append(np.concatenate((p[j][0: c], p[k][c: NUM_BIT]), axis=0))
        a.append(np.concatenate((p[k][0: c], p[j][c: NUM_BIT]), axis=0))

    return a


def crossover_uniform(p):  # 用均勻交配來繁衍子代
    a = []
    for i in range(NUM_CROSSOVER):
        mask = np.random.randint(2, size=NUM_BIT)

        [j, k] = np.random.choice(NUM_PARENT, 2, replace=False)  # 任選兩個index

        child1, child2 = p[j].copy(), p[k].copy()
        remain1, remain2 = list(p[j].copy()), list(p[k].copy())  # 存還沒被用掉的城市

        for m in range(NUM_BIT):
            if mask[m] == 1:
                remain2.remove(child1[m])  # 砍掉 remain2 中的值是 child1[m]
                remain1.remove(child2[m])  # 砍掉 remain1 中的值是 child2[m]

        t = 0
        for m in range(NUM_BIT):
            if mask[m] == 0:
                child1[m] = remain2[t]
                child2[m] = remain1[t]
                t += 1

        a.append(child1)
        a.append(child2)

        # === for p2 ===

    return a


def mutation(p):  # 突變
    for _ in range(NUM_MUTATION):
        row = np.random.randint(NUM_CROSSOVER_2)  # 任選一個染色體

        if np.random.randint(2) == 0:
            [j, k] = np.random.choice(NUM_BIT, 2, replace=False)  # 任選兩個基因

            p[row][j], p[row][k] = p[row][k], p[row][j]  # 此染色體的兩基因互換
        # else:
        #     j = np.random.randint(NUM_ELEVATOR)  # 任選1個基因
        #     p2[row][j] = np.random.randint(1, 11)


def sort_chrome(a, a_fit):  # a的根據a_fit由大排到小
    a_index = range(len(a))  # 產生 0, 1, 2, ..., |a|-1 的 list

    a_fit, a_index = zip(*sorted(zip(a_fit, a_index), reverse=True))  # a_index 根據 a_fit 的大小由大到小連動的排序

    return [a[i] for i in a_index], a_fit  # 根據 a_index 的次序來回傳 a，並把對應的 fit 回傳


def replace(p, p_fit, a, a_fit):  # 適者生存
    b = np.concatenate((p, a), axis=0)  # 把本代 p 和子代 a 合併成 b             # 把本代 p 和子代 a 合併成 b
    b_fit = p_fit + a_fit  # 把上述兩代的 fitness 合併成 b_fit

    b, b_fit = sort_chrome(b, b_fit)  # b 和 b_fit 連動的排序

    return b[:NUM_CHROME], list(b_fit[:NUM_CHROME])  # 回傳 NUM_CHROME 個為新的一個世代


print("CHROME")
pop = initPop()
pop1 = pop

pop = pop.tolist()
pop1 = pop1.tolist()

pop_fit, e1, e2 = evaluatePop(pop)
print(pop_fit)
pop_fit1, e1_, e2_ = evaluatePop(pop1)
print(pop_fit1)

best_outputs = []  # 用此變數來紀錄每一個迴圈的最佳解 (new)
best_outputs.append(np.max(pop_fit))  # 存下初始群體的最佳解

mean_outputs = []  # 用此變數來紀錄每一個迴圈的平均解 (new)
mean_outputs.append(np.average(pop_fit))  # 存下初始群體的最佳解

for i in range(NUM_ITERATION):
    parent = larger_tournament(pop, pop_fit)
    offspring = crossover_uniform(parent)  # 均勻交配
    mutation(offspring)
    # print("offspring")
    # print(np.array(offspring1).shape)
    # print(np.array(offspring2).shape)
    offspring_fit, e1, e2 = evaluatePop(offspring)  # 算子代的 fit
    pop, pop_fit = replace(pop, pop_fit, offspring, offspring_fit)  # 取代

    best_outputs.append(np.max(pop_fit))  # 存下這次的最佳解
    mean_outputs.append(np.average(pop_fit))  # 存下這次的平均解

    if i != NUM_ITERATION - 1:
        print('iteration %d: y = %d' % (i, -pop_fit[0]))  # fit 改負的
    else:
        print('iteration %d: x = %s, y = %d' % (i, pop[0], -pop_fit[0]))  # fit 改負的
pop_fit, e1, e2 = evaluatePop(pop)
ans = []
pre = -1
floor_i = []
ans_e1 = []
ans_e2 = []
for i in range(NUM_BIT):
    loc = int(pop[0][i])
    floor_i.append(location[loc])
    if (location[loc] == pre):
        continue
    else:
        ans.append(location[loc])
        pre = location[loc]

pre_e1 = -1
for i in range(len(e1[0])):
    a = []
    if (len(e1[0][i]) == 0):
        continue
    for j in range(len(e1[0][i])):
        loc = int(e1[0][i][j])
        if (location[loc] == pre_e1):
            continue
        else:
            a.append(location[loc] + 1)
            pre_e1 = location[loc]
    ans_e1.append(a)

pre_e2 = -1
for i in range(len(e2[0])):
    a = []
    if (len(e2[0][i]) == 0):
        continue
    for j in range(len(e2[0][i])):
        loc = int(e2[0][i][j])
        if (location[loc] == pre_e2):
            continue
        else:
            a.append(location[loc] + 1)
            pre_e2 = location[loc]
    ans_e2.append(a)

print(floor_i)
print(np.array(ans) + 1)
print(e1[0])
print(e2[0])
print(ans_e1)
print(ans_e2)

# 畫圖
# import matplotlib.pyplot
#
# matplotlib.pyplot.plot(best_outputs)
# matplotlib.pyplot.plot(mean_outputs)
# matplotlib.pyplot.xlabel("Iteration")
# matplotlib.pyplot.ylabel("Fitness")
# matplotlib.pyplot.show()
best_outputs = []
mean_outputs = []
best_outputs.append(np.max(pop_fit1))  # 存下初始群體的最佳解
mean_outputs.append(np.average(pop_fit1))  # 存下初始群體的最佳解
for i in range(NUM_ITERATION):
    parent = roulette_wheel(pop1, pop_fit1)
    offspring = crossover_uniform(parent)  # 均勻交配
    mutation(offspring)
    # print("offspring")
    # print(np.array(offspring1).shape)
    # print(np.array(offspring2).shape)
    offspring_fit, e1_, e2_ = evaluatePop(offspring)  # 算子代的 fit
    pop1, pop_fit1 = replace(pop1, pop_fit1, offspring, offspring_fit)  # 取代

    best_outputs.append(np.max(pop_fit1))  # 存下這次的最佳解
    mean_outputs.append(np.average(pop_fit1))  # 存下這次的平均解

    if i != NUM_ITERATION - 1:
        print('iteration %d: y = %d' % (i, -pop_fit1[0]))  # fit 改負的
    else:
        print('iteration %d: x = %s, y = %d' % (i, pop1[0], -pop_fit1[0]))  # fit 改負的
pop_fit1, e1_, e2_ = evaluatePop(pop1)
ans = []
pre = -1
floor_i = []
ans_e1 = []
ans_e2 = []
for i in range(NUM_BIT):
    loc = int(pop1[0][i])
    floor_i.append(location[loc])
    if (location[loc] == pre):
        continue
    else:
        ans.append(location[loc])
        pre = location[loc]

pre_e1 = -1
for i in range(len(e1_[0])):
    a = []
    if (len(e1_[0][i]) == 0):
        continue
    for j in range(len(e1_[0][i])):
        loc = int(e1_[0][i][j])
        if (location[loc] == pre_e1):
            continue
        else:
            a.append(location[loc] + 1)
            pre_e1 = location[loc]
    ans_e1.append(a)

pre_e2 = -1
for i in range(len(e2_[0])):
    a = []
    if (len(e2_[0][i]) == 0):
        continue
    for j in range(len(e2_[0][i])):
        loc = int(e2_[0][i][j])
        if (location[loc] == pre_e2):
            continue
        else:
            a.append(location[loc] + 1)
            pre_e2 = location[loc]
    ans_e2.append(a)

print(floor_i)
print(np.array(ans) + 1)
print(e1_[0])
print(e2_[0])
print(ans_e1)
print(ans_e2)

# 畫圖
# import matplotlib.pyplot
#
# matplotlib.pyplot.plot(best_outputs)
# matplotlib.pyplot.plot(mean_outputs)
# matplotlib.pyplot.xlabel("Iteration")
# matplotlib.pyplot.ylabel("Fitness")
# matplotlib.pyplot.show()