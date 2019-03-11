import numpy as np

np.random.seed(0)

x = np.array(((0,0), (0,1), (1,0), (1,1)), dtype=np.float32)
t = np.array(((-1), (-1), (-1), (1)), dtype=np.float32)

lrs = [0.1, 0.01]
linestyles = ['solid', 'dashed']
plts = []

for _i in range(len(lrs)):
    lr = lrs[_i]
    
    # perceptron
    np.random.seed(0)
    w = np.random.normal(0., 1, (3))
    print("weight >>", w)

    # add bias
    _x = np.hstack([x, [[1] for _ in range(4)]])

    # train
    ite = 0
    w1 = [w[0]]
    w2 = [w[1]]
    w3 = [w[2]]

    while True:
        ite += 1
        y = np.array(list(map(lambda x: np.dot(w, x), _x)))

        En = np.array([0 for _ in range(3)], dtype=np.float32)

        for i in range(4):
            if y[i] * t[i] < 0:
                En += t[i] * _x[i]

        print("iteration:", ite, "y >>", y)
        if np.any(En != 0):
            w += lr * En
            w1.append(w[0])
            w2.append(w[1])
            w3.append(w[2])
        else:
            break
    print("training finished!")
    print("weight >>", w)

    inds = list(range(ite))
    import matplotlib.pyplot as plt
    linestyle = linestyles[_i]
    plts.append(plt.plot(inds, w1, markeredgewidth=0, linestyle=linestyle)[0])
    plts.append(plt.plot(inds, w2, markeredgewidth=0, linestyle=linestyle)[0])
    plts.append(plt.plot(inds, w3, markeredgewidth=0, linestyle=linestyle)[0])

plt.legend(plts, ["w1:lr=0.1","w2:lr=0.1","w3:lr=0.1","w1:lr=0.01","w2:lr=0.01","w3:lr=0.01"], loc=1)
plt.savefig("answer_perceptron3.png")
plt.show()

# test
y = np.array(list(map(lambda x: np.dot(w, x), _x)))

for i in range(4):
    y = np.dot(w, _x[i])
    print("in >>", _x[i], ", out >>", y) 
    
