import numpy as np

np.random.seed(0)

xs = np.array([[0,0], [0,1], [1,0], [1,1]], dtype=np.float32)
ts = np.array([[1], [-1], [-1], [1]], dtype=np.float32)

lr = 0.1

# perceptron
w1 = np.random.normal(0, 1, [2, 2])
w2 = np.random.normal(0, 1, [2, 1])
print("weight1 >>\n", w1)
print("weight2 >>\n", w2)

# add bias
#_x = np.hstack([x, [[1] for _ in range(4)]])

# train
ite = 0
while True:
    ite += 1

    # feed forward
    z1 = xs
    z2 = np.dot(z1, w1)
    ys = np.dot(z2, w2)

    print((ys * ts).shape)

    print("iteration:", ite, "y >>\n", ys)
    
    if len(np.where((ys * ts) < 0)[0]) < 1:
        break

    # back propagation
    En = ts * ys
    grad_w2 = np.dot(z2.T, En)
    w2 += lr * grad_w2

    grad_w1 = np.dot(z1.T, np.dot(En, w2.T))
    w1 += lr * grad_w1
    
print("training finished!")
print("weight1 >>\n", w1)
print("weight2 >>\n", w2)

# test
for i in range(4):
    y = np.dot(np.dot(xs[i], w1), w2)
    print("in >>", xs[i], ", out >>", y) 
    
