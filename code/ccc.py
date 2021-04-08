import numpy as np
import matplotlib.pyplot as plt
from sklearn.gaussian_process.kernels import RBF, ConstantKernel as C
from sklearn.gaussian_process import GaussianProcessRegressor
from mpl_toolkits.mplot3d import Axes3D

i = [ 2, 2, 2, 2, 4, 4, 7, 7, 11, 15, 20, 20, 20, 22, 22, 22, 25, 25, 26, 26, 26
    , 28, 28, 29, 43, 59, 66, 74, 84, 94, 105, 122, 147, 159, 170, 189, 214, 228, 241
    , 256, 274, 293, 331, 360, 420, 461, 502, 511, 581, 639, 639, 701, 773, 839, 839
    , 878, 889, 924, 963, 1007, 1101, 1128, 1193, 1307, 1387, 1468, 1693, 1866, 1866, 1953
    , 2178, 2495, 2617, 3139, 3139, 3654, 3906, 4257, 4667, 5530, 6005, 6748, 7370, 7645
    , 8100, 8626, 9787, 10296, 10797, 10797, 11135, 11512, 12368, 12829, 13231, 13441, 14153
    , 13736, 13895, 14088, 14305, 14571, 14877, 15078, 15253, 15253, 15477, 15575, 15663, 15777
    , 15847, 15968, 16049, 16120, 16203, 16237, 16285, 16305, 16367, 16367, 16424, 16513, 16536
    , 16550, 16581, 16623, 16651, 16598, 16673, 16716, 16751, 16787, 16837, 16867, 16911
    , 16958 ]
d = [ 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1
    , 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2, 4, 4, 5, 6, 6, 6, 6, 6, 6, 6
    , 6, 10, 10, 15, 16, 19, 22, 22, 27, 29, 29, 29, 33, 35, 41, 42, 43, 45, 47
    , 49, 52, 54, 54, 56, 57, 62, 63, 77, 77, 85, 92, 93, 94, 99, 99, 108, 123, 143
    , 146, 178, 190, 222, 236, 236, 263, 281, 328, 345, 360, 372, 385, 394, 413, 430
    , 455, 474, 487, 536, 556, 556, 577, 590, 607, 624, 633, 657, 678, 697, 713, 725
    , 744, 749, 768, 768, 777, 796, 808, 820, 830, 846, 858, 881, 887, 894, 898, 899
    , 902, 905, 911, 916]
r = [ 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 4, 9, 9, 9
    , 9, 12, 12, 12, 13, 18, 18, 22, 22, 22, 22, 22, 22, 22, 22, 32, 32, 32, 43
    , 43, 43, 46, 76, 76, 76, 101, 118, 118, 118, 118, 118, 144, 144, 144, 150, 191
    , 232, 235, 235, 285, 310, 359, 372, 404, 424, 424, 424, 472, 472, 514, 514, 514
    , 575, 592, 622, 632, 685, 762, 762, 784, 799, 853, 901, 935, 1069, 1159, 1159, 1239
    , 1356, 1494, 1530, 1656, 1809, 1899, 1899, 2368, 2460, 2975, 3205, 3981, 4156, 4496
    , 4496, 4918, 5146, 5906, 8127, 8293, 8531, 8920, 9868, 10338, 10338, 11153, 11564, 11564
    , 11564, 12672, 13005, 13244, 13413, 13612, 13810, 13973, 14096, 14213, 14267, 14342, 14463
    , 14585, 14702, 14785, 14925 ]
for a in range(0,136):
    r[a]=d[a]+r[a]
for a in range(0,136):
    i[a]=i[a]-r[a]
date=xobs = np.array([[0],[1],[2],[3],[4],[5],[6],[7],[8],[9],[10], [11], [12],[13],[14],[15],[16],[17],[18],[19],[20], [21], [22],[23],[24],[25],[26],[27],[28],[29],[30],[31],[32],[33],[34],[35],[36],[37],[38],[39],[40],[41],[42],[43],[44],[45],])
#数据用japan 1.22-6.5


gama=[]

for a in range(0,135):
    gama.append((r[a+1]-r[a])/i[a])
gama_true = sum(gama)/136
print(gama_true)



#接下来将S0和beta视作二元自变量，因变量为I[t],t=1-136，与真值的均方误差
#这样就可以用高斯过程回归，模拟出s0与beta的分布，估计s0与beta的真值
#第一次0.00000-0.00001,000000-100000
#第二次0.000008+-0.000005,20000+-5000
#第三次0.0000079+-0.0000005，19000+-500
beta_min=0.0000079-0.0000005
beta_max=0.0000079+0.0000005
s0_min=19000-500
s0_max=19000+500


def y(beta0,s0):
    s1=[]
    i1=[]
    s1.append(s0)
    i1.append(i[0])
    res = 0
    for a in range(0,135):
        s1.append(s1[a]-beta0*i1[a]*s1[a])
        i1.append(i1[a]*(1-gama_true)+beta0*i1[a]*s1[a])
    for a in range(1,136):
        res+=((i[a]-i1[a])**2)
    return(res**0.5/136)

beta=[]
s=[]

for a in range(0,10):
    beta.append(beta_min+a*(beta_max-beta_min)/10)
    s.append(s0_min+a*(s0_max-s0_min)/10)
beta_s=np.zeros((100,2))

for a in range(0,10):
    for b in range(0,10):
        beta_s[10*a+b]=[beta[a],s[b]]


#我不知道为什么，直接用上面的beta_s做高斯回归，模型是分块片状的，下面将beta和s变换成相同尺度的坐标
beta_s_for_train = np.zeros((100,2))
for a in range(0,100):
    beta_s_for_train[a,0]=beta_s[a,0]*100000
    beta_s_for_train[a,1]=beta_s[a,1]/10000
    #全部变化到1位数


I=[]

for a in range(0,100):
    I.append(y(beta_s[a,0],beta_s[a,1]))


kernel = C(0.01, (0.0001, 0.01)) * RBF(0.05, (1e-7, 10))
gp2 = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=50, alpha=0.001)
gp2.fit(beta_s_for_train,I)


x_min, x_max = beta_s_for_train[:,0].min(), beta_s_for_train[:,0].max()
y_min, y_max = beta_s_for_train[:,1].min(), beta_s_for_train[:,1].max()
xset, yset = np.meshgrid(np.arange(x_min, x_max, (x_max-x_min)/30), np.arange(y_min, y_max,(y_max-y_min)/30))
output, err = gp2.predict(np.c_[xset.ravel(), yset.ravel()], return_std=True)
output, err = output.reshape(xset.shape), err.reshape(xset.shape)
sigma = np.sum(gp2.predict(beta_s, return_std=True)[1])
up, down = output * (1 + 1.96 * err), output * (1 - 1.96 * err)


fig = plt.figure(figsize=(10.5, 5))
ax1 = fig.add_subplot(121, projection='3d')
surf = ax1.plot_wireframe(xset, yset, output, rstride=1, cstride=1, antialiased=True)

ax1.scatter(beta_s_for_train[:,0], beta_s_for_train[:,1], I, c='red',marker='.')
#ax1.scatter(beta_s[:,0], beta_s[:,1], I[:], c='red')

ax1.set_xlabel('beta')
ax1.set_ylabel('s0')

plt.show()



beta_true=7.90e-06
s0_true=1.94e+04


s_pred=[]
i_pred=[]
s_pred.append(s0_true)
i_pred.append(i[0])
for a in range(0,136):
    s_pred.append(s_pred[a]-beta_true*i_pred[a]*s_pred[a])
    i_pred.append(i_pred[a]*(1-gama_true)+beta_true*i_pred[a]*s_pred[a])


plt.plot(i_pred)
plt.plot(i)


#完整的预测
s_pred=[]
i_pred=[]
r_pred=[]
s_pred.append(s0_true)
i_pred.append(i[0])
r_pred.append(0)
a=0
while i_pred[a]>=i_pred[0]*0.1:
    if s_pred[a]-beta_true*i_pred[a]*s_pred[a]>0:
        s_pred.append(s_pred[a]-beta_true*i_pred[a]*s_pred[a])
    else:
        s_pred.append(0)
    i_pred.append(i_pred[a]*(1-gama_true)+beta_true*i_pred[a]*s_pred[a])
    r_pred.append(r_pred[a]+gama_true*i_pred[a])
    a+=1


plt.plot(i_pred)
plt.plot(r_pred)
plt.plot(s_pred)
plt.plot(i)