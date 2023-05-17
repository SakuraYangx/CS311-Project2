
import matplotlib.pyplot as plt

crossover_Pro = [3968,5912,316, 275, 177, 426, 297]
flip_Pro = [3515,5018,  316,    275,173,400,279]
b = []
for i in range(len(crossover_Pro)):
    b.append((crossover_Pro[i]-flip_Pro[i])/flip_Pro[i]*100)



x = ['egl-e1-A','egl-s1-A','gdb1','gdb10','val1A','val4A','val7A']

plt.plot(x, b, 'b*-', alpha=0.7, linewidth=2, label='Accuracy')#'


for a, b in zip(x, b):
    plt.text(a, b, str(b), ha='center', va='bottom', fontsize=10)  #  ha='center', va='top'
plt.legend()

plt.show()