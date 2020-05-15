# PTA-for-digit-classification-MNIST-
 4 files listed in the beginning of the page as training set images, training set labels, test set images, and test set labels, download them. 
Each image is 28 by 28
One-hot coding used for output; ex:y = [1 0 0 0 0 0 0 0 0 0 0] means the input is 0.
This is one layer Neural network. 10 neurons assumed. The input is 28*28 = 784. 
The following is the steps of algorithm:
''
epoch = 0
for i = 1 to number of input: v = Wxi ==> Then the v = [v0 v1 ··· v9], So max{vi} become the calculated output. for example if v1 is the larger
Then v interpreted as 1, etc.
if the output is not as the same as desired output, error = error + 1
epoch ← epoch + 1
The updating rule for weight is "W ← W + η(d(x) − u(Wx))xT" ; u is the step activation function.
...
