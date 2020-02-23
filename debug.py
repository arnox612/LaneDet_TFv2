def con(x=1):
    y = 2 * x
    return y


def test_bottle(x=3):
    y1 = con()
    y2 = con(x=x)

    print('y1=%d,y2=%d' % (y1, y2))
    print('y1={},y2={}'.format(y1, y2))
    return y1, y2


def test_encoder():
    y = test_bottle(x=2)
    print(y)
    return y



test_encoder()

from keras.layers import Conv2D

def Conv2d(input, filter_size, str, padding, use_bias ):
    x = Conv2D(fillters, kernel_size= , strides=, padding= , use_bias= , **kwargs)

from tensorflow import add

a = [1,2,3]
b = [1,1,1]
c =