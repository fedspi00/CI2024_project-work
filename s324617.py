import numpy as np

def f0(x: np.ndarray) -> np.ndarray: 
    return np.subtract(np.reciprocal(np.reciprocal(x[0])), np.tanh(np.tanh(np.tanh(np.multiply(x[1], -0.1817006964170753)))))

def f1(x: np.ndarray) -> np.ndarray: 
    return np.sin(x[0])
  
def f2(x: np.ndarray) -> np.ndarray: 
    return np.multiply(np.cosh(np.negative(np.square(-2.891705823115304))), np.multiply(np.arctan(np.add(np.add(x[1], x[2]), np.add(x[0], x[0]))), np.cosh(np.square(np.cosh(-1.70765096352428)))))

def f3(x: np.ndarray) -> np.ndarray: 
    return np.add(np.multiply(np.absolute(np.maximum(x[1], -4.973875097263662)), np.multiply(np.minimum(np.negative(x[1]), x[1]), np.maximum(x[1], -4.973875097263662))), np.subtract(np.subtract(np.cosh(np.minimum(4.365965511080824, x[0])), np.add(np.sinh(-2.0174843148004316), x[2])), np.add(np.add(x[2], -1.9699415189846148), np.add(np.remainder(x[0], -2.07102849364453), np.divide(x[2], 0.6608317890232218)))))
  
def f4(x: np.ndarray) -> np.ndarray: 
    return np.subtract(np.maximum(np.arccos(np.tanh(np.multiply(5.5921039198210485, x[0]))), np.arccos(np.tanh(np.divide(-4.350344424685952, x[0])))), np.minimum(np.multiply(np.add(-2.5649473664638034, np.minimum(-4.786393096058477, x[0])), np.cos(x[1])), np.multiply(np.cos(x[1]), np.add(np.cos(x[1]), np.subtract(-5.648104136297606, 0.041896408583709466)))))
  
def f5(x: np.ndarray) -> np.ndarray: 
    return np.multiply(np.minimum(np.multiply(np.add(np.subtract(x[0], x[1]), np.maximum(2.2612182619474845, x[0])), np.divide(np.subtract(3.495395070093316, x[1]), np.reciprocal(x[0]))), np.arctan(np.divide(np.reciprocal(3.9475633927744758), np.power(x[1], x[1])))), np.square(-2.180789186394816e-05))

def f6(x: np.ndarray) -> np.ndarray: 
    return np.add(np.add(np.multiply(np.add(x[1], np.log(1.1135794785464643)), np.maximum(-0.0974477804335967, -4.8350566121696135)), np.reciprocal(np.divide(np.exp(0.09836178124853223), np.minimum(x[1], 3.1764013732538716)))), np.add(np.multiply(x[0], np.add(np.maximum(-0.7971972842251871, -4.79648789365918), np.sin(3.0327581081551704))), np.add(np.multiply(np.multiply(1.1761942159177892, x[1]), -0.0974477804335967), x[1])))

def f7(x: np.ndarray) -> np.ndarray: 
    return np.cosh(np.subtract(-2.280284342585475, np.multiply(np.maximum(np.add(-0.0416842700855411, x[0]), np.minimum(-1.2412509359118018, x[1])), np.maximum(x[1], np.minimum(x[0], -0.716333093981695)))))
  
def f8(x: np.ndarray) -> np.ndarray:
    return np.multiply(np.subtract(np.add(114.57246359623406, np.exp(x[5])), np.subtract(np.sinh(x[5]), np.multiply(-4.708598611033789, -5.42828459880723))), np.subtract(np.subtract(np.minimum(np.subtract(x[4], -3.4939274899130344), np.tanh(x[3])), np.subtract(np.cbrt(x[5]), np.sinh(x[5]))), np.subtract(np.divide(np.square(x[4]), 2.579579025259571), np.multiply(x[5], 0.9293568845654132))))
