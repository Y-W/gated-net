def proc(a):
    d = np.array(eval(a.replace('][','],[').replace(' ',',')))
    e = np.sum(d, axis=0)
    g = 1.0 - np.sum(d*d/e)
    print e[0]
    print g
