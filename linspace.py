def linspace(a,b,c):
    d=[]
    k=(b-a)/(c-1)
    for i in range(c):
        d.append(a+k*i)

    return d

