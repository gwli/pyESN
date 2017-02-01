class A(object):
    def __init__(self, a, **kwds):
        print "inter A init"
        self.a = a
        super(A, self).__init__(**kwds)
        print "leave a init"
         
class B(object):
    def __init__(self, b, c, **kwds): 
        print "inter B init"
        self.b = b
        self.c = c
        super(B, self).__init__(**kwds)
        print "leave B init"
     
class M(A, B): pass
class M1(A, B): pass
class N(B, A): pass 
class N1(B, A): pass 
 
M(a=1, b=2, c=3)
N(a=1, b=2, c=3)

#M1(1, 2, 3)
M1(1,b=2,c=3)
N1(1, 2, 3)
print "aaa"
