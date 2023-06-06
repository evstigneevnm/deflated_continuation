import numpy as np

class block_system_test(object):
    """docstring for block_system_test"""
    def __init__(self):
        super(block_system_test, self).__init__()
        self.A = None
        self.b = None
        self.c = None
        self.d = None
        self.alpha = 1.0e-9
        self.beta = 1.9
        self.folder = "../../../../dat_files/"
        self.load_data()
        self.cT = None
        self.d1 = None
        self.rhs_block = None
        self.A_block = None
        self.x_block = None
        self.rhs_reduced = None
        self.A_reduced = None
        self.x_ = None
        self.v_ = None
        self.x_reduced = None
        self.x_sm = None
        self.x_sm_sa = None
        self.x_reduced_v = None

    def load_data(self):
        self.A = np.loadtxt(self.folder + 'A.dat')
        self.b = np.loadtxt(self.folder + 'b.dat')
        self.c = np.loadtxt(self.folder + 'c.dat')
        self.d = np.loadtxt(self.folder + 'd.dat')

    def prepare_data(self):
        self.cT = np.reshape(self.c, (1,len(self.c)) )
        self.d1 = np.reshape(self.d, (len(self.d),1) )
        self.rhs_block = np.block([self.b,self.beta])
        self.A_block = np.block([[self.A,self.d1],[self.cT,self.alpha]])

        self.A_reduced = self.A - np.dot(self.d1,self.cT)/self.alpha
        self.rhs_reduced = self.b-self.d*self.beta/self.alpha


    def solve_block(self):
        self.x_block = np.linalg.solve(self.A_block, self.rhs_block)

    def solve_reduced(self):
        self.x_ = np.linalg.solve(self.A_reduced, self.rhs_reduced)
        self.v_ = (self.beta - np.dot(self.c, self.x_) )/self.alpha
        self.x_reduced = np.block([self.x_, self.v_])

    def solve_reduced_v(self):
        p1 = np.linalg.solve(self.A, self.d)
        s1 = np.linalg.solve(self.A, self.b)

        p2 = (-np.dot(self.c, p1) + self.alpha)
        rhs = self.beta - np.dot(self.c, s1)
        v_ = rhs/p2
        x_ = s1 - p1*v_
        self.x_reduced_v = np.block([x_, v_])



    def solve_sm(self):
        p1 = np.linalg.solve(self.A, self.rhs_reduced)
        p2 = -np.linalg.solve(self.A, self.d)/self.alpha
        num = p2*np.dot(self.c, p1)
        din = 1+np.dot(self.c, p2)
        x_ = p1-num/din
        v_ = (self.beta - np.dot(self.c, x_) )/self.alpha
        self.x_sm = np.block([x_, v_])
        
    def solve_sm_small_alpha(self):
        p1 = np.linalg.solve(self.A, self.rhs_reduced)
        p2 = -np.linalg.solve(self.A, self.d)
        num = p2*np.dot(self.c, p1)
        din = self.alpha+np.dot(self.c, p2)
        x_ = p1-num/din
        v_ = (self.beta - np.dot(self.c, x_) )/self.alpha
        self.x_sm_sa = np.block([x_, v_])


    def compare(self):
        print("block residual: ", np.linalg.norm(np.dot(self.A_block, self.x_block)-self.rhs_block) )
        print("x_reduced->block residual: ", np.linalg.norm(np.dot(self.A_block, self.x_reduced)-self.rhs_block) )
        print("x_reduced_v->block residual: ", np.linalg.norm(np.dot(self.A_block, self.x_reduced_v)-self.rhs_block) )
        print("x_sm->block residual: ", np.linalg.norm(np.dot(self.A_block, self.x_sm)-self.rhs_block) )
        print("x_sm_sa->block residual: ", np.linalg.norm(np.dot(self.A_block, self.x_sm_sa)-self.rhs_block) )
    
    def compare_v_value(self):
        print("block: ", self.x_block[-1:] )
        print("x_reduced: ", self.x_reduced[-1:] )
        print("x_reduced_v: ", self.x_reduced_v[-1:] )
        print("x_sm: ", self.x_sm[-1:])
        print("x_sm_sa: ", self.x_sm_sa[-1:] )



def main():
    test = block_system_test()
    test.prepare_data()
    test.solve_block()
    test.solve_reduced()
    test.solve_sm()
    test.solve_sm_small_alpha()
    test.solve_reduced_v()
    test.compare_v_value()
    test.compare()

if __name__ == '__main__':
    main()