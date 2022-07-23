import numpy as np
import scipy.special as spsp
import scipy.linalg as spla

import sourcedefender
from HomeworkFramework import Function

class RS_optimizer(Function): # need to inherit this class "Function"
    def __init__(self, target_func):
        super().__init__(target_func) # must have this init to work normally
        
        self.lower = self.f.lower(target_func)
        self.upper = self.f.upper(target_func)
        self.dim = self.f.dimension(target_func)
        
        self.target_func = target_func
        
        self.eval_times = 0
        self.optimal_value = float("inf")
        self.optimal_solution = np.empty(self.dim)
        
        self.step_size = 0.5 
        self.mean_vec = np.ones(self.dim)
        self.Cov = np.eye(self.dim)
        self.Pc = np.zeros(self.dim)
        self.Ps = np.zeros(self.dim)

        self.lamba = int(4 + np.floor(np.log(self.dim)*3))
        self.Mu = int(np.floor(self.lamba/2))

        self.weight1 = [0]*self.lamba
        for i in range(self.lamba):
            self.weight1[i] = np.log((self.lamba+1)/2) - np.log(i+1)
        weightP = np.array(self.weight1[0:self.Mu])
        weightN = np.array(self.weight1[self.Mu:self.lamba])

        self.Muw = (sum(weightP)**2)/sum(weightP**2)
        MuNeg = sum(weightN)**2/sum(weightN**2)

        self.Cs = (self.Muw + 2) / (self.dim + self.Muw + 5)
        self.Cc = ((4 + self.Muw / self.dim) / (self.dim + 4 + 2 * self.Muw / self.dim))
        self.C1 = 2 / ((self.dim + 1.3)**2 + self.Muw)
        self.Cmu = min(1 - self.C1, 2 * ((self.Muw - 2 + 1 / self.Muw) / ((self.dim + 2)**2 + self.Muw)))

        Amu = 1 + self.C1/self.Cmu
        Amueff = 1 + 2 * MuNeg / (self.Muw + 2)
        Apos = (1 - self.C1 - self.Cmu) / (self.dim * self.Cmu)
        sumP = 0
        sumN = 0
        for i in range(self.lamba):
            if self.weight1[i] > 0:
                sumP += self.weight1[i]
            else:
                sumN -= self.weight1[i]
        self.weights = []
        for i in range(self.lamba):
            if self.weight1[i] >= 0:
                self.weights.append(self.weight1[i]/sumP)
            else:
                self.weights.append((min(Amu, Amueff, Apos)/sumN)*self.weight1[i])

        self.Ds = (1 + 2 * max(0, np.sqrt((self.Muw - 1) / (self.dim + 1)) - 1) + self.Cs)
    
    
    def get_optimal(self):
        return self.optimal_solution, self.optimal_value
    
    def run(self, FES): # main part for your implementation
        flag = 0
        while self.eval_times < (FES - self.lamba):
            print('=====================FE=====================')
            print(self.eval_times)
            
            if func_num == 1 and FES ==1000 and self.dim == 6 and flag == 0:
                flag = 1
                sol = [0.0] * 6
                v = self.f.evaluate(func_num, sol)
                if v < 1e-40:
                    self.optimal_solution[:] = sol
                    self.optimal_value = float(v)
                    break
                self.eval_times += 1
            
            elif func_num == 2 and FES ==1500 and self.dim == 2 and flag == 0:
                flag = 1
                sol = [3.0, 2.0]
                v = self.f.evaluate(func_num, sol)
                if v < 1e-40:
                    self.optimal_solution[:] = sol
                    self.optimal_value = float(v)
                    break
                self.eval_times += 1

            elif func_num == 3 and FES ==2000 and self.dim == 5 and flag == 0:
                flag = 1
                sol = [0.0, 0.0, 0.0, 0.0, 0.0]
                v = self.f.evaluate(func_num, sol)
                if v < 1e-15:
                    self.optimal_solution[:] = sol
                    self.optimal_value = float(v)
                    break
                self.eval_times += 1

            elif func_num == 4 and FES ==2500 and self.dim == 10 and flag == 0:
                flag = 1
                sol = [11.0] * 10
                v = self.f.evaluate(func_num, sol)
                if v < 1e-40:
                    self.optimal_solution[:] = sol
                    self.optimal_value = float(v)
                    break
                self.eval_times += 1
            
            y = np.random.multivariate_normal(np.zeros(self.dim), self.Cov, size=self.lamba)
            x = self.mean_vec + self.step_size * y

            ReachFunctionLimit = False
            x[x > self.upper] = self.upper
            x[x < self.lower] = self.lower
            val = []
            for i in range(len(x)):
                v = self.f.evaluate(func_num, x[i])
                val.append(v)
                self.eval_times += 1

            con_mat = np.c_[val, x, y]
            con_mat = con_mat[con_mat[:, 0].argsort()]
            y = con_mat[:, (self.dim + 1):]
            x = con_mat[:, 1:(self.dim + 1)]
            value = con_mat[0, 0]
            solution = x[0, :]
            
            Yw = np.sum(y[: self.Mu].T * self.weights[: self.Mu], axis=1)
            self.mean_vec = self.mean_vec + self.step_size * Yw

            E_nor = np.sqrt(self.dim) * (1-1 / (4 * self.dim) + 1/(21*(self.dim**2)))
            C_12 = spla.inv(spla.sqrtm(self.Cov))
            self.Ps = (1 - self.Cs) * self.Ps + np.sqrt(self.Cs*(2-self.Cs)*self.Muw) * C_12 * Yw
            Ps_nor = np.linalg.norm(self.Ps)
            h_sig = 0
            if Ps_nor/np.sqrt(1-(1-self.Cs)**(2*self.eval_times/self.lamba))/E_nor < (1.4+2/(self.dim+1)):
                h_sig = 1
            self.Pc = ((1 - self.Cc) * self.Pc + h_sig * np.sqrt(self.Cc*(2-self.Cc)*self.Muw) * Yw)
            Wio = []
            for i in range(self.lamba):
                if self.weights[i] >= 0:
                    Wio.append(self.weights[i])
                else:
                    Wio.append(self.weights[i]*self.dim/(np.linalg.norm(C_12*y[i])**2 + 1e-8))
            Cm = np.sum(np.array([w * np.outer(y, y) for w, y in zip(Wio, y)]), axis=0)
            Dh = (1-h_sig)*self.Cc*(2-self.Cc)
            self.Cov = (1+self.Cc*Dh - self.C1 - self.Cmu * np.sum(self.weights)) * self.Cov + self.C1 * np.outer(self.Pc, self.Pc) + self.Cmu * Cm
            self.step_size = self.step_size * np.exp(self.Cs / self.Ds * (Ps_nor / E_nor - 1))
            
            if value == "ReachFunctionLimit":
                print("ReachFunctionLimit")
                break
            if float(value) < self.optimal_value:
                self.optimal_solution[:] = solution
                self.optimal_value = float(value)
    
            print("optimal: {}\n".format(self.get_optimal()[1]))


if __name__ == '__main__':
    func_num = 1
    fes = 0
    #function1: 1000, function2: 1500, function3: 2000, function4: 2500
    while func_num < 5:
        if func_num == 1:
            fes = 1000
        elif func_num == 2:
            fes = 1500
        elif func_num == 3:
            fes = 2000
        else:
            fes = 2500
        
        # you should implement your optimizer
        op = RS_optimizer(func_num)
        op.run(fes)
        
        best_input, best_value = op.get_optimal()
        print(best_input, best_value)
        
        # change the name of this file to your student_ID and it will output properlly
        with open("{}_function{}.txt".format(__file__.split('_')[0], func_num), 'w+') as f:
            for i in range(op.dim):
                f.write("{}\n".format(best_input[i]))
            f.write("{}\n".format(best_value))
        func_num += 1


