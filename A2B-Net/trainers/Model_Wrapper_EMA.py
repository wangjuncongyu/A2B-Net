

class Model_Wrapper:
    def __init__(self, model, decay):
        self.model = model
        self.decay = decay
        self.teacher = {}
        self.backup = {}
        self.register()

    # 注册所有需要跟踪的变量
    def register(self):
        for param in self.model.variables:
            if param.trainable:
                self.teacher[param.name] = param.value()               

    # 每次变量的值改变后更新Teacher变量的值
    def update_teacher(self):
        for param in self.model.variables:
            if param.trainable:
                assert param.name in self.teacher
                new_average = (1.0 - self.decay) * param.value() + self.decay * self.teacher[param.name]
                self.teacher[param.name] = new_average
                
   
    # 将模型参数变成Teacher值，backup是真实值的备份
    def apply_teacher(self):
        for param in self.model.variables:
            if param.trainable:
                assert param.name in self.teacher
                self.backup[param.name] = param.value()
                param.assign(self.teacher[param.name])
                
   
    # 将模型的参数变回真实值
    def restore(self):
        if len(self.backup) == 0:
            return 
        for param in self.model.variables:
            if param.trainable:
                assert param.name in self.backup
                param.assign(self.backup[param.name])
        self.backup = {}