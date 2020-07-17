class Base:
    def func0(self):
        obj = self.A()
        obj.func0("Base.func0")

    def func1(self):
        obj = self.A()
        obj.func1()

    def func2(self):
        obj = self.B()
        obj.func2()

    class NestedBase:
        def func0(self, f):
            print(f"{f}->Base.NestedBase.func0")
            print()

    class A(NestedBase):
        def func1(self):
            self.func0("Base.A")
            print("Base.BaseA.func1")
            print()

    class B(NestedBase):
        def func2(self):
            self.func0("Base.B")
            print("Base.BaseB.func2")
            print()


class Master(Base):
    def func0(self):
        obj = self.A()
        obj.func0("Master.func0")

    def func1(self):
        obj = self.A()
        obj.func1()

    def func2(self):
        obj = self.B()
        obj.func2()

    class NestedMaster(Base.NestedBase):
        def func0(self, f):
            print(f"{f}->Master.NestedMaster.func0")
            print()

    class A(Base.A):
        def func1(self):
            self.func0("Master.A")
            print("Master.MasterA.func1")
            print()

    class B(Base.B):
        def func2(self):
            self.func0("Master.B")
            print("Master.MasterB.func2")
            print()


m = Master()
m.func0()
m.func1()
m.func2()
