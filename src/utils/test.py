class thing():

    def do(self, string):
        print(string)

    def new_do(self, string):
        print(string + "cfa")

obj = thing()

obj.do = obj.new_do

obj.do("ss")