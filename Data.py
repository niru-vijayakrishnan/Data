class Data:

    def __init__(self, DataFile):
        self.DataFile = DataFile
        self.readTextFile()

    def readTextFile(self):
        #Separates Data into X and Y from .txt file
        self.X_DATA = []
        self.Y_DATA = []
        with open(self.DataFile) as data_file:
            for line in data_file:
                dataLine = line.split()
                self.Y_DATA.append(float(dataLine.pop()))
                self.X_DATA.append(dataLine)
        for i in range(len(self.X_DATA)):
            for j in range(len(self.X_DATA[i])):
                self.X_DATA[i][j] = float(self.X_DATA[i][j])