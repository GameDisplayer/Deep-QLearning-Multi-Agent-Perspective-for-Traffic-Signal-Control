import matplotlib.pyplot as plt
import os

class Visualization:
    def __init__(self, path, dpi):
            self._path = path
            self._dpi = dpi


    def save_data_and_plot(self, data, filename, xlabel, ylabel):
        """
        Produce a plot of performance of the agent over the session and save the relative data to txt
        """
        min_val = min(data)
        max_val = max(data)

        plt.rcParams.update({'font.size': 24})  # set bigger font size

        plt.plot(data)
        plt.ylabel(ylabel)
        plt.xlabel(xlabel)
        plt.margins(0)
        plt.ylim(min_val - 0.05 * abs(min_val), max_val + 0.05 * abs(max_val))
        fig = plt.gcf()
        fig.set_size_inches(20, 11.25)
        fig.savefig(os.path.join(self._path, 'plot_'+filename+'.png'), dpi=self._dpi)
        plt.close("all")

        with open(os.path.join(self._path, 'plot_'+filename + '_data.txt'), "w") as file:
            for value in data:
                    file.write("%s\n" % value)
                    
                    
    def save_data_and_plot_multiple_curves(self, list_of_data, filename, xlabel, ylabel, scenarios):
        """
        Produce a plot of performance of the agent over the session and save the relative data to txt
        """
        
        data = list_of_data[0]
        data1 = list_of_data[1]
        data2 = list_of_data[2]
        data3 = list_of_data[3]
        
        min_val = min(data + data1 + data2 + data3)
        max_val = max(data + data1 + data2 + data3)

        plt.rcParams.update({'font.size': 24})  # set bigger font size

        plt.plot(data, color="r", label=scenarios[0])
        plt.plot(data1, color="g", label=scenarios[1])
        plt.plot(data2, color="b", label=scenarios[2])
        plt.plot(data3, color="k", label=scenarios[3])
        plt.legend(framealpha=1, frameon=True)
        plt.ylabel(ylabel)
        plt.xlabel(xlabel)
        plt.margins(0)
        plt.ylim(min_val - 0.05 * abs(min_val), max_val + 0.05 * abs(max_val))
        fig = plt.gcf()
        fig.set_size_inches(20, 11.25)
        fig.savefig(os.path.join(self._path, 'plot_'+filename+'.png'), dpi=self._dpi)
        plt.close("all")

        for i in range(len(list_of_data)):
            with open(os.path.join(self._path, 'plot_'+filename + '_data' + scenarios[i] + '.txt'), "w") as file:
                for value in list_of_data[i]:
                        file.write("%s\n" % value)
    