import matplotlib.pyplot as plt
import numpy as np
import tikzplotlib



def main():
    datafile = open('inference_results.txt', 'r')
    
    guided_best = []
    unguided_best = []

    state = 'out'
    guided_durations = []
    guided_values = []
    unguided_durations = []
    unguided_values = []


    plt.figure(1)
    cmap = plt.get_cmap('hsv')
    colors = cmap(np.linspace(0,1,4)) #get 10 colors along the full range of hsv colormap

    
    for line in datafile.readlines():
        if 'unguided' in line:
            # unguided:t: 0.43 :v: 1.7158132211000248
            if state != 'unguided':
                if len(unguided_durations) > 0:
                    plt.plot(unguided_durations, unguided_values, 'k')            
                    unguided_best.append(unguided_values[-1])
                state = 'unguided'
                unguided_durations = []
                unguided_values = []
            time, value = float(line.split(':')[2]), float(line.split(':')[4])
            unguided_durations.append(time)
            unguided_values.append(value)     
        elif 'guided' in line:
            # guided:t: 1.02 :v: 0.7017698238005445
            if state != 'guided':
                if len(guided_durations) > 0:
                    plt.plot(guided_durations, guided_values, 'g')
                    guided_best.append(guided_values[-1])
                state = 'guided'
                guided_durations = []
                guided_values = []
            time, value = float(line.split(':')[2]), float(line.split(':')[4])
            guided_durations.append(time)
            guided_values.append(value)
  
        
    datafile.close()
    print('average guided best', np.mean(guided_best), 'average unguided best', np.mean(unguided_best))
    
    if len(guided_durations) > 0:
        plt.plot(guided_durations, guided_values, 'g')    
    if len(unguided_durations) > 0:
        plt.plot(unguided_durations, unguided_values, 'k')       
    
    plt.ylabel("solution gap")
    plt.xlabel("solving time (s)")
    plt.draw()
    plt.waitforbuttonpress(10) 
    
    tikzplotlib.save("inference_seattle_len_34_.tex")
    


main()
