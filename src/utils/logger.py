'''
Logger Utility Function
Author: Yuya Jeremy Ong (yuyajeremyong@gmail.com)
'''
from __future__ import print_function

'''
TODO:
- Export logger function to better infrastructure support with monolog.
- Use an external logger service similar to weights and biases.
'''

class Logger:
    def __init__(self, root_dir, header=None):
        self.root_dir = root_dir
        output = open(self.root_dir, 'w')
        if header is not None: output.write(header + '\n')
        output.close()

    def log(self, data):
        output = open(self.root_dir, 'a')
        output.write(data + '\n')
        output.close()

# Unit Testing
if __name__ == '__main__':
    logger = Logger('./test.csv', 'id,name,age')
    logger.log('1,yuya,21')
    logger.log('2,bobby,20')
    
