# helloWorld.py
# -------------
# Licensing Information: Please do not distribute or publish solutions to this
# project. You are free to use and extend these projects for educational
# purposes. The Pacman AI projects were developed at UC Berkeley, primarily by
# John DeNero (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# For more info, see http://inst.eecs.berkeley.edu/~cs188/sp09/pacman.html

# Comment: This is a fairly simple Python script

print('Hello, World!')


def fib(n):
    first = 0
    second = 1
    tmp = 0
    if n < 2:
        return n
    else:
        for i in range(2, n):
           tmp = first+second
           first = second
           second = tmp
        return first

if __name__ == '__main__':
    print(fib(3))


