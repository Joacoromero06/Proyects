a = 0
def fn():
    global a
    print(f'the value of global a is: {a}')
    a = 5
fn()
print(f'the value of global a was changed: {a}')

def sub1():
    a = 2
    b = 3
    print(f'the value of local a: {a} and local b: {b}')
def sub2():
    global a
    print(f'the value of global a: {a}')
    c = 0
    def sub3():
        nonlocal c
        c = 10
        print('the value of c was changed')
    sub3()
    print(f'the value of c: {c}')
sub2()

def recursiveprint_by_slice(l: list):
    if l:
        recursiveprint_by_slice(l[:len(l)-1]) # print of the first elems
        print(l[len(l)-1]) # print last element
        l[0] = 1000 # change l, not shared resource
recursiveprint_by_slice([ i for i in range (10) ])


def recursiveprint_by_index(l: list, i: int):
    if i >= 0:
        recursiveprint_by_index(l, i-1)
        print(l[i])
        l[0] = i + 10000
        print(f'i changed the original l: {l[0]}')
l = [ i for i in range(10) ]
recursiveprint_by_index( l, len(l)-1 )
print(l)
