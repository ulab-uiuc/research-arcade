

arr = [None, True, False]

def true_false(x):
    if x: print("1")

    if not x: print("0")

def true_false_none(x):
    if x == None: print("-1")
    if x == True: print("1")
    if x == False: print("0")

for element in arr:
    true_false_none(element)