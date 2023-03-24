listOfElems = [11, 22, 33, 45, 66, 77, 88, 99, 101]
# Count odd numbers in the list
count = sum(map(lambda x : x%2 == 1, listOfElems))
print('Count of odd numbers in a list : ', count)