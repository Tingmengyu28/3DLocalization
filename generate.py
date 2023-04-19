with open('./Data/id_test.txt', 'w') as f:
    for i in range(1, 10001):
        f.write(str(i) + '\n')