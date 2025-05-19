def gen(n):
    if n <= 2:
        for i in range(n):
            yield i
        return
    if (n % 2 == 1):
      yield n // 2
    for i in gen(n // 2):
        yield i
        yield i + ((n + 1) // 2)
def gen_lim(max_iter, n):
    cnt = 0
    while cnt < max_iter:
        for i in gen(n):
            if cnt == max_iter:
                return
            cnt += 1
            yield i

for j in range(10):
    print('\n', j, ':')
    for i in gen(j):
        print(i)

n = 1000

st = set(gen(n))
ls = list(gen(n))
print(len(ls), len(st))
for i in gen(n):
    if i not in st:
        print(i)