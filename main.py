def conv(s, x):
    ans = 0
    for ch in s:
        ich = x if ch == 'x' else int(ch)
        ans = ans * 94 + ich
    return ans

for x in range(93, -1, -1):
    n1 = conv('450x93', x)
    n2 = conv('879x37', x)
    n3 = conv('285x56', x)
    n = n1 + n2 + n3
    if n % 93 == 0:
        print(n // 31)
        break



