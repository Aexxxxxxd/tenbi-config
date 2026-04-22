total_sum = 0
for x in range(3000):
    value = 12 * (19 ** 12) + 9 * (19 ** 9) + 5 * (19 ** 5) - x # Формат умножение > число > степень
    digits = []
    temp = value
    while temp > 0:
        digits.append(temp % 19) # 19 = СИс
        temp //= 19
    digits.reverse()
    zero_count = 0
    for d in digits:
        if d == 0:
            zero_count += 1
    if zero_count % 2 == 0:
        total_sum += x
print(total_sum)


