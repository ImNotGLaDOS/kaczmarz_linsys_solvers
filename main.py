from get_results import save_results

print("""
0 - generate_random_system
1 - generate_well_conditioned_system
2 - generate_vandermond_system

Введите номер типа генерации матрицы системы уравнений
""")

generation_type = int(input())

print('Введите число уравнений\n')

m = int(input())

print('Введите число неизвестных\n')

n = int(input())

print('Введите число итераций в эксперименте\n')

max_iter = int(input())

save_results(generation_type, n, m, max_iter)
