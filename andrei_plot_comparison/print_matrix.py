import numpy as np

print('Ввведите id эксперимента')
experiment_id = input()


def print_system():
    data = np.load(f'graphics/{experiment_id}/system.npz')
    txt_path = f'graphics/{experiment_id}/system_info.txt'

    with open(txt_path, 'w') as f:
        for key in data.files:
            array = data[key]
            f.write(f"▶ {key}:\n\n")
            np.savetxt(f, array, fmt='%.6f', comments='  ', delimiter='\t')
            f.write("\n" + "-" * 40 + "\n\n")


print_system()
