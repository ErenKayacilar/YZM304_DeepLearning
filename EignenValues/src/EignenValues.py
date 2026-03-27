import numpy as np

def get_dimensions(matrix):

    return [len(matrix), len(matrix[0])]


def find_determinant(matrix, excluded=1):

    dimensions = get_dimensions(matrix)

    if dimensions == [2, 2]:
        return excluded * ((matrix[0][0] * matrix[1][1]) - (matrix[0][1] * matrix[1][0]))

    new_matrices = []
    excluded_vals = []
    exclude_row = 0

    for exclude_column in range(dimensions[1]):
        tmp = []
        excluded_vals.append(matrix[exclude_row][exclude_column])
        for row in range(1, dimensions[0]):
            tmp_row = []
            for column in range(dimensions[1]):
                if column != exclude_column:
                    tmp_row.append(matrix[row][column])
            tmp.append(tmp_row)
        new_matrices.append(tmp)


    determinants = [find_determinant(new_matrices[j], excluded_vals[j])
                    for j in range(len(new_matrices))]

    determinant = 0
    for i in range(len(determinants)):
        determinant += ((-1) ** i) * determinants[i]

    return determinant


def list_multiply(list1, list2):

    result = [0 for _ in range(len(list1) + len(list2) - 1)]
    for i in range(len(list1)):
        for j in range(len(list2)):
            result[i + j] += list1[i] * list2[j]
    return result


def list_add(list1, list2, sub=1):

    return [i + (sub * j) for i, j in zip(list1, list2)]


def determinant_equation(matrix, excluded=[1, 0]):

    dimensions = get_dimensions(matrix)

    # 2x2 temel durum
    if dimensions == [2, 2]:
        tmp = list_add(
            list_multiply(matrix[0][0], matrix[1][1]),
            list_multiply(matrix[0][1], matrix[1][0]),
            sub=-1
        )
        return list_multiply(tmp, excluded)

    # Büyük matrisler için Laplace açılımı
    new_matrices = []
    excluded_list = []
    exclude_row = 0

    for exclude_column in range(dimensions[1]):
        tmp = []
        excluded_list.append(matrix[exclude_row][exclude_column])
        for row in range(1, dimensions[0]):
            tmp_row = []
            for column in range(dimensions[1]):
                if column != exclude_column:
                    tmp_row.append(matrix[row][column])
            tmp.append(tmp_row)
        new_matrices.append(tmp)

    # Alt-matris polinom denklemlerini hesapla
    determinant_equations = [determinant_equation(new_matrices[j], excluded_list[j])
                             for j in range(len(new_matrices))]

    # Polinomları topla (eleman bazında)
    dt_equation = [sum(i) for i in zip(*determinant_equations)]
    return dt_equation


def identity_matrix(dimensions):

    matrix = [[0 for _ in range(dimensions[1])] for _ in range(dimensions[0])]
    for i in range(dimensions[0]):
        matrix[i][i] = 1
    return matrix


def characteristic_equation(matrix):

    dimensions = get_dimensions(matrix)
    return [[[a, -b] for a, b in zip(row, id_row)]
            for row, id_row in zip(matrix, identity_matrix(dimensions))]


def find_eigenvalues(matrix):

    char_matrix = characteristic_equation(matrix)
    dt_equation = determinant_equation(char_matrix)
    # np.roots yüksek dereceden düşük dereceye bekler → ters çevir
    return np.roots(dt_equation[::-1])


# ─────────────────────────────────────────────
# ANA PROGRAM
# ─────────────────────────────────────────────

if __name__ == "__main__":

    print("=" * 60)
    print("  ÖZDEĞERLERİN HESAPLANMASI: MANUEL vs NUMPY")
    print("=" * 60)

    # Test matrisi (referans çalışma ile aynı)
    A = [
        [6, 1, -1],
        [0, 7, 0],
        [3, -1, 2]
    ]
    A_np = np.array(A, dtype=float)

    print("\nMatris A:")
    for row in A:
        print(" ", row)

    # ── 1. MANUEL HESAPLAMA ──────────────────
    print("\n" + "-" * 40)
    print("1) MANUEL HESAPLAMA (Karakteristik Polinom Yöntemi)")
    print("-" * 40)

    char_mat = characteristic_equation(A)
    dt_eq = determinant_equation(char_mat)

    print("\nKarakteristik polinom katsayıları (sabitten yüksek dereceye):")
    print(" ", dt_eq)

    # Polinom ifadesini yazdır
    polinom_str = "det(A - λI) = "
    terimler = []
    for i, c in enumerate(dt_eq):
        if abs(c) < 1e-10:
            continue
        c_fmt = f"{c:+.4g}"
        if i == 0:
            terimler.append(f"{c:.4g}")
        elif i == 1:
            terimler.append(f"{c_fmt}λ")
        else:
            terimler.append(f"{c_fmt}λ^{i}")
    print(" ", polinom_str + " + ".join(terimler).replace("+ -", "- "))

    eigenvalues_manual = find_eigenvalues(A)
    print("\nManuel Özdeğerler:")
    for i, val in enumerate(sorted(eigenvalues_manual, key=lambda x: x.real), 1):
        print(f"  λ{i} = {val.real:.6f}" + (f" + {val.imag:.6f}j" if abs(val.imag) > 1e-10 else ""))

    # ── 2. NUMPY HESAPLAMA ───────────────────
    print("\n" + "-" * 40)
    print("2) NUMPY np.linalg.eig() FONKSİYONU")
    print("-" * 40)

    eigenvalues_np, eigenvectors_np = np.linalg.eig(A_np)

    print("\nNumPy Özdeğerler:")
    for i, val in enumerate(sorted(eigenvalues_np, key=lambda x: x.real), 1):
        print(f"  λ{i} = {val.real:.6f}" + (f" + {val.imag:.6f}j" if abs(val.imag) > 1e-10 else ""))

    print("\nNumPy Özvektörler (sütunlar):")
    print(np.round(eigenvectors_np, 6))

    # ── 3. KARŞILAŞTIRMA ─────────────────────
    print("\n" + "-" * 40)
    print("3) KARŞILAŞTIRMA")
    print("-" * 40)

    ev_manual_sorted = sorted(eigenvalues_manual, key=lambda x: x.real)
    ev_numpy_sorted = sorted(eigenvalues_np, key=lambda x: x.real)

    print(f"\n{'Özdeğer':<10} {'Manuel':>18} {'NumPy':>18} {'Fark':>18}")
    print(" " + "-" * 66)
    for i, (m, n) in enumerate(zip(ev_manual_sorted, ev_numpy_sorted), 1):
        fark = abs(m - n)
        print(f"  λ{i}       {m.real:>18.6f} {n.real:>18.6f} {fark.real:>18.2e}")

    # Doğrulama: maksimum fark
    max_fark = max(abs(m - n) for m, n in zip(ev_manual_sorted, ev_numpy_sorted))
    print(f"\n  Maksimum mutlak fark: {max_fark.real:.2e}")

    if max_fark.real < 1e-8:
        print("  ✓ Sonuçlar eşleşiyor — manuel implementasyon doğru!")
    else:
        print("  ✗ Sonuçlar arasında anlamlı fark var, kontrol gerekli.")

    print("\n" + "=" * 60)