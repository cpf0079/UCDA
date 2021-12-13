import xlrd


def get_label(address, num_samples):
    excel = xlrd.open_workbook(address)

    sheet = excel.sheet_by_name('Sheet1')

    t2 = []
    t3 = []
    # for i in range(1, 1014):
    for i in range(1, num_samples+1):
        t1 = []
        for j in range(1, 6):
            # t1.append(int(sheet.cell_value(i-1, j-1))/30)
            t1.append(sheet.cell_value(i - 1, j - 1))
        t2.append(t1)
        t3.append(i)

    label = dict(zip(t3, t2))
    return label


if __name__ == '__main__':
    # address = r'D:\cpf\repository\VQA\pytorch_version\distribution.xlsx'
    address = r'D:\cpf\repository\cpfcpf\excelwrite.xls'
    label = get_label(address, 5)
    print(label[1])
