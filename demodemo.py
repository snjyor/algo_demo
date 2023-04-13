import json


def algo():
    """
    冒泡排序
    :return:
    """
    # 从键盘输入数字
    num = input("请输入数字：")
    # 将输入的数字转换成列表
    num_list = list(num)
    # 将列表中的元素转换成int类型
    for i in range(len(num_list)):
        # 将列表中的元素转换成int类型
        num_list[i] = int(num_list[i])
    # 冒泡排序
    for i in range(len(num_list)):
        for j in range(len(num_list) - i - 1):
            # 交换位置
            if num_list[j] > num_list[j + 1]:
                num_list[j], num_list[j + 1] = num_list[j + 1], num_list[j]
    # 将列表转换成字符串
    num = ''.join(str(i) for i in num_list)
    # 输出排序后的数字
    print("排序后的数字为：", num)


def quick_sort(num_list):
    """
    快速排序
    :param num_list:
    :return:
    """
    if len(num_list) < 2:
        return num_list
    else:
        pivot = num_list[0]
        # 由所有小于基准值的元素组成的子列表
        less = [i for i in num_list[1:] if i <= pivot]
        # 由所有大于基准值的元素组成的子列表
        greater = [i for i in num_list[1:] if i > pivot]
        # 递归调用
        return quick_sort(less) + [pivot] + quick_sort(greater)


def rank_data():
    """
    快速排序
    :return:
    """
    # 从键盘输入数字
    num = input("请输入数字：")
    # 将输入的数字转换成列表
    num_list = list(num)
    # 将列表中的元素转换成int类型
    for i in range(len(num_list)):
        num_list[i] = int(num_list[i])
    # 快速排序
    num_list = quick_sort(num_list)
    # 将列表转换成字符串
    num = ''.join(str(i) for i in num_list)
    # 输出排序后的数字
    print("排序后的数字为：", num)


def get_data_from_akshare():
    """
    从akshare获取指数历史数据
    :return:
    """
    # 导入akshare
    import akshare as ak
    # 获取指数历史数据
    index_hfq_df = ak.index_hfq_daily(symbol="sh000001")
    # 输出数据
    print(index_hfq_df)


if __name__ == '__main__':
    # algo()
    # rank_data()
    get_data_from_akshare()
