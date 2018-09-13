# 단순하게 y = 5x
def linear(value):
    return value*5

if __name__=="__main__":
    value1 = 1
    value2 = 3

    # re_value는 함수 결과값
    re_value1 = linear(value1)
    re_value2 = linear(value2)

    re_value3 = linear(value1 + value2)

    if re_value1 + re_value2 == re_value3:
        print("선형 함수입니다")